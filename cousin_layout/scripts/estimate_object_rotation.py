import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..models.feature_matcher import FeatureMatcher
from ..utils.processing_utils import compute_bbox_from_mask
from torchvision.ops.boxes import _box_xyxy_to_cxcywh

_FM_CACHE = {}


def get_feature_matcher(
    device: str = "cuda",
    encoder: str = "DinoV2Encoder",
    encoder_kwargs: dict | None = None,
) -> FeatureMatcher:
    """
    Process-local singleton cache for FeatureMatcher.
    This avoids re-loading DinoV2 / text encoder / GroundedSAM on every call.
    """
    key = (str(device), str(encoder), json.dumps(encoder_kwargs or {}, sort_keys=True))
    fm = _FM_CACHE.get(key)
    if fm is not None:
        return fm
    fm = FeatureMatcher(
        encoder=encoder,
        encoder_kwargs={} if encoder_kwargs is None else dict(encoder_kwargs),
        device=device,
        verbose=False,
    )
    _FM_CACHE[key] = fm
    return fm


def preload_rotation_models(
    device: str = "cuda",
    encoder: str = "DinoV2Encoder",
    encoder_kwargs: dict | None = None,
) -> None:
    """
    Explicit warmup hook to load rotation-estimation models once.
    """
    _ = get_feature_matcher(device=device, encoder=encoder, encoder_kwargs=encoder_kwargs)


def rotation_matrix_from_index(
    idx: int,
    step_degrees: float = 22.5,
    mode: str = "camera_orbit",
) -> np.ndarray:
    """
    Given a snapshot index idx (0, 1, 2, ...), compute the rotation matrix
    relative to the default object coordinate system using the capture rules
    in snapshot_generation.

    - The camera orbits the object by +angle around the y axis.
    - Equivalently, the object rotates by -angle around the y axis relative
      to the camera.
    """
    angle_deg = idx * step_degrees
    if mode == "camera_orbit":
        theta = -np.deg2rad(angle_deg)
    elif mode == "object_spin":
        theta = np.deg2rad(angle_deg)
    else:
        raise ValueError(f"Invalid mode: {mode}, expected 'camera_orbit' or 'object_spin'.")

    c, s = np.cos(theta), np.sin(theta)
    # Rotation around the y axis.
    R = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )
    return R


def estimate_rotation_for_object(
    step1_output_path: str,
    object_name: str,
    asset_dir: str,
    step_degrees: float = 22.5,
    device: str = "cuda",
    encoder: str = "DinoV2Encoder",
    rotation_mode: str = "camera_orbit",
    normalize_crops: bool = True,
    normalize_target_size: int = 448,
    normalize_margin_frac: float = 0.15,
    compute_transform_matrix: bool = True,
) -> dict:
    step1_output_path = os.path.abspath(os.path.expanduser(step1_output_path))
    asset_dir = os.path.abspath(os.path.expanduser(asset_dir))
    obj_name = object_name

    if not os.path.isfile(step1_output_path):
        raise FileNotFoundError(f"Step1 output not found: {step1_output_path}")
    if not os.path.isdir(asset_dir):
        raise NotADirectoryError(f"Asset directory not found: {asset_dir}")

    snapshot_dir = os.path.join(asset_dir, "snapshot")
    if not os.path.isdir(snapshot_dir):
        raise NotADirectoryError(f"Snapshot subdirectory not found in asset directory: {snapshot_dir}")

    # 1. Read Step 1 output to obtain input_rgb, segmentation_dir, boxes,
    # logits, and phrases.
    with open(step1_output_path, "r") as f:
        step1_info = json.load(f)
    with open(step1_info["detected_categories"], "r") as f:
        det = json.load(f)

    names = det["names"]
    if obj_name not in names:
        raise ValueError(f"object-name '{obj_name}' is not in Step 1 names. Valid values: {names}")

    idx = names.index(obj_name)
    input_rgb_path = step1_info["input_rgb"]
    seg_dir = det["segmentation_dir"]

    boxes = torch.tensor(det["boxes"])
    logits = torch.tensor(det["logits"])
    phrases = det["phrases"]

    logit = logits[idx]
    phrase = phrases[idx]

    # Read the object's non-projected mask, consistent with Step 2
    # (_nonprojected_mask.png).
    obj_mask_fpath = os.path.join(seg_dir, f"{obj_name}_nonprojected_mask.png")
    if not os.path.isfile(obj_mask_fpath):
        raise FileNotFoundError(f"Mask not found for object '{obj_name}': {obj_mask_fpath}")
    mask = np.array(Image.open(obj_mask_fpath))
    obj_masks = mask.reshape(1, 1, *mask.shape)

    # Recompute the bounding box from the mask, as in Step 2.
    bbox_xyxy = compute_bbox_from_mask(obj_mask_fpath)
    bboxes = _box_xyxy_to_cxcywh(torch.tensor(bbox_xyxy)).unsqueeze(0)

    # 2. Build snapshot candidates.
    snap_paths = sorted(
        str(p)
        for p in Path(snapshot_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not snap_paths:
        raise RuntimeError(f"No snapshot images found under {snapshot_dir}")

    # 3. Initialize FeatureMatcher using the Step 2 bbox + mask pipeline,
    # while ignoring articulated-object special handling.
    fm = get_feature_matcher(device=device, encoder=encoder, encoder_kwargs={})

    category = phrase.replace(" ", "_")
    tmp_save_dir = os.path.join(os.path.dirname(step1_output_path), "rotation_tmp")
    os.makedirs(tmp_save_dir, exist_ok=True)

    results = fm.find_nearest_neighbor_candidates(
        input_category=category,
        input_img_fpath=input_rgb_path,
        candidate_imgs_fdirs=None,
        candidate_imgs=snap_paths,
        candidate_filter=None,
        n_candidates=1,
        save_dir=tmp_save_dir,
        visualize_resolution=(640, 480),
        boxes=bboxes,
        logits=logit.unsqueeze(0),
        phrases=[phrase],
        obj_masks=obj_masks,
        save_prefix=f"{obj_name}_rotation",
        remove_background=True,
        normalize_crops=bool(normalize_crops),
        normalize_target_size=int(normalize_target_size),
        normalize_margin_frac=float(normalize_margin_frac),
    )

    best_path = results["candidates"][0]
    best_score = None
    if isinstance(results, dict):
        scores = results.get("candidate_scores", None)
        if isinstance(scores, list) and len(scores) > 0:
            try:
                best_score = float(scores[0])
            except Exception:
                best_score = None
    basename = os.path.basename(best_path)
    stem, _ = os.path.splitext(basename)
    if not stem.isdigit():
        raise ValueError(
            f"The most similar snapshot filename is not numeric: '{basename}'. "
            "Cannot infer the angle from the filename."
        )
    best_idx = int(stem)

    out: dict = {
        "object_name": obj_name,
        "best_snapshot_path": best_path,
        "best_snapshot_file": basename,
        "best_snapshot_index": int(best_idx),
        "best_match_score": best_score,
    }
    if compute_transform_matrix:
        R = rotation_matrix_from_index(best_idx, step_degrees=step_degrees, mode=rotation_mode)
        out["transform_matrix_3x3"] = R.tolist()
    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Use ACDC Step 1 masks and bounding boxes to compare the input "
            "against snapshots of a specified asset with FeatureMatcher "
            "(ignoring articulated-object special handling), then output a "
            "rotation matrix relative to the default object coordinate system."
        )
    )
    parser.add_argument(
        "--step1-output",
        type=str,
        required=True,
        help=(
            "Path to the Step 1 step_1_output_info.json, for example "
            "cousin_layout/desk_layout/1777367355.2785048/step_1_output_info.json"
        ),
    )
    parser.add_argument(
        "--object-name",
        type=str,
        required=True,
        help=(
            "Object name to process. It must be one of "
            "detected_categories['names'] from Step 1, such as desk_0 or chair_1."
        ),
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        required=True,
        help=(
            "Asset root directory, for example our_assets/actor/laptop/1. "
            "The script automatically locates its snapshot subdirectory."
        ),
    )
    parser.add_argument(
        "--step-degrees",
        type=float,
        default=22.5,
        help="Angular interval between snapshots (degrees). Default: 22.5, "
        "matching snapshot_generation.py.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device used for feature matching. Default: cuda; cpu is supported.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="DinoV2Encoder",
        help="Vision encoder for similarity comparison. Default: DinoV2Encoder; "
        "CLIPEncoder is also supported.",
    )
    parser.add_argument(
        "--rotation-mode",
        type=str,
        default="camera_orbit",
        choices=["camera_orbit", "object_spin"],
        help="Snapshot generation mode: camera_orbit (camera circles the object) "
        "or object_spin (fixed camera, rotating object).",
    )

    args = parser.parse_args()

    out = estimate_rotation_for_object(
        step1_output_path=args.step1_output,
        object_name=args.object_name,
        asset_dir=args.asset_dir,
        step_degrees=float(args.step_degrees),
        device=args.device,
        encoder=args.encoder,
        rotation_mode=args.rotation_mode,
    )

    np.set_printoptions(precision=6, suppress=True)
    print("Object name:", out["object_name"])
    print("Most similar snapshot file:", out["best_snapshot_file"])
    print("Corresponding snapshot index:", out["best_snapshot_index"])
    print("Rotation matrix (from the default object coordinate system to this orientation):")
    print(np.array(out["transform_matrix_3x3"], dtype=np.float64))


if __name__ == "__main__":
    main()


