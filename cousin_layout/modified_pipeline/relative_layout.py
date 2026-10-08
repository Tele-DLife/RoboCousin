"""
Extract 2D relative object layout and map to a user-defined region.

Given a region with size (a, b), this module maps each detected object to:
    - relative position in [0, 1] x [0, 1]
    - absolute position in region coordinates [0, a] x [0, b]

Supported inputs:
1) Step-1 output (position-only geometry from depth + masks).
2) Step-1 + Step-2 outputs (math-only reorientation; no Omni launch).
3) Optional compatibility mode: Step-3 scene info.
4) End-to-end: run modified Step-1 from input image, then geometry extraction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import yaml

from .. import constants
from ..models.feature_matcher import FeatureMatcher
from .extraction import RealWorldExtractor
from ..utils.processing_utils import (
    compute_point_cloud_from_depth,
    get_reproject_offset,
    unprocess_depth_linear,
)
from ..utils.scene_utils import (
    compute_object_z_offset,
    compute_relative_cam_pose_from,
    slice_points_world_xyz_for_support_footprint,
)
from ..utils import transform_utils as T


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _normalize_cxcywh(box: Any, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    """
    Step-1 stores GroundingDINO boxes as cxcywh. In most runs they are normalized to [0, 1].
    If values are in pixels, this function normalizes them.
    """
    arr = np.asarray(box, dtype=float).reshape(-1)
    if arr.size != 4:
        return 0.5, 0.5, 0.0, 0.0

    cx, cy, w, h = [float(x) for x in arr.tolist()]
    looks_normalized = max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5

    if not looks_normalized:
        image_w = max(int(image_w), 1)
        image_h = max(int(image_h), 1)
        cx /= image_w
        w /= image_w
        cy /= image_h
        h /= image_h

    return _clip01(cx), _clip01(cy), _clip01(w), _clip01(h)


def _robust_span(values: np.ndarray, lo_q: float = 5.0, hi_q: float = 95.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    lo = np.percentile(arr, lo_q)
    hi = np.percentile(arr, hi_q)
    return float(max(hi - lo, 0.0))


def _fit_plane_svd(points_world: np.ndarray) -> tuple[np.ndarray, float] | None:
    """
    Fit a plane ax+by+cz+d=0 via SVD.

    Returns:
        (normal, d) with ||normal||=1, or None if insufficient points.
    """
    pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None
    centroid = np.mean(pts, axis=0)
    A = pts - centroid
    _u, _s, vh = np.linalg.svd(A, full_matrices=False)
    n = vh[-1, :]
    n_norm = float(np.linalg.norm(n))
    if (not np.isfinite(n_norm)) or n_norm < 1e-9:
        return None
    n = n / n_norm
    # Prefer upward-facing normal for desk/floor support planes
    if float(n[2]) < 0.0:
        n = -n
    d = -float(np.dot(n, centroid))
    return n.astype(float), float(d)


def _sample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] <= max_points:
        return pts
    idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
    return pts[idx]


def _filter_points_near_plane(
    pts_world: np.ndarray,
    plane: tuple[np.ndarray, float],
    *,
    keep_low_quantile: float = 10.0,
    extra_margin_m: float = 0.01,
) -> np.ndarray:
    """
    Keep points close to a support plane (e.g. desk surface) to avoid tall vertical
    structures (e.g. laptop screen) polluting top-down footprint.
    """
    pts = np.asarray(pts_world, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts
    n, d = plane
    dist = np.abs(pts @ n + d)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return pts
    q = float(np.percentile(dist, keep_low_quantile))
    thr = q + float(extra_margin_m)
    keep = np.abs(pts @ n + d) <= thr
    return pts[keep]


def _mask_world_points(
    mask_path: str,
    pc_cam: np.ndarray,
    cam_to_world_tf: np.ndarray,
) -> np.ndarray:
    if not os.path.exists(mask_path):
        return np.zeros((0, 3), dtype=float)
    mask = np.array(Image.open(mask_path))
    return _mask_world_points_from_mask(mask=mask, pc_cam=pc_cam, cam_to_world_tf=cam_to_world_tf)


def _mask_world_points_from_mask(
    mask: np.ndarray,
    pc_cam: np.ndarray,
    cam_to_world_tf: np.ndarray,
) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim >= 3:
        mask = mask[..., 0]
    idx = np.flatnonzero(mask.reshape(-1) > 0)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=float)
    pts_cam = pc_cam.reshape(-1, 3)[idx]
    finite = np.isfinite(pts_cam).all(axis=1)
    pts_cam = pts_cam[finite]
    if pts_cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)
    pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=float)], axis=1)
    pts_world = (pts_h @ cam_to_world_tf.T)[:, :3]
    return pts_world


def _bbox_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1) + 1)
    ih = max(0, min(ay2, by2) - max(ay1, by1) + 1)
    inter = float(iw * ih)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1))
    area_b = float(max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1))
    union = max(area_a + area_b - inter, 1e-9)
    return float(inter / union)


def _gate_mask_by_detection_box(
    mask: np.ndarray,
    box_norm_cxcywh: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    *,
    max_mask_to_box_area_ratio: float = 1.75,
    min_mask_box_iou: float = 0.20,
    expand_ratio: float = 0.12,
) -> np.ndarray:
    """
    Lightweight quality gate:
    if mask bbox is much larger than detection box or has tiny overlap with it,
    keep only mask pixels within an expanded detection box.
    """
    m = np.asarray(mask)
    if m.ndim >= 3:
        m = m[..., 0]
    ys, xs = np.nonzero(m > 0)
    if xs.size == 0:
        return m

    cx, cy, bw, bh = box_norm_cxcywh
    x1 = int(np.clip(round((cx - bw / 2.0) * (image_w - 1)), 0, image_w - 1))
    x2 = int(np.clip(round((cx + bw / 2.0) * (image_w - 1)), 0, image_w - 1))
    y1 = int(np.clip(round((cy - bh / 2.0) * (image_h - 1)), 0, image_h - 1))
    y2 = int(np.clip(round((cy + bh / 2.0) * (image_h - 1)), 0, image_h - 1))
    if x2 <= x1 or y2 <= y1:
        return m

    mx1, mx2 = int(xs.min()), int(xs.max())
    my1, my2 = int(ys.min()), int(ys.max())
    mask_bbox_area = float((mx2 - mx1 + 1) * (my2 - my1 + 1))
    det_bbox_area = float((x2 - x1 + 1) * (y2 - y1 + 1))
    ratio = mask_bbox_area / max(det_bbox_area, 1.0)
    iou = _bbox_iou_xyxy((mx1, my1, mx2, my2), (x1, y1, x2, y2))
    if (ratio <= float(max_mask_to_box_area_ratio)) and (iou >= float(min_mask_box_iou)):
        return m

    pad_x = int(round(expand_ratio * max(1, x2 - x1 + 1)))
    pad_y = int(round(expand_ratio * max(1, y2 - y1 + 1)))
    gx1 = int(np.clip(x1 - pad_x, 0, image_w - 1))
    gx2 = int(np.clip(x2 + pad_x, 0, image_w - 1))
    gy1 = int(np.clip(y1 - pad_y, 0, image_h - 1))
    gy2 = int(np.clip(y2 + pad_y, 0, image_h - 1))

    gated = np.zeros_like(m)
    gated[gy1 : gy2 + 1, gx1 : gx2 + 1] = m[gy1 : gy2 + 1, gx1 : gx2 + 1]

    orig_count = int(xs.size)
    kept_count = int(np.count_nonzero(gated > 0))
    min_keep = max(80, int(0.10 * orig_count))
    if kept_count < min_keep:
        return np.zeros_like(m)
    return gated


def _safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "obj"


def _build_step1_support_graph(
    obj_records: list[dict[str, Any]],
    verbose: bool = False,
    footprint_slice_frac: float | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build an ACDC-like vertical support graph from Step-1 world point clouds.

    Output schema mirrors Step-3 `scene_<k>_graph.json`:
      {
        "floor": {"objOnTop": [...], "objBeneath": None, "mount": {"floor": True, "wall": False}},
        "<obj>": {"objOnTop": [...], "objBeneath": "<support>", "mount": {...}},
        ...
      }
    """
    # Construct bbox info compatible with `digital_cousins.utils.scene_utils.compute_object_z_offset`
    all_obj_bbox_info: dict[str, dict[str, Any]] = {}
    for rec in obj_records:
        name = str(rec.get("name", ""))
        label = str(rec.get("label", "") or "").strip().lower()
        # Force full footprint for broad support surfaces, regardless of a global slice fraction.
        # This prevents desk/table footprints from collapsing to legs / narrow bases when using
        # `support_footprint_slice_frac`, which would otherwise misclassify objects resting on
        # the surface as being on the floor.
        is_support_surface = (name.strip().lower() == "floor") or (label == "desk") or (label == "table")
        pts = np.asarray(rec.get("points_world_xyz", np.zeros((0, 3), dtype=float)), dtype=float)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if name == "" or pts.shape[0] == 0:
            continue

        lo = np.min(pts, axis=0)
        hi = np.max(pts, axis=0)
        cxcycz = np.median(pts, axis=0)

        xmin, ymin, zmin = [float(x) for x in lo.tolist()]
        xmax, ymax, zmax = [float(x) for x in hi.tolist()]

        bbox_bottom = np.array(
            [
                [xmin, ymin, zmin],
                [xmin, ymax, zmin],
                [xmax, ymax, zmin],
                [xmax, ymin, zmin],
            ],
            dtype=float,
        )
        bbox_top = np.array(
            [
                [xmin, ymin, zmax],
                [xmin, ymax, zmax],
                [xmax, ymax, zmax],
                [xmax, ymin, zmax],
            ],
            dtype=float,
        )

        all_obj_bbox_info[name] = {
            "lower": np.array([xmin, ymin, zmin], dtype=float),
            "upper": np.array([xmax, ymax, zmax], dtype=float),
            "center": np.array([float(cxcycz[0]), float(cxcycz[1]), float(cxcycz[2])], dtype=float),
            # Minimal fields used by compute_object_z_offset:
            "bbox_bottom_in_desired_frame": bbox_bottom,
            "bbox_top_in_desired_frame": bbox_top,
            "desired_frame_to_world": np.eye(4, dtype=float),
            # Optional: provide raw points so z-sliced footprint can be used.
            "points_world_xyz": pts,
            # Keep thresholds sane (treat as non-articulated by default)
            "articulated": bool(rec.get("articulated", False)),
            # Per-object override used inside compute_object_z_offset polygon builder.
            "support_surface": bool(is_support_surface),
        }

    sorted_z_obj_bbox_info = dict(sorted(all_obj_bbox_info.items(), key=lambda x: float(x[1]["lower"][2])))

    scene_graph_info: dict[str, dict[str, Any]] = {
        "__meta__": {
            "source": "step1_geometry",
            "support_footprint_slice_frac": None if footprint_slice_frac is None else float(footprint_slice_frac),
        },
        "floor": {"objOnTop": [], "objBeneath": None, "mount": {"floor": True, "wall": False}},
    }

    for name in sorted_z_obj_bbox_info:
        obj_name_beneath, _z_offset = compute_object_z_offset(
            target_obj_name=name,
            sorted_obj_bbox_info=sorted_z_obj_bbox_info,
            verbose=verbose,
            footprint_slice_frac=(1.0 if footprint_slice_frac is None else float(footprint_slice_frac)),
        )

        if name not in scene_graph_info:
            scene_graph_info[name] = {"objOnTop": [], "objBeneath": obj_name_beneath, "mount": {"floor": True, "wall": None}}
        else:
            scene_graph_info[name]["objBeneath"] = obj_name_beneath

        if obj_name_beneath not in scene_graph_info:
            scene_graph_info[obj_name_beneath] = {"objOnTop": [name], "objBeneath": None, "mount": {"floor": True, "wall": None}}
        else:
            scene_graph_info[obj_name_beneath].setdefault("objOnTop", [])
            scene_graph_info[obj_name_beneath]["objOnTop"].append(name)

    return scene_graph_info


def _collect_descendants(scene_graph: dict[str, dict[str, Any]], roots: set[str]) -> set[str]:
    out: set[str] = set()
    q = list(roots)
    seen = set(roots)
    while q:
        cur = q.pop(0)
        children = scene_graph.get(cur, {}).get("objOnTop", []) or []
        for ch in children:
            if ch in seen:
                continue
            seen.add(ch)
            out.add(ch)
            q.append(ch)
    return out


def _extract_subgraph(scene_graph: dict[str, dict[str, Any]], nodes: set[str]) -> dict[str, dict[str, Any]]:
    """
    Return an induced subgraph with only @nodes. `objOnTop` lists are filtered to stay within @nodes.
    """
    out: dict[str, dict[str, Any]] = {}
    for n in nodes:
        info = scene_graph.get(n, {})
        children = [c for c in (info.get("objOnTop", []) or []) if c in nodes]
        beneath = info.get("objBeneath", None)
        out[n] = {
            "objOnTop": children,
            "objBeneath": beneath if beneath in nodes else None,
            "mount": info.get("mount", None),
        }
    return out


def _collect_ontop_levels(
    scene_graph: dict[str, dict[str, Any]],
    roots: set[str],
    allowed: set[str],
) -> list[dict[str, Any]]:
    """
    Return level-by-level ontop relations from @roots, e.g.
      depth=1: (desk_0 -> book_2), (desk_0 -> laptop_0)
      depth=2: (book_2 -> book_1)
      depth=3: (book_1 -> book_0)
    Only traverses nodes in @allowed.
    """
    levels: list[dict[str, Any]] = []
    frontier = sorted([r for r in roots if r in allowed])
    seen: set[str] = set(frontier)
    depth = 0
    while frontier:
        depth += 1
        next_frontier: list[str] = []
        rels: list[dict[str, str]] = []
        for parent in frontier:
            children = [c for c in (scene_graph.get(parent, {}).get("objOnTop", []) or []) if c in allowed]
            for child in children:
                rels.append({"parent": parent, "child": child})
                if child not in seen:
                    seen.add(child)
                    next_frontier.append(child)
        if rels:
            levels.append({"depth": depth, "relations": rels})
        frontier = next_frontier
    return levels


def _source_box_bounds_norm(box_cxcywh: Any) -> tuple[float, float, float, float]:
    arr = np.asarray(box_cxcywh, dtype=float).reshape(-1)
    if arr.size != 4:
        return 0.5, 0.5, 0.5, 0.5
    cx, cy, w, h = [float(x) for x in arr.tolist()]
    half_w = max(w, 0.0) / 2.0
    half_h = max(h, 0.0) / 2.0
    return (
        _clip01(cx - half_w),
        _clip01(cy - half_h),
        _clip01(cx + half_w),
        _clip01(cy + half_h),
    )


def _source_box_child_coverage(child_box: Any, parent_box: Any) -> float:
    child_x1, child_y1, child_x2, child_y2 = _source_box_bounds_norm(child_box)
    parent_x1, parent_y1, parent_x2, parent_y2 = _source_box_bounds_norm(parent_box)
    iw = max(0.0, min(child_x2, parent_x2) - max(child_x1, parent_x1))
    ih = max(0.0, min(child_y2, parent_y2) - max(child_y1, parent_y1))
    inter = iw * ih
    child_area = max((child_x2 - child_x1) * (child_y2 - child_y1), 1e-9)
    return float(inter / child_area)


def _robust_bounds(values: np.ndarray, lo_q: float = 5.0, hi_q: float = 95.0) -> tuple[float, float] | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo = float(np.percentile(arr, lo_q))
    hi = float(np.percentile(arr, hi_q))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)):
        return None
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _parent_surface_points_near_child_bottom(parent_pts: np.ndarray, child_pts: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent_pts, dtype=float).reshape(-1, 3)
    child = np.asarray(child_pts, dtype=float).reshape(-1, 3)
    parent = parent[np.isfinite(parent).all(axis=1)]
    child = child[np.isfinite(child).all(axis=1)]
    if parent.shape[0] == 0 or child.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)

    child_bottom_z = float(np.percentile(child[:, 2], 5.0))
    child_zspan = _robust_span(child[:, 2], lo_q=5.0, hi_q=95.0)
    parent_zspan = _robust_span(parent[:, 2], lo_q=5.0, hi_q=95.0)
    margin = max(0.015, min(0.06, max(child_zspan * 0.75, parent_zspan * 0.08)))
    dz = np.abs(parent[:, 2] - child_bottom_z)
    near = parent[dz <= margin]
    min_pts = max(30, int(0.02 * parent.shape[0]))
    if near.shape[0] >= min_pts:
        return near

    # If the z band is sparse, keep the closest parent points to the child's bottom.
    k = min(parent.shape[0], max(min_pts, int(0.10 * parent.shape[0])))
    if k <= 0:
        return np.zeros((0, 3), dtype=float)
    idx = np.argpartition(dz, kth=k - 1)[:k]
    return parent[idx]


def _adjust_child_xy_by_parent_support(
    obj_records: list[dict[str, Any]],
    scene_graph: dict[str, dict[str, Any]],
    *,
    pivot_xy: np.ndarray,
    rot_xy: np.ndarray,
    min_child_coverage: float = 0.50,
    min_layout_overlap_frac: float = 0.35,
    skip_parent_labels: set[str] | None = None,
) -> None:
    """
    Make ontop children inherit their XY placement from the parent's support area
    when the source 2D boxes agree that the child visually sits on the parent.
    """
    name_to_rec = {str(rec.get("name", "")): rec for rec in obj_records}
    skip_labels = {str(x).strip().lower() for x in (skip_parent_labels or set()) if str(x).strip()}
    for child_name, node in scene_graph.items():
        if child_name == "__meta__" or child_name == "floor":
            continue
        parent_name = str((node or {}).get("objBeneath") or "")
        if parent_name == "" or parent_name == "floor":
            continue
        child_rec = name_to_rec.get(child_name)
        parent_rec = name_to_rec.get(parent_name)
        if child_rec is None or parent_rec is None:
            continue
        parent_label = str(parent_rec.get("label", "") or "").strip().lower()
        if parent_label in skip_labels:
            continue

        coverage = _source_box_child_coverage(
            child_rec.get("source_box_cxcywh"),
            parent_rec.get("source_box_cxcywh"),
        )
        if coverage < float(min_child_coverage):
            continue

        child_box = np.asarray(child_rec.get("source_box_cxcywh", [0.5, 0.5, 0.0, 0.0]), dtype=float).reshape(-1)
        parent_box = np.asarray(parent_rec.get("source_box_cxcywh", [0.5, 0.5, 0.0, 0.0]), dtype=float).reshape(-1)
        if child_box.size != 4 or parent_box.size != 4:
            continue
        child_cx_img, child_cy_img = float(child_box[0]), float(child_box[1])
        parent_cx_img, parent_cy_img = float(parent_box[0]), float(parent_box[1])
        parent_w_img = max(float(parent_box[2]), 1e-6)
        parent_h_img = max(float(parent_box[3]), 1e-6)

        parent_pts = np.asarray(parent_rec.get("points_world_xyz", np.zeros((0, 3), dtype=float)), dtype=float)
        child_pts = np.asarray(child_rec.get("points_world_xyz", np.zeros((0, 3), dtype=float)), dtype=float)
        surface_pts = _parent_surface_points_near_child_bottom(parent_pts=parent_pts, child_pts=child_pts)
        if surface_pts.shape[0] > 0:
            surface_xy = (surface_pts[:, :2] - pivot_xy) @ rot_xy.T + pivot_xy
            bounds_x = _robust_bounds(surface_xy[:, 0], lo_q=5.0, hi_q=95.0)
            bounds_y = _robust_bounds(surface_xy[:, 1], lo_q=5.0, hi_q=95.0)
        else:
            bounds_x = None
            bounds_y = None

        parent_center = np.asarray(parent_rec.get("center_layout_xy", np.zeros(2, dtype=float)), dtype=float).reshape(2)
        parent_fp = np.asarray(parent_rec.get("footprint_layout_xy", np.zeros(2, dtype=float)), dtype=float).reshape(2)
        child_fp = np.asarray(child_rec.get("footprint_layout_xy", np.zeros(2, dtype=float)), dtype=float).reshape(2)
        parent_half_x = max(float(parent_fp[0]) / 2.0, 1e-6)
        parent_half_y = max(float(parent_fp[1]) / 2.0, 1e-6)
        parent_bounds_x = (float(parent_center[0] - parent_half_x), float(parent_center[0] + parent_half_x))
        parent_bounds_y = (float(parent_center[1] - parent_half_y), float(parent_center[1] + parent_half_y))

        if bounds_x is None or (bounds_x[1] - bounds_x[0]) < 1e-6:
            bounds_x = parent_bounds_x
        else:
            ix = (max(float(bounds_x[0]), parent_bounds_x[0]), min(float(bounds_x[1]), parent_bounds_x[1]))
            bounds_x = ix if ix[1] > ix[0] else parent_bounds_x
        if bounds_y is None or (bounds_y[1] - bounds_y[0]) < 1e-6:
            bounds_y = parent_bounds_y
        else:
            iy = (max(float(bounds_y[0]), parent_bounds_y[0]), min(float(bounds_y[1]), parent_bounds_y[1]))
            bounds_y = iy if iy[1] > iy[0] else parent_bounds_y

        support_w = max(bounds_x[1] - bounds_x[0], 1e-6)
        support_h = max(bounds_y[1] - bounds_y[0], 1e-6)
        offset_x = (child_cx_img - parent_cx_img) / parent_w_img
        # Image y grows downward, while layout y grows upward.
        offset_y = -(child_cy_img - parent_cy_img) / parent_h_img
        candidate_x = float(parent_center[0] + offset_x * support_w)
        candidate_y = float(parent_center[1] + offset_y * support_h)
        target_xy = np.array(
            [
                float(np.clip(candidate_x, bounds_x[0], bounds_x[1])),
                float(np.clip(candidate_y, bounds_y[0], bounds_y[1])),
            ],
            dtype=float,
        )

        old_xy = np.asarray(child_rec.get("center_layout_xy", np.zeros(2, dtype=float)), dtype=float).reshape(2)
        coverage_t = np.clip((float(coverage) - float(min_child_coverage)) / max(1.0 - float(min_child_coverage), 1e-6), 0.0, 1.0)
        blend = float(0.35 + 0.25 * coverage_t)
        max_delta = np.array(
            [
                max(float(parent_fp[0]) * 0.35, float(child_fp[0]) * 0.25, 0.01),
                max(float(parent_fp[1]) * 0.35, float(child_fp[1]) * 0.25, 0.01),
            ],
            dtype=float,
        )
        delta = np.clip((target_xy - old_xy) * blend, -max_delta, max_delta)
        new_xy = old_xy + delta
        min_overlap = float(np.clip(min_layout_overlap_frac, 0.0, 1.0))
        required_overlap = np.array(
            [
                max(float(child_fp[0]) * min_overlap, 0.0),
                max(float(child_fp[1]) * min_overlap, 0.0),
            ],
            dtype=float,
        )
        child_half = np.array(
            [
                max(float(child_fp[0]) / 2.0, 0.0),
                max(float(child_fp[1]) / 2.0, 0.0),
            ],
            dtype=float,
        )
        min_overlap_bounds = [
            (
                float(bounds_x[0] + required_overlap[0] - child_half[0]),
                float(bounds_x[1] - required_overlap[0] + child_half[0]),
            ),
            (
                float(bounds_y[0] + required_overlap[1] - child_half[1]),
                float(bounds_y[1] - required_overlap[1] + child_half[1]),
            ),
        ]
        for axis, (lo_allowed, hi_allowed) in enumerate(min_overlap_bounds):
            if hi_allowed < lo_allowed:
                new_xy[axis] = float(parent_center[axis])
            else:
                new_xy[axis] = float(np.clip(new_xy[axis], lo_allowed, hi_allowed))

        child_rec["center_layout_xy_before_parent_adjustment"] = old_xy.copy()
        child_rec["center_layout_xy"] = new_xy
        child_rec["xy_adjusted_by_parent"] = {
            "parent": parent_name,
            "source_box_child_coverage": float(coverage),
            "method": "soft_source_box_offset_with_min_layout_overlap",
            "blend": float(blend),
            "max_delta_layout_xy": [float(max_delta[0]), float(max_delta[1])],
            "min_layout_overlap_frac": float(min_overlap),
            "old_center_layout_xy": [float(old_xy[0]), float(old_xy[1])],
            "target_center_layout_xy": [float(target_xy[0]), float(target_xy[1])],
            "new_center_layout_xy": [float(new_xy[0]), float(new_xy[1])],
            "support_bounds_layout_xy": [
                [float(bounds_x[0]), float(bounds_x[1])],
                [float(bounds_y[0]), float(bounds_y[1])],
            ],
            "min_overlap_center_bounds_layout_xy": [
                [float(min_overlap_bounds[0][0]), float(min_overlap_bounds[0][1])],
                [float(min_overlap_bounds[1][0]), float(min_overlap_bounds[1][1])],
            ],
        }


def _draw_topdown_layout(
    objects: list[dict[str, Any]],
    region_a: float,
    region_b: float,
    output_path: str,
) -> None:
    # Keep aspect ratio of region; clamp to readable image size.
    long_edge_px = 1200
    ratio = region_b / max(region_a, 1e-9)
    if ratio >= 1.0:
        canvas_h = long_edge_px
        canvas_w = int(round(long_edge_px / ratio))
    else:
        canvas_w = long_edge_px
        canvas_h = int(round(long_edge_px * ratio))
    canvas_w = max(canvas_w, 500)
    canvas_h = max(canvas_h, 500)

    img = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (canvas_w - 1, canvas_h - 1)], outline=(40, 40, 40), width=3)

    for obj in objects:
        x_rel = _safe_float(obj.get("region_position_rel", [0.0, 0.0])[0], 0.0)
        y_rel = _safe_float(obj.get("region_position_rel", [0.0, 0.0])[1], 0.0)
        fx_rel = _safe_float(obj.get("region_footprint_rel", [0.0, 0.0])[0], 0.0)
        fy_rel = _safe_float(obj.get("region_footprint_rel", [0.0, 0.0])[1], 0.0)

        x_px = int(round(x_rel * (canvas_w - 1)))
        y_px = int(round((1.0 - y_rel) * (canvas_h - 1)))  # region y-axis points upward

        half_w = int(round(max(fx_rel * canvas_w / 2.0, 5)))
        half_h = int(round(max(fy_rel * canvas_h / 2.0, 5)))
        draw.rectangle(
            [(x_px - half_w, y_px - half_h), (x_px + half_w, y_px + half_h)],
            outline=(64, 128, 255),
            width=2,
        )
        draw.ellipse([(x_px - 4, y_px - 4), (x_px + 4, y_px + 4)], fill=(220, 30, 30))
        draw.text((x_px + 8, y_px - 8), str(obj.get("name", "obj")), fill=(20, 20, 20))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _extract_layout_from_scene_info(
    scene_info: dict[str, Any],
    region_a: float,
    region_b: float,
    save_path: str | None = None,
    visualization_path: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    """
    Extract layout directly from Step-3 `scene_info` (pipeline-consistent poses).
    """
    cam_pose = scene_info["cam_pose"]  # [cam_pos, cam_quat]
    cam_pose_mat = T.pose2mat((np.array(cam_pose[0], dtype=float), np.array(cam_pose[1], dtype=float)))
    objects = scene_info.get("objects", {})

    obj_records: list[dict[str, Any]] = []
    x_lo_list: list[float] = []
    x_hi_list: list[float] = []
    y_lo_list: list[float] = []
    y_hi_list: list[float] = []

    for name, info in objects.items():
        tf_from_cam = np.array(info["tf_from_cam"], dtype=float)
        obj_pose_world = T.pose_in_A_to_pose_in_B(pose_A=tf_from_cam, pose_A_in_B=cam_pose_mat)
        obj_pos, _obj_quat = T.mat2pose(obj_pose_world)
        obj_pos = np.array(obj_pos, dtype=float)

        bbox_extent = np.array(info.get("bbox_extent", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        if bbox_extent.size < 2:
            bbox_extent = np.array([0.0, 0.0, 0.0], dtype=float)
        fx_w = float(max(bbox_extent[0], 0.0))
        fy_w = float(max(bbox_extent[1], 0.0))

        x_lo_list.append(float(obj_pos[0] - fx_w / 2.0))
        x_hi_list.append(float(obj_pos[0] + fx_w / 2.0))
        y_lo_list.append(float(obj_pos[1] - fy_w / 2.0))
        y_hi_list.append(float(obj_pos[1] + fy_w / 2.0))

        obj_records.append(
            {
                "name": name,
                "category": info.get("category", ""),
                "model": info.get("model", ""),
                "center_world_xyz": obj_pos,
                "footprint_world_xy": np.array([fx_w, fy_w], dtype=float),
            }
        )

    if len(obj_records) == 0:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    else:
        x_min = float(min(x_lo_list))
        x_max = float(max(x_hi_list))
        y_min = float(min(y_lo_list))
        y_max = float(max(y_hi_list))

    scene_dx = max(x_max - x_min, 1e-6)
    scene_dy = max(y_max - y_min, 1e-6)

    output_objects: list[dict[str, Any]] = []
    for rec in obj_records:
        cx_w, cy_w, cz_w = rec["center_world_xyz"].tolist()
        fx_w, fy_w = rec["footprint_world_xy"].tolist()

        x_rel = _clip01((cx_w - x_min) / scene_dx)
        y_rel = _clip01((cy_w - y_min) / scene_dy)
        fx_rel = _clip01(fx_w / scene_dx)
        fy_rel = _clip01(fy_w / scene_dy)

        output_objects.append(
            {
                "name": rec["name"],
                "category": rec["category"],
                "model": rec["model"],
                "center_world_xyz": [float(cx_w), float(cy_w), float(cz_w)],
                "region_position_rel": [x_rel, y_rel],
                "region_position_abs": [float(region_a) * x_rel, float(region_b) * y_rel],
                "region_footprint_rel": [fx_rel, fy_rel],
                "region_footprint_abs": [float(region_a) * fx_rel, float(region_b) * fy_rel],
            }
        )

    layout_info: dict[str, Any] = {
        "source": {
            "scene_info_path": os.path.abspath(source_path) if source_path else None,
            "mode": "step3_scene_info",
        },
        "region": {
            "a": float(region_a),
            "b": float(region_b),
            "axes": {"x": "right", "y": "up", "origin": "bottom_left"},
        },
        "projection": {
            "method": "step3_tf_projection",
            "world_xy_bounds": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            },
        },
        "objects": output_objects,
    }

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(layout_info, f, indent=4)

    if visualization_path is not None:
        _draw_topdown_layout(
            objects=output_objects,
            region_a=float(region_a),
            region_b=float(region_b),
            output_path=visualization_path,
        )

    return layout_info


def extract_relative_layout_from_scene_info(
    scene_info_path: str,
    region_a: float,
    region_b: float,
    save_path: str | None = None,
    visualization_path: str | None = None,
) -> dict[str, Any]:
    scene_info = _load_json(scene_info_path)
    return _extract_layout_from_scene_info(
        scene_info=scene_info,
        region_a=region_a,
        region_b=region_b,
        save_path=save_path,
        visualization_path=visualization_path,
        source_path=scene_info_path,
    )


def extract_relative_layout_from_step3_output(
    step_3_output_path: str,
    scene_name: str,
    region_a: float,
    region_b: float,
    save_path: str | None = None,
    visualization_path: str | None = None,
) -> dict[str, Any]:
    step3 = _load_json(step_3_output_path)
    if scene_name not in step3:
        raise ValueError(f"scene_name '{scene_name}' not found in {step_3_output_path}")
    scene_info = step3[scene_name]
    return _extract_layout_from_scene_info(
        scene_info=scene_info,
        region_a=region_a,
        region_b=region_b,
        save_path=save_path,
        visualization_path=visualization_path,
        source_path=step_3_output_path,
    )


def extract_relative_layout_from_step1(
    step_1_output_path: str,
    region_a: float,
    region_b: float,
    save_path: str | None = None,
    visualization_path: str | None = None,
    *,
    layout_to_region_mode: str = "center_preserving_uniform_scale",
    footprint_mode: str = "robust_xy_span",
    ontop_base_label: str | None = None,
    ontop_base_name: str | None = None,
    ontop_include_base: bool = False,
    support_footprint_slice_frac: float | None = None,
    ontop_save_path: str | None = None,
    ontop_visualization_path: str | None = None,
    ontop_graph_save_path: str | None = None,
) -> dict[str, Any]:
    """
    Convert detected objects from Step-1 output to 2D layout in the target region.

    Args:
        step_1_output_path: Path to step_1_output_info.json
        region_a: Region length (x-axis)
        region_b: Region width (y-axis)
        save_path: Optional path to write layout json
        visualization_path: Optional path to write top-down png
    """
    step_1_output = _load_json(step_1_output_path)
    detected = _load_json(step_1_output["detected_categories"])

    rgb_path = step_1_output["input_rgb"]
    image_w, image_h = Image.open(rgb_path).size

    # Recover 3D point cloud in camera frame from Step-1 depth.
    raw_depth = np.array(Image.open(step_1_output["input_depth"]))
    depth_limits = np.array(step_1_output["depth_limits"], dtype=float)
    depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
    K = np.array(step_1_output["K"], dtype=float)
    pc_cam = compute_point_cloud_from_depth(depth=depth, K=K)

    # Same camera-frame -> world-frame conversion used by pipeline Step-3.
    z_dir = np.array(step_1_output["z_direction"], dtype=float)
    origin_pos = np.array(step_1_output["origin_pos"], dtype=float)
    cam_pos, cam_quat = compute_relative_cam_pose_from(z_dir=z_dir, origin_pos=origin_pos)
    og_cam_local_tf = T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
    og_cam_global_tf = T.pose2mat((cam_pos, cam_quat))
    cam_to_world_tf = og_cam_global_tf @ og_cam_local_tf

    names = detected.get("names", [])
    # NOTE: recaption currently introduces label-collapsing failure modes (e.g. many "pen"
    # instances + a "holder" all turning into "pen holder"). For now we ignore recaptioned
    # phrases and stick to original detector phrases for stable instance-level labeling.
    labels = detected.get("phrases", [])
    boxes = detected.get("boxes", [])
    seg_dir = detected.get("segmentation_dir", "")

    # Optional: estimate support plane (desk/floor) and compute footprints from points near that plane.
    footprint_mode_norm = str(footprint_mode or "").strip().lower()
    support_plane: tuple[np.ndarray, float] | None = None
    support_plane_meta: dict[str, Any] | None = None
    if footprint_mode_norm in ("support_plane", "support_plane_filtered", "support_plane_filtered_span"):
        floor_mask_path = str(step_1_output.get("floor_mask", "") or "").strip()
        if floor_mask_path and os.path.exists(floor_mask_path):
            floor_pts = _mask_world_points(mask_path=floor_mask_path, pc_cam=pc_cam, cam_to_world_tf=cam_to_world_tf)
            floor_pts = _sample_points(floor_pts, max_points=30000)
            fit = _fit_plane_svd(floor_pts)
            if fit is not None:
                support_plane = fit
                n, d = support_plane
                support_plane_meta = {
                    "source": "floor_mask",
                    "normal_world": [float(n[0]), float(n[1]), float(n[2])],
                    "d": float(d),
                    "num_points": int(floor_pts.shape[0]),
                }
        # Fallback: fit plane from downsampled full depth point cloud
        if support_plane is None:
            pts_cam = pc_cam.reshape(-1, 3)
            pts_cam = pts_cam[np.isfinite(pts_cam).all(axis=1)]
            pts_cam = _sample_points(pts_cam, max_points=50000) if pts_cam.shape[0] > 0 else pts_cam
            if pts_cam.shape[0] > 0:
                pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=float)], axis=1)
                pts_world = (pts_h @ cam_to_world_tf.T)[:, :3]
                fit = _fit_plane_svd(pts_world)
                if fit is not None:
                    support_plane = fit
                    n, d = support_plane
                    support_plane_meta = {
                        "source": "all_depth_points",
                        "normal_world": [float(n[0]), float(n[1]), float(n[2])],
                        "d": float(d),
                        "num_points": int(pts_world.shape[0]),
                    }

    # First pass: collect per-object world points and centers.
    obj_records: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        pruned_mask_path = os.path.join(seg_dir, f"{name}_nonprojected_mask_pruned.png")
        raw_mask_path = os.path.join(seg_dir, f"{name}_nonprojected_mask.png")
        mask_path = pruned_mask_path if os.path.exists(pruned_mask_path) else raw_mask_path
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            box = boxes[idx] if idx < len(boxes) else [0.5, 0.5, 0.0, 0.0]
            cx, cy, bw, bh = _normalize_cxcywh(box=box, image_w=image_w, image_h=image_h)
            mask_gated = _gate_mask_by_detection_box(
                mask=mask,
                box_norm_cxcywh=(cx, cy, bw, bh),
                image_w=image_w,
                image_h=image_h,
            )
            pts_world = _mask_world_points_from_mask(mask=mask_gated, pc_cam=pc_cam, cam_to_world_tf=cam_to_world_tf)
        else:
            pts_world = np.zeros((0, 3), dtype=float)

        # Fallback for rare cases where mask points are empty: use bbox center with local depth.
        if pts_world.shape[0] == 0:
            box = boxes[idx] if idx < len(boxes) else [0.5, 0.5, 0.0, 0.0]
            cx, cy, bw, bh = _normalize_cxcywh(box=box, image_w=image_w, image_h=image_h)
            px = int(np.clip(round(cx * (image_w - 1)), 0, image_w - 1))
            py = int(np.clip(round(cy * (image_h - 1)), 0, image_h - 1))
            p_cam = pc_cam[py, px]
            if np.isfinite(p_cam).all():
                p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=float)
                p_world = (cam_to_world_tf @ p_h)[:3]
                pts_world = p_world.reshape(1, 3)

        pts_support: np.ndarray | None = None
        thin_object_uses_full_cloud = False
        if pts_world.shape[0] == 0:
            center_world = np.array([0.0, 0.0, 0.0], dtype=float)
            pts_layout_fp = np.zeros((0, 3), dtype=float)
        else:
            if support_plane is not None:
                pts_support = _filter_points_near_plane(
                    pts_world,
                    support_plane,
                    keep_low_quantile=10.0,
                    extra_margin_m=0.01,
                )
                if pts_support.shape[0] < max(30, int(0.05 * pts_world.shape[0])):
                    pts_support = pts_world
            else:
                pts_support = pts_world

            # Thin flat objects (books/notebooks/papers) are easily biased by a
            # bottom-z slice because the slice may collapse onto one edge. For
            # objects whose robust point-cloud thickness is under 5cm, use the
            # full object cloud for both center and footprint.
            robust_z_span = _robust_span(pts_world[:, 2], lo_q=5.0, hi_q=95.0)
            thin_object_uses_full_cloud = bool(robust_z_span < 0.05)
            if thin_object_uses_full_cloud:
                pts_layout_fp = pts_world
            else:
                # Halve effective bottom-band fraction vs CLI (matches relative_layout_modified intent);
                # use a local value so the function arg is not mutated across objects in the loop.
                footprint_slice_for_center = (
                    float(support_footprint_slice_frac) * 0.5
                    if support_footprint_slice_frac is not None
                    else None
                )
                pts_layout_fp = slice_points_world_xyz_for_support_footprint(pts_world, footprint_slice_for_center)
                if pts_layout_fp.shape[0] == 0:
                    pts_layout_fp = pts_world
                # When the bottom slice has too few points, the median is unreliable.
                # Fall back to the full cloud.
                _min_slice_pts = 30
                if (
                    footprint_slice_for_center is not None
                    and pts_layout_fp.shape[0] < _min_slice_pts
                    and pts_world.shape[0] >= _min_slice_pts
                ):
                    pts_layout_fp = pts_world
            # More robust against mask tails / mixed-depth leakage than plain mean.
            center_world = np.median(pts_layout_fp, axis=0)

        obj_records.append(
            {
                "name": name,
                "label": labels[idx] if idx < len(labels) else name,
                "center_world_xyz": center_world,
                "points_world_xy": pts_world[:, :2] if pts_world.shape[0] > 0 else np.zeros((0, 2), dtype=float),
                "points_world_xyz": pts_world if pts_world.shape[0] > 0 else np.zeros((0, 3), dtype=float),
                "points_layout_footprint_world_xyz": pts_layout_fp,
                "thin_object_uses_full_cloud": thin_object_uses_full_cloud,
                "points_support_world_xy": pts_support[:, :2] if pts_support is not None and pts_support.shape[0] > 0 else np.zeros((0, 2), dtype=float),
                "source_box_cxcywh": _normalize_cxcywh(
                    box=boxes[idx] if idx < len(boxes) else [0.5, 0.5, 0.0, 0.0],
                    image_w=image_w,
                    image_h=image_h,
                ),
            }
        )

    # Camera-angle compensation in XY:
    # align layout axes with camera right direction to reduce desk yaw mismatch.
    cam_right_world = cam_to_world_tf[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=float)
    cam_right_xy = np.asarray(cam_right_world[:2], dtype=float)
    cam_right_xy_norm = float(np.linalg.norm(cam_right_xy))
    comp_enabled = cam_right_xy_norm > 1e-6
    comp_yaw = float(np.arctan2(cam_right_xy[1], cam_right_xy[0])) if comp_enabled else 0.0
    comp_angle = -comp_yaw if comp_enabled else 0.0

    centers_xy = np.array([np.asarray(rec["center_world_xyz"][:2], dtype=float) for rec in obj_records], dtype=float) if len(obj_records) > 0 else np.zeros((0, 2), dtype=float)
    if centers_xy.shape[0] > 0:
        pivot_xy = np.median(centers_xy, axis=0)
    else:
        pivot_xy = np.array([0.0, 0.0], dtype=float)

    c = float(np.cos(comp_angle))
    s = float(np.sin(comp_angle))
    rot_xy = np.array([[c, -s], [s, c]], dtype=float)

    all_xy_points_aligned: list[np.ndarray] = []
    all_xy_centers_aligned: list[np.ndarray] = []
    for rec in obj_records:
        center_xy = np.asarray(rec["center_world_xyz"][:2], dtype=float)
        center_xy_aligned = (center_xy - pivot_xy) @ rot_xy.T + pivot_xy
        rec["center_layout_xy"] = center_xy_aligned
        all_xy_centers_aligned.append(center_xy_aligned)

        pls = rec.get("points_layout_footprint_world_xyz")
        if isinstance(pls, np.ndarray) and pls.shape[0] > 0:
            pts_xy = np.asarray(pls[:, :2], dtype=float)
        elif footprint_mode_norm in ("support_plane", "support_plane_filtered", "support_plane_filtered_span"):
            pts_xy = np.asarray(rec.get("points_support_world_xy", rec.get("points_world_xy", np.zeros((0, 2)))), dtype=float)
        else:
            pts_xy = np.asarray(rec["points_world_xy"], dtype=float)
        if pts_xy.shape[0] > 0:
            pts_xy_aligned = (pts_xy - pivot_xy) @ rot_xy.T + pivot_xy
            # When using bottom-slice footprints, tighten quantiles to reduce long-tail stretch
            # from thin vertical structures (e.g. open laptop screen edge).
            span_lo_q, span_hi_q = 5.0, 95.0
            if (
                not bool(rec.get("thin_object_uses_full_cloud", False))
                and support_footprint_slice_frac is not None
                and float(support_footprint_slice_frac) < 1.0
            ):
                span_lo_q, span_hi_q = 12.5, 87.5
            span_x = _robust_span(pts_xy_aligned[:, 0], lo_q=span_lo_q, hi_q=span_hi_q)
            span_y = _robust_span(pts_xy_aligned[:, 1], lo_q=span_lo_q, hi_q=span_hi_q)
            all_xy_points_aligned.append(pts_xy_aligned)
        else:
            span_x = 0.0
            span_y = 0.0
        rec["footprint_layout_xy"] = np.array([span_x, span_y], dtype=float)

    needs_ontop = (ontop_base_label is not None) or (ontop_base_name is not None)
    ontop_slice_frac = support_footprint_slice_frac
    if ontop_slice_frac is not None:
        try:
            ontop_slice_frac = float(ontop_slice_frac)
        except Exception:
            ontop_slice_frac = None
    if ontop_slice_frac is not None:
        ontop_slice_frac = float(np.clip(ontop_slice_frac, 0.0, 1.0))

    scene_graph: dict[str, dict[str, Any]] | None = None
    if needs_ontop:
        # Build ontop before region normalization so parent-child XY constraints affect
        # the saved region positions, not only the debug graph.
        scene_graph = _build_step1_support_graph(
            obj_records=obj_records,
            verbose=False,
            footprint_slice_frac=ontop_slice_frac,
        )
        _adjust_child_xy_by_parent_support(
            obj_records=obj_records,
            scene_graph=scene_graph,
            pivot_xy=pivot_xy,
            rot_xy=rot_xy,
            skip_parent_labels={"desk", "table"},
        )
        all_xy_centers_aligned = [
            np.asarray(rec.get("center_layout_xy", np.zeros(2, dtype=float)), dtype=float) for rec in obj_records
        ]

    # Build normalization bounds from aligned geometry, not from 2D boxes.
    if len(all_xy_points_aligned) > 0:
        all_xy = np.concatenate(all_xy_points_aligned, axis=0)
    else:
        centers = np.array(all_xy_centers_aligned, dtype=float)
        all_xy = centers if centers.size > 0 else np.zeros((1, 2), dtype=float)

    x_min, y_min = np.min(all_xy, axis=0).tolist()
    x_max, y_max = np.max(all_xy, axis=0).tolist()
    scene_dx = max(float(x_max - x_min), 1e-6)
    scene_dy = max(float(y_max - y_min), 1e-6)

    output_objects: list[dict[str, Any]] = []
    for rec in obj_records:
        cx_w, cy_w, cz_w = rec["center_world_xyz"].tolist()
        cx_l, cy_l = [float(v) for v in np.asarray(rec["center_layout_xy"], dtype=float).tolist()]
        fx_l, fy_l = [float(v) for v in np.asarray(rec["footprint_layout_xy"], dtype=float).tolist()]

        if layout_to_region_mode == "center_preserving_uniform_scale":
            # Map layout XY (meters) into the target region (meters).
            # If the scene already fits, keep scale (s=1). Otherwise, uniformly shrink to fit.
            scene_cx = float((x_min + x_max) / 2.0)
            scene_cy = float((y_min + y_max) / 2.0)
            region_cx = float(region_a) / 2.0
            region_cy = float(region_b) / 2.0

            fits_without_scaling = (scene_dx <= float(region_a) + 1e-9) and (scene_dy <= float(region_b) + 1e-9)
            if fits_without_scaling:
                layout_to_region_scale = 1.0
            else:
                layout_to_region_scale = float(min(float(region_a) / scene_dx, float(region_b) / scene_dy))

            x_abs = (cx_l - scene_cx) * layout_to_region_scale + region_cx
            y_abs = (cy_l - scene_cy) * layout_to_region_scale + region_cy
            fx_abs = max(fx_l * layout_to_region_scale, 0.0)
            fy_abs = max(fy_l * layout_to_region_scale, 0.0)

            x_rel = _clip01(x_abs / max(float(region_a), 1e-9))
            y_rel = _clip01(y_abs / max(float(region_b), 1e-9))
            fx_rel = _clip01(fx_abs / max(float(region_a), 1e-9))
            fy_rel = _clip01(fy_abs / max(float(region_b), 1e-9))
        else:
            # Default (legacy): normalize scene bounds into [0,1] then scale by region size.
            x_rel = _clip01((cx_l - x_min) / scene_dx)
            y_rel = _clip01((cy_l - y_min) / scene_dy)
            fx_rel = _clip01(fx_l / scene_dx)
            fy_rel = _clip01(fy_l / scene_dy)
            x_abs = float(region_a) * x_rel
            y_abs = float(region_b) * y_rel
            fx_abs = float(region_a) * fx_rel
            fy_abs = float(region_b) * fy_rel

        output_objects.append(
            {
                "name": rec["name"],
                "label": rec["label"],
                "center_world_xyz": [float(cx_w), float(cy_w), float(cz_w)],
                "region_position_rel": [x_rel, y_rel],  # bottom-left region origin
                "region_position_abs": [float(x_abs), float(y_abs)],
                "region_footprint_rel": [fx_rel, fy_rel],
                "region_footprint_abs": [float(fx_abs), float(fy_abs)],
                "source_box_cxcywh": list(rec["source_box_cxcywh"]),
            }
        )
        if "xy_adjusted_by_parent" in rec:
            output_objects[-1]["xy_adjusted_by_parent"] = rec["xy_adjusted_by_parent"]

    layout_info: dict[str, Any] = {
        "source": {
            "step_1_output_path": os.path.abspath(step_1_output_path),
            "input_rgb": os.path.abspath(rgb_path),
            "image_resolution_wh": [image_w, image_h],
            "depth_path": os.path.abspath(step_1_output["input_depth"]),
            "cam_pose_world": {
                "position": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                "quat_xyzw": [float(cam_quat[0]), float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3])],
            },
        },
        "region": {
            "a": float(region_a),
            "b": float(region_b),
            "axes": {
                "x": "right",
                "y": "up",
                "origin": "bottom_left",
            },
        },
        "projection": {
            "method": "3d_point_cloud_projection",
            "camera_yaw_compensation": {
                "enabled": bool(comp_enabled),
                "yaw_rad": float(comp_yaw),
                "yaw_deg": float(np.degrees(comp_yaw)),
                "rotate_applied_rad": float(comp_angle),
                "pivot_world_xy": [float(pivot_xy[0]), float(pivot_xy[1])],
            },
            "world_xy_bounds": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            },
            "footprint_mode": str(footprint_mode),
            "support_plane": support_plane_meta,
            "layout_geometry": {
                "center_world_xyz": "mean_of_support_z_slice",
                "footprint_layout_xy": "robust_span_of_support_z_slice_xy_after_yaw",
                "support_footprint_slice_frac": None
                if support_footprint_slice_frac is None
                else float(support_footprint_slice_frac),
                "ontop_child_xy_adjustment": "soft_source_box_offset_with_min_layout_overlap"
                if needs_ontop
                else None,
            },
        },
        "objects": output_objects,
    }

    if layout_to_region_mode == "center_preserving_uniform_scale":
        scene_cx = float((x_min + x_max) / 2.0)
        scene_cy = float((y_min + y_max) / 2.0)
        region_cx = float(region_a) / 2.0
        region_cy = float(region_b) / 2.0
        fits_without_scaling = (scene_dx <= float(region_a) + 1e-9) and (scene_dy <= float(region_b) + 1e-9)
        if fits_without_scaling:
            layout_to_region_scale = 1.0
        else:
            layout_to_region_scale = float(min(float(region_a) / scene_dx, float(region_b) / scene_dy))
        (layout_info.get("projection", {}) or {})["layout_to_region"] = {
            "mode": "center_preserving_uniform_scale",
            "fits_without_scaling": bool(fits_without_scaling),
            "scale": float(layout_to_region_scale),
            "scene_center_layout_xy": [float(scene_cx), float(scene_cy)],
            "region_center_xy": [float(region_cx), float(region_cy)],
            "scene_span_layout_xy": [float(scene_dx), float(scene_dy)],
        }

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(layout_info, f, indent=4)

    if visualization_path is not None:
        _draw_topdown_layout(
            objects=output_objects,
            region_a=float(region_a),
            region_b=float(region_b),
            output_path=visualization_path,
        )

    # Optional: compute an ontop/support graph from Step-1 geometry and save only objects on top of a base.
    if needs_ontop and scene_graph is not None:
        if ontop_graph_save_path is not None:
            Path(ontop_graph_save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(ontop_graph_save_path, "w") as f:
                json.dump(scene_graph, f, indent=4)

        base_names: set[str] = set()
        # Support comma-separated labels, e.g. "desk,table".
        label_qs = [x.strip().lower() for x in (ontop_base_label or "").split(",") if x.strip()]
        name_q = (ontop_base_name or "").strip().lower()
        for rec in obj_records:
            nm = str(rec.get("name", ""))
            lb = str(rec.get("label", ""))
            if name_q and nm.lower() == name_q:
                base_names.add(nm)
            if label_qs:
                lb_norm = lb.strip().lower()
                for q in label_qs:
                    if lb_norm == q:
                        base_names.add(nm)
                        break

        # Fallback: if caller asked for desk/table but none exist, treat floor as the base surface.
        # This keeps "ontop desk/table" usable in scenes without an explicit desk/table detection.
        if not base_names and label_qs:
            qset = set(label_qs)
            if ("desk" in qset) or ("table" in qset):
                base_names.add("floor")

        selected = _collect_descendants(scene_graph=scene_graph, roots=base_names)
        if ontop_include_base:
            selected |= base_names

        ontop_objects = [o for o in output_objects if o.get("name") in selected]
        induced_nodes = set(selected) | set(base_names)
        ontop_levels = _collect_ontop_levels(scene_graph=scene_graph, roots=base_names, allowed=induced_nodes)

        ontop_info = {
            **layout_info,
            "filter": {
                "type": "ontop_transitive_closure",
                "base_label": ontop_base_label,
                "base_name": ontop_base_name,
                "include_base": bool(ontop_include_base),
                "support_footprint_slice_frac": ontop_slice_frac,
                "selected_names": sorted(selected),
            },
            "ontop": {
                "base_names": sorted(base_names),
                "levels": ontop_levels,
            },
            "objects": ontop_objects,
        }

        if ontop_save_path is not None:
            Path(ontop_save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(ontop_save_path, "w") as f:
                json.dump(ontop_info, f, indent=4)

        if ontop_visualization_path is not None:
            _draw_topdown_layout(
                objects=ontop_objects,
                region_a=float(region_a),
                region_b=float(region_b),
                output_path=ontop_visualization_path,
            )

    return layout_info


def extract_relative_layout_from_position_manifest(
    position_manifest_path: str,
    region_a: float,
    region_b: float,
    save_path: str | None = None,
    visualization_path: str | None = None,
) -> dict[str, Any]:
    """
    Geometry-only layout extraction (no full Step1/Step2 pipeline required).

    Manifest schema (JSON):
    {
      "input_rgb": "...",
      "input_depth": "...",
      "depth_limits": [min, max],
      "K": [[...],[...],[...]],
      "z_direction": [x, y, z],
      "origin_pos": [x, y, z],
      "segmentation_dir": "...",
      "names": ["obj_0", ...],
      "labels": ["desk", ...],               # optional
      "boxes": [[cx,cy,w,h], ...]            # optional
    }
    """
    manifest = _load_json(position_manifest_path)

    rgb_path = manifest["input_rgb"]
    image_w, image_h = Image.open(rgb_path).size

    raw_depth = np.array(Image.open(manifest["input_depth"]))
    depth_limits = np.array(manifest["depth_limits"], dtype=float)
    depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
    K = np.array(manifest["K"], dtype=float)
    pc_cam = compute_point_cloud_from_depth(depth=depth, K=K)

    z_dir = np.array(manifest["z_direction"], dtype=float)
    origin_pos = np.array(manifest["origin_pos"], dtype=float)
    cam_pos, cam_quat = compute_relative_cam_pose_from(z_dir=z_dir, origin_pos=origin_pos)
    og_cam_local_tf = T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
    og_cam_global_tf = T.pose2mat((cam_pos, cam_quat))
    cam_to_world_tf = og_cam_global_tf @ og_cam_local_tf

    names = list(manifest.get("names", []))
    labels = list(manifest.get("labels", names))
    boxes = list(manifest.get("boxes", []))
    seg_dir = manifest["segmentation_dir"]

    obj_records: list[dict[str, Any]] = []
    all_xy_points: list[np.ndarray] = []
    for idx, name in enumerate(names):
        pruned_mask_path = os.path.join(seg_dir, f"{name}_nonprojected_mask_pruned.png")
        raw_mask_path = os.path.join(seg_dir, f"{name}_nonprojected_mask.png")
        mask_path = pruned_mask_path if os.path.exists(pruned_mask_path) else raw_mask_path
        pts_world = _mask_world_points(mask_path=mask_path, pc_cam=pc_cam, cam_to_world_tf=cam_to_world_tf)

        if pts_world.shape[0] == 0:
            box = boxes[idx] if idx < len(boxes) else [0.5, 0.5, 0.0, 0.0]
            cx, cy, _bw, _bh = _normalize_cxcywh(box=box, image_w=image_w, image_h=image_h)
            px = int(np.clip(round(cx * (image_w - 1)), 0, image_w - 1))
            py = int(np.clip(round(cy * (image_h - 1)), 0, image_h - 1))
            p_cam = pc_cam[py, px]
            if np.isfinite(p_cam).all():
                p_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=float)
                p_world = (cam_to_world_tf @ p_h)[:3]
                pts_world = p_world.reshape(1, 3)

        if pts_world.shape[0] == 0:
            center_world = np.array([0.0, 0.0, 0.0], dtype=float)
            span_x = 0.0
            span_y = 0.0
        else:
            center_world = np.median(pts_world, axis=0)
            span_x = _robust_span(pts_world[:, 0])
            span_y = _robust_span(pts_world[:, 1])
            all_xy_points.append(pts_world[:, :2])

        obj_records.append(
            {
                "name": name,
                "label": labels[idx] if idx < len(labels) else name,
                "center_world_xyz": center_world,
                "footprint_world_xy": np.array([span_x, span_y], dtype=float),
            }
        )

    if len(all_xy_points) > 0:
        all_xy = np.concatenate(all_xy_points, axis=0)
    else:
        centers = np.array([rec["center_world_xyz"][:2] for rec in obj_records], dtype=float)
        all_xy = centers if centers.size > 0 else np.zeros((1, 2), dtype=float)

    x_min, y_min = np.min(all_xy, axis=0).tolist()
    x_max, y_max = np.max(all_xy, axis=0).tolist()
    scene_dx = max(float(x_max - x_min), 1e-6)
    scene_dy = max(float(y_max - y_min), 1e-6)

    output_objects: list[dict[str, Any]] = []
    for rec in obj_records:
        cx_w, cy_w, cz_w = rec["center_world_xyz"].tolist()
        fx_w, fy_w = rec["footprint_world_xy"].tolist()
        x_rel = _clip01((cx_w - x_min) / scene_dx)
        y_rel = _clip01((cy_w - y_min) / scene_dy)
        fx_rel = _clip01(fx_w / scene_dx)
        fy_rel = _clip01(fy_w / scene_dy)

        output_objects.append(
            {
                "name": rec["name"],
                "label": rec["label"],
                "center_world_xyz": [float(cx_w), float(cy_w), float(cz_w)],
                "region_position_rel": [x_rel, y_rel],
                "region_position_abs": [float(region_a) * x_rel, float(region_b) * y_rel],
                "region_footprint_rel": [fx_rel, fy_rel],
                "region_footprint_abs": [float(region_a) * fx_rel, float(region_b) * fy_rel],
            }
        )

    layout_info: dict[str, Any] = {
        "source": {
            "position_manifest_path": os.path.abspath(position_manifest_path),
            "mode": "position_only_geometry",
        },
        "region": {
            "a": float(region_a),
            "b": float(region_b),
            "axes": {"x": "right", "y": "up", "origin": "bottom_left"},
        },
        "projection": {
            "method": "3d_point_cloud_projection",
            "world_xy_bounds": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            },
        },
        "objects": output_objects,
    }

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(layout_info, f, indent=4)

    if visualization_path is not None:
        _draw_topdown_layout(
            objects=output_objects,
            region_a=float(region_a),
            region_b=float(region_b),
            output_path=visualization_path,
        )

    return layout_info


def extract_relative_layout_from_step1_step2_math(
    step_1_output_path: str,
    step_2_output_path: str,
    region_a: float,
    region_b: float,
    cousin_index: int = 0,
    save_path: str | None = None,
    visualization_path: str | None = None,
) -> dict[str, Any]:
    """
    Math-only Step3-like pose extraction (no Omni launch).

    Uses:
      - Step-1 depth / masks / camera geometry
      - Step-2 selected cousin z-angle
    and reproduces the object-center reorientation math used by Step-3 alignment.
    """
    step_1_output = _load_json(step_1_output_path)
    detected = _load_json(step_1_output["detected_categories"])
    step_2_output = _load_json(step_2_output_path)

    rgb_path = step_1_output["input_rgb"]
    image_w, image_h = Image.open(rgb_path).size

    raw_depth = np.array(Image.open(step_1_output["input_depth"]))
    depth_limits = np.array(step_1_output["depth_limits"], dtype=float)
    depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
    K = np.array(step_1_output["K"], dtype=float)
    pc_cam = compute_point_cloud_from_depth(depth=depth, K=K)

    z_dir = np.array(step_1_output["z_direction"], dtype=float)
    origin_pos = np.array(step_1_output["origin_pos"], dtype=float)
    cam_pos, cam_quat = compute_relative_cam_pose_from(z_dir=z_dir, origin_pos=origin_pos)
    og_cam_local_tf = T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
    og_cam_global_tf = T.pose2mat((cam_pos, cam_quat))
    cam_to_world_tf = og_cam_global_tf @ og_cam_local_tf

    names = detected.get("names", [])
    # See note above: ignore recaptioned phrases for now.
    labels = detected.get("phrases", [])
    boxes = detected.get("boxes", [])
    seg_dir = detected.get("segmentation_dir", "")
    step2_objects = step_2_output.get("objects", {})

    tilt_angle = np.arctan2(z_dir[1], z_dir[2])
    tilt_mat = T.euler2mat([tilt_angle, 0, 0])

    obj_records: list[dict[str, Any]] = []
    x_lo_list: list[float] = []
    x_hi_list: list[float] = []
    y_lo_list: list[float] = []
    y_hi_list: list[float] = []

    for idx, name in enumerate(names):
        pruned_mask_path = os.path.join(seg_dir, f"{name}_nonprojected_mask_pruned.png")
        raw_mask_path = os.path.join(seg_dir, f"{name}_nonprojected_mask.png")
        mask_path = pruned_mask_path if os.path.exists(pruned_mask_path) else raw_mask_path

        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            pts_idx = np.flatnonzero(mask.reshape(-1) > 0)
            pc_obj = pc_cam.reshape(-1, 3)[pts_idx]
            pc_obj = pc_obj[np.isfinite(pc_obj).all(axis=1)]
        else:
            pc_obj = np.zeros((0, 3), dtype=float)

        # Fallback if segmentation is missing
        if pc_obj.shape[0] == 0:
            box = boxes[idx] if idx < len(boxes) else [0.5, 0.5, 0.0, 0.0]
            cx, cy, _bw, _bh = _normalize_cxcywh(box=box, image_w=image_w, image_h=image_h)
            px = int(np.clip(round(cx * (image_w - 1)), 0, image_w - 1))
            py = int(np.clip(round(cy * (image_h - 1)), 0, image_h - 1))
            p_cam = pc_cam[py, px]
            if np.isfinite(p_cam).all():
                pc_obj = p_cam.reshape(1, 3)

        # Default: no additional reorientation
        pc_obj_rot = pc_obj @ tilt_mat.T if pc_obj.shape[0] > 0 else pc_obj
        used_z_angle = 0.0
        pan_angle_offset = 0.0

        # Reorientation from Step-2 (same cue path as generation.py)
        if name in step2_objects and pc_obj.shape[0] > 0:
            cousins = step2_objects[name].get("cousins", [])
            if len(cousins) > 0:
                cidx = int(np.clip(cousin_index, 0, len(cousins) - 1))
                cousin_info = cousins[cidx]
                pan_angle_offset, _ = get_reproject_offset(
                    pc_obj=pc_obj.copy(),
                    z_dir=z_dir.copy(),
                    xy_dist=2.30,
                    z_dist=0.65,
                )
                used_z_angle = float(cousin_info.get("z_angle", 0.0)) + float(pan_angle_offset)
                z_rot_mat = T.euler2mat([0, 0, -used_z_angle])
                pc_obj_rot = pc_obj @ tilt_mat.T @ z_rot_mat.T

        if pc_obj_rot.shape[0] == 0:
            center_world = np.array([0.0, 0.0, 0.0], dtype=float)
            span_x = 0.0
            span_y = 0.0
        else:
            obj_min, obj_max = pc_obj_rot.min(axis=0), pc_obj_rot.max(axis=0)
            input_obj_aabb_center_rot = (obj_max + obj_min) / 2.0
            z_rot_mat = T.euler2mat([0, 0, -used_z_angle])
            input_obj_aabb_center = tilt_mat.T @ z_rot_mat.T @ input_obj_aabb_center_rot
            center_world = (cam_to_world_tf @ np.array([*input_obj_aabb_center, 1.0], dtype=float))[:3]

            # Footprint is measured from observed object points in world XY.
            # (Keep this in camera/world-consistent coordinates; reorientation affects center estimate above.)
            pts_cam_h = np.concatenate([pc_obj, np.ones((pc_obj.shape[0], 1), dtype=float)], axis=1)
            pts_world_obs = (cam_to_world_tf @ pts_cam_h.T).T[:, :3]
            span_x = _robust_span(pts_world_obs[:, 0])
            span_y = _robust_span(pts_world_obs[:, 1])

        x_lo_list.append(float(center_world[0] - span_x / 2.0))
        x_hi_list.append(float(center_world[0] + span_x / 2.0))
        y_lo_list.append(float(center_world[1] - span_y / 2.0))
        y_hi_list.append(float(center_world[1] + span_y / 2.0))

        obj_records.append(
            {
                "name": name,
                "label": labels[idx] if idx < len(labels) else name,
                "center_world_xyz": center_world,
                "footprint_world_xy": np.array([span_x, span_y], dtype=float),
                "used_z_angle": float(used_z_angle),
                "pan_angle_offset": float(pan_angle_offset),
            }
        )

    if len(obj_records) == 0:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    else:
        x_min = float(min(x_lo_list))
        x_max = float(max(x_hi_list))
        y_min = float(min(y_lo_list))
        y_max = float(max(y_hi_list))
    scene_dx = max(float(x_max - x_min), 1e-6)
    scene_dy = max(float(y_max - y_min), 1e-6)

    output_objects: list[dict[str, Any]] = []
    for rec in obj_records:
        cx_w, cy_w, cz_w = rec["center_world_xyz"].tolist()
        fx_w, fy_w = rec["footprint_world_xy"].tolist()
        x_rel = _clip01((cx_w - x_min) / scene_dx)
        y_rel = _clip01((cy_w - y_min) / scene_dy)
        fx_rel = _clip01(fx_w / scene_dx)
        fy_rel = _clip01(fy_w / scene_dy)

        output_objects.append(
            {
                "name": rec["name"],
                "label": rec["label"],
                "center_world_xyz": [float(cx_w), float(cy_w), float(cz_w)],
                "used_z_angle": float(rec["used_z_angle"]),
                "pan_angle_offset": float(rec["pan_angle_offset"]),
                "region_position_rel": [x_rel, y_rel],
                "region_position_abs": [float(region_a) * x_rel, float(region_b) * y_rel],
                "region_footprint_rel": [fx_rel, fy_rel],
                "region_footprint_abs": [float(region_a) * fx_rel, float(region_b) * fy_rel],
            }
        )

    layout_info: dict[str, Any] = {
        "source": {
            "step_1_output_path": os.path.abspath(step_1_output_path),
            "step_2_output_path": os.path.abspath(step_2_output_path),
            "mode": "step3_math_only_no_omni",
            "cam_pose_world": {
                "position": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                "quat_xyzw": [float(cam_quat[0]), float(cam_quat[1]), float(cam_quat[2]), float(cam_quat[3])],
            },
        },
        "region": {
            "a": float(region_a),
            "b": float(region_b),
            "axes": {"x": "right", "y": "up", "origin": "bottom_left"},
        },
        "projection": {
            "method": "step3_math_reorient_projection",
            "world_xy_bounds": {
                "x_min": float(x_min),
                "x_max": float(x_max),
                "y_min": float(y_min),
                "y_max": float(y_max),
            },
        },
        "objects": output_objects,
    }

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(layout_info, f, indent=4)

    if visualization_path is not None:
        _draw_topdown_layout(
            objects=output_objects,
            region_a=float(region_a),
            region_b=float(region_b),
            output_path=visualization_path,
        )

    return layout_info


def extract_relative_layout_from_step_outputs(
    step_1_output_path: str,
    region_a: float,
    region_b: float,
    step_2_output_path: str | None = None,
    cousin_index: int = 0,
    save_path: str | None = None,
    visualization_path: str | None = None,
    *,
    footprint_mode: str = "robust_xy_span",
    ontop_base_label: str | None = None,
    ontop_base_name: str | None = None,
    ontop_include_base: bool = False,
    support_footprint_slice_frac: float | None = None,
    ontop_save_path: str | None = None,
    ontop_visualization_path: str | None = None,
    ontop_graph_save_path: str | None = None,
) -> dict[str, Any]:
    """
    Unified step-output entrypoint:
      - With step_2_output_path: use math-only reorientation path.
      - Without step_2_output_path: use pure Step-1 geometry path.
    """
    if step_2_output_path:
        layout = extract_relative_layout_from_step1_step2_math(
            step_1_output_path=step_1_output_path,
            step_2_output_path=step_2_output_path,
            region_a=region_a,
            region_b=region_b,
            cousin_index=cousin_index,
            save_path=save_path,
            visualization_path=visualization_path,
        )
        # If requested, also compute ontop filtering using Step-1 geometry (no need to run Step-3).
        if (ontop_base_label is not None) or (ontop_base_name is not None):
            extract_relative_layout_from_step1(
                step_1_output_path=step_1_output_path,
                region_a=region_a,
                region_b=region_b,
                save_path=None,
                visualization_path=None,
                footprint_mode=str(footprint_mode),
                ontop_base_label=ontop_base_label,
                ontop_base_name=ontop_base_name,
                ontop_include_base=ontop_include_base,
                support_footprint_slice_frac=support_footprint_slice_frac,
                ontop_save_path=ontop_save_path,
                ontop_visualization_path=ontop_visualization_path,
                ontop_graph_save_path=ontop_graph_save_path,
            )
        return layout
    return extract_relative_layout_from_step1(
        step_1_output_path=step_1_output_path,
        region_a=region_a,
        region_b=region_b,
        save_path=save_path,
        visualization_path=visualization_path,
        footprint_mode=str(footprint_mode),
        ontop_base_label=ontop_base_label,
        ontop_base_name=ontop_base_name,
        ontop_include_base=ontop_include_base,
        support_footprint_slice_frac=support_footprint_slice_frac,
        ontop_save_path=ontop_save_path,
        ontop_visualization_path=ontop_visualization_path,
        ontop_graph_save_path=ontop_graph_save_path,
    )


def extract_relative_layout_from_image(
    input_image_path: str,
    region_a: float,
    region_b: float,
    gpt_api_key: str | None,
    save_dir: str,
    captions: list[str] | None = None,
    gpt_version: str = "qwen-vl-max",
    max_retries: int = 3,
    retry_wait_time: float = 5.0,
    recaption_similarity_threshold: float = 0.5,
    recaption_filter_enabled: bool = True,
    *,
    layout_to_region_mode: str = "center_preserving_uniform_scale",
    footprint_mode: str = "robust_xy_span",
    ontop_base_label: str | None = None,
    ontop_base_name: str | None = None,
    ontop_include_base: bool = False,
    support_footprint_slice_frac: float | None = None,
    verbose: bool = False,
) -> tuple[dict[str, Any], str]:
    """
    Run modified Step-1 on input image, then map object layout to (a, b).

    Returns:
        (layout_info, step_1_output_path)
    """
    config_path = str(Path(constants.CONFIG_DEFAULT_YAML))
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader) or {}
    local_keys_path = Path(constants.ROOT_DIR) / "configs" / "local_keys.yaml"
    if local_keys_path.exists():
        with open(local_keys_path, "r", encoding="utf-8") as f:
            local_keys = yaml.load(f, Loader=yaml.FullLoader) or {}
        if isinstance(local_keys, dict):
            def _deep_merge(base: dict, extra: dict) -> dict:
                merged = dict(base)
                for k, v in extra.items():
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged[k] = _deep_merge(merged[k], v)
                    else:
                        merged[k] = v
                return merged

            if isinstance(config, dict):
                config = _deep_merge(config, local_keys)

    # Allow callers to omit GPT settings; fall back to defaults in config.
    # Config path: pipeline.RealWorldExtractor.call.*
    step1_cfg = (((config or {}).get("pipeline", {}) or {}).get("RealWorldExtractor", {}) or {}).get("call", {}) or {}
    if gpt_api_key is None:
        gpt_api_key = step1_cfg.get("gpt_api_key", None)
    if captions is None:
        cfg_caps = step1_cfg.get("captions", None)
        if isinstance(cfg_caps, list):
            captions = [str(x) for x in cfg_caps]
        elif isinstance(cfg_caps, str) and cfg_caps.strip():
            captions = [x.strip() for x in cfg_caps.split(",") if x.strip()]
    if not gpt_version:
        gpt_version = str(step1_cfg.get("gpt_version", "qwen-vl-max"))
    if max_retries is None:
        max_retries = int(step1_cfg.get("max_retries", 3))
    if retry_wait_time is None:
        retry_wait_time = float(step1_cfg.get("retry_wait_time", 5.0))

    fm = FeatureMatcher(**config["models"]["FeatureMatcher"])
    step_1 = RealWorldExtractor(feature_matcher=fm, verbose=verbose)

    success, step_1_output_path = step_1(
        input_path=input_image_path,
        gpt_api_key=gpt_api_key,
        gpt_version=gpt_version,
        max_retries=max_retries,
        retry_wait_time=retry_wait_time,
        captions=captions,
        save_dir=save_dir,
        visualize=False,
        recaption_similarity_threshold=recaption_similarity_threshold,
        recaption_filter_enabled=recaption_filter_enabled,
    )
    if not success or step_1_output_path is None:
        raise RuntimeError("Step-1 extraction failed, cannot compute relative layout.")

    out_dir = os.path.join(save_dir, "relative_layout")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = os.path.join(out_dir, "relative_layout.json")
    out_png = os.path.join(out_dir, "relative_layout_topdown.png")

    ontop_slug = _safe_slug(ontop_base_name or ontop_base_label or "ontop")
    ontop_json = os.path.join(out_dir, f"relative_layout_ontop_{ontop_slug}.json") if (ontop_base_label or ontop_base_name) else None
    ontop_png = os.path.join(out_dir, f"relative_layout_ontop_{ontop_slug}_topdown.png") if (ontop_base_label or ontop_base_name) else None
    ontop_graph = os.path.join(out_dir, f"support_graph_step1_{ontop_slug}.json") if (ontop_base_label or ontop_base_name) else None

    layout_info = extract_relative_layout_from_step1(
        step_1_output_path=step_1_output_path,
        region_a=region_a,
        region_b=region_b,
        save_path=out_json,
        visualization_path=out_png,
        layout_to_region_mode=str(layout_to_region_mode),
        footprint_mode=str(footprint_mode),
        ontop_base_label=ontop_base_label,
        ontop_base_name=ontop_base_name,
        ontop_include_base=bool(ontop_include_base),
        support_footprint_slice_frac=support_footprint_slice_frac,
        ontop_save_path=ontop_json,
        ontop_visualization_path=ontop_png,
        ontop_graph_save_path=ontop_graph,
    )
    return layout_info, step_1_output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 2D relative object layout into region (a, b).")
    parser.add_argument("--region_a", type=float, default=6.0, help="Region length along x.")
    parser.add_argument("--region_b", type=float, default=4.0, help="Region width along y.")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--footprint_mode",
        type=str,
        default="robust_xy_span",
        choices=["robust_xy_span", "support_plane"],
        help="How to compute top-down footprint from point cloud. "
        "robust_xy_span uses robust XY span of all object points; "
        "support_plane filters object points near the fitted support plane (desk/floor) before computing footprint.",
    )
    parser.add_argument(
        "--ontop_base_label",
        type=str,
        default=None,
        help="If set (e.g. 'desk'), also write an ontop-filtered layout containing only objects on top of that label (transitively).",
    )
    parser.add_argument(
        "--ontop_base_name",
        type=str,
        default=None,
        help="Exact base object name (e.g. 'desk_0'). If set, used together with --ontop_base_label.",
    )
    parser.add_argument(
        "--ontop_include_base",
        action="store_true",
        help="Include the base object itself in the ontop-filtered output.",
    )
    parser.add_argument(
        "--support-footprint-slice-frac",
        type=float,
        default=None,
        dest="support_footprint_slice_frac",
        help=(
            "Optional: when building Step-1 support/ontop graph, restrict each object's footprint "
            "to the bottom slice of its point cloud before XY projection. "
            "Example: 0.2 uses only the bottom 20%% of the object's height (good for lamps). "
            "Default None disables slicing (uses full AABB footprint, legacy behavior)."
        ),
    )
    parser.add_argument(
        "--support_footprint_slice_frac",
        type=float,
        default=None,
        dest="support_footprint_slice_frac",
        help=argparse.SUPPRESS,
    )

    # Mode A (recommended): Step outputs only (no Omni launch needed)
    parser.add_argument("--step_1_output_path", type=str, default=None, help="Path to step_1_output_info.json")
    parser.add_argument("--step_2_output_path", type=str, default=None, help="Path to step_2_output_info.json.")
    parser.add_argument("--cousin_index", type=int, default=0, help="Which cousin index to use from Step-2.")

    # Mode B: direct from Step-3 outputs (compatibility)
    parser.add_argument("--scene_info_path", type=str, default=None, help="Path to scene_<k>_info.json from Step-3.")
    parser.add_argument("--step_3_output_path", type=str, default=None, help="Path to step_3_output_info.json.")
    parser.add_argument("--scene_name", type=str, default="scene_0", help="Scene key in step_3_output_info.json.")

    # Mode C: run Step-1 from image
    parser.add_argument("--input_image_path", type=str, default=None, help="Path to input RGB image.")
    parser.add_argument("--gpt_api_key", type=str, default=None, help="DashScope API key (needed if captions not given).")
    parser.add_argument("--captions", type=str, default=None, help="Comma-separated object captions to avoid GPT captioning.")
    parser.add_argument("--gpt_version", type=str, default="qwen-vl-max")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_wait_time", type=float, default=5.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    region_a = float(args.region_a)
    region_b = float(args.region_b)
    if region_a <= 0 or region_b <= 0:
        raise ValueError("region_a and region_b must be positive.")

    if (
        args.step_1_output_path is None
        and args.scene_info_path is None
        and args.step_3_output_path is None
        and args.input_image_path is None
    ):
        raise ValueError(
            "Must provide one of --step_1_output_path / --scene_info_path / --step_3_output_path / --input_image_path."
        )
    if args.step_2_output_path is not None and args.step_1_output_path is None:
        raise ValueError("--step_2_output_path requires --step_1_output_path.")

    save_dir = args.save_dir
    if save_dir is None:
        if args.step_2_output_path is not None:
            save_dir = str(Path(args.step_2_output_path).resolve().parent)
        elif args.step_1_output_path is not None:
            save_dir = str(Path(args.step_1_output_path).resolve().parent)
        elif args.scene_info_path is not None:
            save_dir = str(Path(args.scene_info_path).resolve().parent)
        elif args.step_3_output_path is not None:
            save_dir = str(Path(args.step_3_output_path).resolve().parent)
        else:
            save_dir = str(Path(args.input_image_path).resolve().parent)
    save_dir = os.path.abspath(os.path.expanduser(save_dir))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if args.step_1_output_path is not None:
        out_dir = os.path.join(save_dir, "relative_layout")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_json = os.path.join(out_dir, "relative_layout.json")
        out_png = os.path.join(out_dir, "relative_layout_topdown.png")
        ontop_slug = _safe_slug(args.ontop_base_name or args.ontop_base_label or "ontop")
        ontop_json = os.path.join(out_dir, f"relative_layout_ontop_{ontop_slug}.json") if (args.ontop_base_label or args.ontop_base_name) else None
        ontop_png = os.path.join(out_dir, f"relative_layout_ontop_{ontop_slug}_topdown.png") if (args.ontop_base_label or args.ontop_base_name) else None
        ontop_graph = os.path.join(out_dir, f"support_graph_step1_{ontop_slug}.json") if (args.ontop_base_label or args.ontop_base_name) else None
        extract_relative_layout_from_step_outputs(
            step_1_output_path=args.step_1_output_path,
            step_2_output_path=args.step_2_output_path,
            region_a=region_a,
            region_b=region_b,
            cousin_index=int(args.cousin_index),
            save_path=out_json,
            visualization_path=out_png,
            footprint_mode=str(args.footprint_mode),
            ontop_base_label=args.ontop_base_label,
            ontop_base_name=args.ontop_base_name,
            ontop_include_base=bool(args.ontop_include_base),
            support_footprint_slice_frac=args.support_footprint_slice_frac,
            ontop_save_path=ontop_json,
            ontop_visualization_path=ontop_png,
            ontop_graph_save_path=ontop_graph,
        )
        print(f"Saved: {out_json}")
        print(f"Saved: {out_png}")
        if ontop_json is not None:
            print(f"Saved: {ontop_json}")
        if ontop_png is not None:
            print(f"Saved: {ontop_png}")
        if ontop_graph is not None:
            print(f"Saved: {ontop_graph}")
        return

    if args.scene_info_path is not None:
        out_dir = os.path.join(save_dir, "relative_layout")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_json = os.path.join(out_dir, "relative_layout.json")
        out_png = os.path.join(out_dir, "relative_layout_topdown.png")
        extract_relative_layout_from_scene_info(
            scene_info_path=args.scene_info_path,
            region_a=region_a,
            region_b=region_b,
            save_path=out_json,
            visualization_path=out_png,
        )
        print(f"Saved: {out_json}")
        print(f"Saved: {out_png}")
        return

    if args.step_3_output_path is not None:
        out_dir = os.path.join(save_dir, "relative_layout")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_json = os.path.join(out_dir, "relative_layout.json")
        out_png = os.path.join(out_dir, "relative_layout_topdown.png")
        extract_relative_layout_from_step3_output(
            step_3_output_path=args.step_3_output_path,
            scene_name=args.scene_name,
            region_a=region_a,
            region_b=region_b,
            save_path=out_json,
            visualization_path=out_png,
        )
        print(f"Saved: {out_json}")
        print(f"Saved: {out_png}")
        return

    captions = None
    if args.captions:
        captions = [x.strip() for x in args.captions.split(",") if x.strip()]

    # End-to-end default: if user didn't specify any ontop base, assume desk/table.
    ontop_base_label = args.ontop_base_label
    ontop_base_name = args.ontop_base_name
    if ontop_base_label is None and ontop_base_name is None:
        ontop_base_label = "desk,table"

    _layout_info, step_1_out = extract_relative_layout_from_image(
        input_image_path=args.input_image_path,
        region_a=region_a,
        region_b=region_b,
        gpt_api_key=args.gpt_api_key,
        save_dir=save_dir,
        captions=captions,
        gpt_version=args.gpt_version,
        max_retries=args.max_retries,
        retry_wait_time=args.retry_wait_time,
        verbose=args.verbose,
        ontop_base_label=ontop_base_label,
        ontop_base_name=ontop_base_name,
        ontop_include_base=bool(args.ontop_include_base),
        footprint_mode=str(args.footprint_mode),
    )
    print(f"Step-1 output: {step_1_out}")
    print(f"Saved relative layout to: {os.path.join(save_dir, 'relative_layout')}")


if __name__ == "__main__":
    main()


