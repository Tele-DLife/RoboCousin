# Portions of this file are adapted from Digital Cousins:
# https://github.com/cremebrule/digital-cousins
# Modified by RoboCousin contributors.
# Licensed under Apache-2.0; see licenses/Digital-Cousins-Apache-2.0.txt.
import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
import json
import faiss
import os
from groundingdino.util.inference import load_image, annotate
from groundingdino.datasets.transforms import RandomResize
from torchvision.ops import box_convert
from copy import deepcopy

from .clip import CLIPEncoder
from .dino_v2 import DinoV2Encoder
from .grounded_sam_v2 import GroundedSAMv2


class FeatureMatcher(torch.nn.Module):
    ENCODERS = {
        "DinoV2Encoder": DinoV2Encoder,
        "CLIPEncoder": CLIPEncoder,
    }
    def __init__(
            self,

            # Encoder kwargs
            encoder="DinoV2Encoder",
            encoder_kwargs=None,

            # Grounded SAM v2 kwargs
            gsam_box_threshold=0.25,
            gsam_text_threshold=0.25,

            # General kwargs
            device="cuda",
            verbose=False,
    ):
        """
        Args:
            encoder (str): Type of visual encoder used to generate visual embeddings.
                Valid options are {"DinoV2Encoder", "CLIPEncoder"}
            encoder_kwargs (None or dict): If specified, encoder-specific kwargs to pass to the encoder constructor
            gsam_box_threshold (float): Confidence threshold for generating GroundedSAM bounding box. 
                If there are undetected objects in the input scene, consider decrease this value.
            gsam_text_threshold (float): Confidence threshold for generating GroundedSAM text annotation.
                If there are undetected objects in the input scene, consider decrease this value.
            device (str): Device to use for storing tensors
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        # Call super first
        super().__init__()

        # Initialize internal variables
        self.device = device
        self.verbose = verbose
        self.encoder_name = encoder

        # Create encoder
        assert encoder in self.ENCODERS, \
            f"Invalid encoder specified! Valid options are: {self.ENCODERS.keys()}, got: {encoder}"
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        encoder_kwargs["device"] = device
        self.encoder = self.ENCODERS[self.encoder_name](**encoder_kwargs)

        # Create SAM model
        self.gsam = GroundedSAMv2(
            box_threshold=gsam_box_threshold,
            text_threshold=gsam_text_threshold,
            device=self.device,
        )
        self.eval()

    def compute_segmentation_mask(
            self,
            input_category,
            input_img_fpath,
            save_dir=None,
            save_prefix=None,
            multi_results=False,
    ):
        """
        Args:
            input_category (str): Name of the desired object category to segment from
                @input_img_fpath. It is this category that is assumed will be attempted
                to be segmented from the image located at @input_img_fpath
            input_img_fpath (str): Absolute filepath to the input object image
            save_dir (None or str): If specified, the absolute path to the directory where the results should be saved.
                If None, will default to the same directory of @input_img_fpath. Note that in either case the
                file saved is named f"{input_category}_mask.png"
            save_prefix (None or str): If specified, the prefix string name for saved outputs.
                If None, saved outputs will be prepended with @input_category instead
            multi_results (bool): Whether to compute multiple bounding boxes or only the highest probability one

        Returns:
            2-tuple:
                - list of np.ndarray: (H, W) segmented mask(s) (>1 if multi_results=True)
                - list of str: List of absolute paths to the segmented mask(s) (>1 if multi_results=True)
        """
        # Standardize save dir and make sure it exists
        save_dir = str(Path(input_img_fpath).parent) if save_dir is None else save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_prefix = input_category if save_prefix is None else save_prefix

        # Load the input image
        image_source, image = load_image(input_img_fpath)

        # Predict the bounding boxes
        boxes, logits, phrases = self.gsam.predict_boxes(img=image, caption=f"{input_category}.")

        # Save this image
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(f"{save_dir}/{save_prefix}_annotated_bboxes.png", annotated_frame)

        # Only keep pixels within the segmented category box
        # Sort them based on filtering mechanism
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Calculate the number of actually category-level bboxes, make sure we have at least 1
        category_boxes = [(box, logit) for box, logit in zip(boxes_xyxy, logits)]
        n_obj_bboxes = len(category_boxes)
        assert n_obj_bboxes > 0, "Did not find any valid category-level obj bboxes!"

        if n_obj_bboxes > 1:
            if not multi_results:
                # Grab highest probability one
                obj_bboxes = sorted(category_boxes, key=lambda x: x[1])
                obj_bbox = [obj_bboxes[-1][0]]
        else:
            obj_bbox = [category_boxes[0][0]]

        masks, file_dirs = [], []
        if n_obj_bboxes > 1 and multi_results:
            for i, (box, logit) in enumerate(category_boxes):
                # Get segmentation mask for the object category
                obj_masks = self.gsam.predict_segmentation(
                    img_source=image_source,
                    boxes=np.array([box]),
                    cxcywh=False,
                )
                obj_mask = obj_masks[0].squeeze(axis=0)
                save_fpath = f"{save_dir}/{save_prefix}_{i}_mask.png"
                Image.fromarray(obj_mask).save(save_fpath)
                masks.append(obj_mask)
                file_dirs.append(save_fpath)
        else:
            # Get segmentation mask for the object category
            obj_masks = self.gsam.predict_segmentation(
                img_source=image_source,
                boxes=np.array(obj_bbox),
                cxcywh=False,
            )
            obj_mask = obj_masks[0].squeeze(axis=0)
            save_fpath = f"{save_dir}/{save_prefix}_mask.png"
            Image.fromarray(obj_mask).save(save_fpath)
            masks.append(obj_mask)
            file_dirs.append(save_fpath)

        return masks, file_dirs

    def find_nearest_neighbor_candidates(
            self,
            input_category,
            input_img_fpath,
            candidate_imgs_fdirs=None,
            candidate_imgs=None,
            candidate_filter=None,
            n_candidates=4,
            save_dir=None,
            visualize_resolution=(640, 480),
            boxes=None,
            logits=None,
            phrases=None,
            obj_masks=None,
            save_prefix=None,
            remove_background=True,
            use_input_img_without_bbox=False,
            normalize_crops: bool = True,
            normalize_target_size: int = 448,
            normalize_margin_frac: float = 0.15,
    ):
        """
        Args:
            input_category (str): Name of the desired object category to segment from
                @input_img_fpath. It is this category that is assumed will be attempted
                to be matched to nearest neighbor candidate(s)
            input_img_fpath (str): Absolute filepath to the input object image
            candidate_imgs_fdirs (None or str or list of str): Absolute filepath(s) to the candidate images directory(s)
            candidate_imgs (None or list of str): Absolute filepath(s) to the candidate images. If this is not None, directly use this. Otherwise, use candidate_imgs_fdirs
            candidate_filter (None or TextFilter): If specified, TextFilter for pruning all possible
                candidates from @candidate_imgs_fdir
            n_candidates (int): The number of nearest neighbor candidates to return.
            save_dir (None or str): If specified, the absolute path to the directory where the results should be saved.
                If None, will default to the same directory of @input_img_fpath.
            visualize_resolution (2-tuple): (H, W) when visualizing candidate results
            boxes (None or tensor): If specified, pre-computed SAM boxes to use
            logits (None or tensor): If specified, pre-computed SAM logits to use
            phrases (None or list): If specified, pre-computed SAM phrases to use
            obj_masks (None or np.array): If specified, pre-computed SAM segmentation mask to use
            save_prefix (None or str): If specified, the prefix string name for saved outputs.
                If None, saved outputs will be prepended with @input_category instead
            remove_background (bool): Whether to remove background before computing DINO features
            use_input_img_without_bbox (bool): Whether to directly use the input image to compute dino score,
                or with a bounding box of the target object

        Returns:
            dict: Dictionary of outputs. Note that this will also be saved to f"{save_prefix}_feature_matcher_results.json"
                in @save_dir
        """
        assert " " not in input_category
        assert (candidate_imgs_fdirs is not None) or (candidate_imgs is not None)

        # Standardize save dir and make sure it exists
        save_dir = str(Path(input_img_fpath).parent) if save_dir is None else save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_prefix = input_category if save_prefix is None else save_prefix

        # Standardize other inputs
        if candidate_imgs is None:
            candidate_imgs_fdirs = [candidate_imgs_fdirs] if isinstance(candidate_imgs_fdirs, str) else candidate_imgs_fdirs

        # Load the input image
        image_source, image = load_image(input_img_fpath)
        ref_img_vis = cv2.resize(image_source, visualize_resolution)
        H_ref, W_ref, _ = image_source.shape

        # Predict the bounding boxes
        if self.verbose:
            print(f"{self.__class__.__name__}: Computing GroundedSAMv2 obj boxes...")

        # First find category-level boxes
        if not use_input_img_without_bbox:
            if boxes is None or logits is None or phrases is None:
                # Either all or none of must be None
                assert boxes is None and logits is None and phrases is None,\
                    "All of boxes, logits, and phrases must be None if at least one of them is None!"
                boxes, logits, phrases = self.gsam.predict_boxes(img=image, caption=f"{input_category.replace('_', ' ')}.")

            # Save this image
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(f"{save_dir}/{save_prefix}_annotated_bboxes.png", annotated_frame)

            # Only keep pixels within the segmented category box
            # Sort them based on filtering mechanism
            boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            category_boxes = []

            # Infer which of the bboxes belong to the object itself if there's more than 1
            if len(logits) > 1:
                for box, logit, phrase in zip(boxes_xyxy, logits, phrases):
                    if phrase.replace("_", " ") == input_category.replace("_", " "):
                        category_boxes.append((box, logit))
            else:
                category_boxes = [(boxes_xyxy[0], logits[0])]

            # Calculate the number of actually category-level bboxes, make sure we have at least 1
            n_obj_bboxes = len(category_boxes)
            assert n_obj_bboxes > 0, "Did not find any valid category-level obj bboxes!"
            if n_obj_bboxes > 1:
                # Grab highest probability one
                obj_bboxes = sorted(category_boxes, key=lambda x: x[1])
                obj_bbox = obj_bboxes[-1][0]
            else:
                obj_bbox = category_boxes[0][0]

            # Calculate the obj pixels based on its bbox
            box_pixels = (obj_bbox * np.array([W_ref, H_ref, W_ref, H_ref])).astype(int)

            # Get segmentation mask for the object itself
            if obj_masks is None:
                obj_masks = self.gsam.predict_segmentation(
                    img_source=image_source,
                    boxes=np.array([obj_bbox]),
                    cxcywh=False,
                )
            obj_mask = obj_masks[0].squeeze(axis=0)
            Image.fromarray(obj_mask).save(f"{save_dir}/{save_prefix}_mask.png")
        else:
            height, width, _ = image_source.shape
            obj_mask = np.ones((height, width))

        # Mask the original image source and image and then infer the corresponding part-level segmentations

        # TODO: Black out all background pixels from ref_img_cropped using segmentation mask
        preprocess = RandomResize([800], max_size=1333)
        obj_mask_resized = np.expand_dims(np.array(preprocess(Image.fromarray(obj_mask))[0]), axis=0)
        image_source_masked = image_source * np.expand_dims(np.where(obj_mask > 0.5, 1.0, 0.0), axis=-1).astype(np.uint8) if remove_background else image_source
        image_masked = image * torch.tensor(obj_mask_resized, requires_grad=False) if remove_background else image
        if use_input_img_without_bbox:
            ref_img_cropped = image_source_masked
            obj_mask_cropped = obj_mask
        else:
            ref_img_cropped = image_source_masked[box_pixels[1]:box_pixels[3], box_pixels[0]:box_pixels[2]]
            obj_mask_cropped = obj_mask[box_pixels[1]:box_pixels[3], box_pixels[0]:box_pixels[2]]
        ref_img_masked_vis = cv2.resize(image_source_masked, visualize_resolution)

        def _tight_bbox_from_mask(mask_2d: np.ndarray) -> tuple[int, int, int, int] | None:
            if mask_2d is None or mask_2d.size == 0:
                return None
            ys, xs = np.where(mask_2d > 0)
            if xs.size == 0 or ys.size == 0:
                return None
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            return x0, y0, x1, y1

        def _expand_bbox(
            x0: int,
            y0: int,
            x1: int,
            y1: int,
            h: int,
            w: int,
            margin_frac: float,
        ) -> tuple[int, int, int, int]:
            margin_frac = float(margin_frac)
            margin_frac = max(0.0, min(margin_frac, 2.0))
            bw = max(1, x1 - x0 + 1)
            bh = max(1, y1 - y0 + 1)
            mx = int(round(bw * margin_frac))
            my = int(round(bh * margin_frac))
            xx0 = max(0, x0 - mx)
            yy0 = max(0, y0 - my)
            xx1 = min(w - 1, x1 + mx)
            yy1 = min(h - 1, y1 + my)
            return xx0, yy0, xx1, yy1

        def _square_pad_and_resize(
            img_rgb: np.ndarray,
            mask_2d: np.ndarray | None,
            target_size: int,
            pad_value: int,
        ) -> tuple[np.ndarray, np.ndarray | None]:
            if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
                return img_rgb, mask_2d
            h, w, _ = img_rgb.shape
            side = int(max(h, w))
            if side <= 0:
                return img_rgb, mask_2d
            top = (side - h) // 2
            bottom = side - h - top
            left = (side - w) // 2
            right = side - w - left
            img_pad = cv2.copyMakeBorder(
                img_rgb,
                top,
                bottom,
                left,
                right,
                borderType=cv2.BORDER_CONSTANT,
                value=(pad_value, pad_value, pad_value),
            )
            mask_pad = None
            if mask_2d is not None and mask_2d.ndim == 2:
                mask_u8 = mask_2d.astype(np.uint8)
                mask_pad = cv2.copyMakeBorder(
                    mask_u8,
                    top,
                    bottom,
                    left,
                    right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=0,
                )
            ts = int(target_size)
            ts = max(32, ts)
            img_out = cv2.resize(img_pad, (ts, ts), interpolation=cv2.INTER_AREA)
            if mask_pad is not None:
                mask_out = cv2.resize(mask_pad, (ts, ts), interpolation=cv2.INTER_NEAREST)
            else:
                mask_out = None
            return img_out, mask_out

        # Normalize reference crop: tight bbox on mask -> expand margin -> square pad -> resize.
        if normalize_crops:
            try:
                if obj_mask_cropped is not None and obj_mask_cropped.ndim == 2 and ref_img_cropped.ndim == 3:
                    bb = _tight_bbox_from_mask(obj_mask_cropped.astype(np.uint8))
                    if bb is not None:
                        x0, y0, x1, y1 = _expand_bbox(
                            bb[0], bb[1], bb[2], bb[3],
                            h=int(obj_mask_cropped.shape[0]),
                            w=int(obj_mask_cropped.shape[1]),
                            margin_frac=float(normalize_margin_frac),
                        )
                        ref_img_cropped = ref_img_cropped[y0:y1 + 1, x0:x1 + 1]
                        obj_mask_cropped = obj_mask_cropped[y0:y1 + 1, x0:x1 + 1]
                # Use black padding since reference background has been zeroed when remove_background=True.
                ref_img_cropped, obj_mask_cropped = _square_pad_and_resize(
                    img_rgb=ref_img_cropped,
                    mask_2d=obj_mask_cropped,
                    target_size=int(normalize_target_size),
                    pad_value=0,
                )
            except Exception:
                # Fall back silently to original behavior if normalization fails.
                pass

        # Get all valid candidates and load them
        models = list(sorted(f"{candidate_imgs_fdir}/{model}"
                             for candidate_imgs_fdir in candidate_imgs_fdirs for model in os.listdir(candidate_imgs_fdir)
                             if (candidate_filter is None or candidate_filter.process(model)))) \
            if candidate_imgs is None else sorted(candidate_imgs)
        model_imgs_list = [np.array(Image.open(model).convert("RGB")) for model in models]

        # Normalize candidate crops: tight bbox via non-white proxy -> expand margin -> square pad -> resize.
        if normalize_crops:
            norm_list = []
            for img_rgb in model_imgs_list:
                try:
                    if img_rgb.ndim == 3 and img_rgb.shape[-1] == 3:
                        bg = (img_rgb[..., 0] > 245) & (img_rgb[..., 1] > 245) & (img_rgb[..., 2] > 245)
                        fg = (~bg).astype(np.uint8)
                        bb = _tight_bbox_from_mask(fg)
                        if bb is not None:
                            x0, y0, x1, y1 = _expand_bbox(
                                bb[0], bb[1], bb[2], bb[3],
                                h=int(img_rgb.shape[0]),
                                w=int(img_rgb.shape[1]),
                                margin_frac=float(normalize_margin_frac),
                            )
                            img_rgb = img_rgb[y0:y1 + 1, x0:x1 + 1]
                        # Keep white padding for candidates to match their typical background.
                        img_rgb, _ = _square_pad_and_resize(
                            img_rgb=img_rgb,
                            mask_2d=None,
                            target_size=int(normalize_target_size),
                            pad_value=255,
                        )
                except Exception:
                    pass
                norm_list.append(img_rgb)
            model_imgs_list = norm_list

        model_imgs = np.array(model_imgs_list)

        # Compute DINO features and reshape them to be (N, D) arrays
        ref_img_feats = self.encoder.get_features(ref_img_cropped).squeeze(axis=0)  # (84, 112, 384)
        model_imgs_feats = self.encoder.get_features(model_imgs)    # (64, 84, 112, 384)
        ref_feat_vecs = ref_img_feats.reshape(-1, self.encoder.embedding_dim)   # (9408, 384)
        # TODO: Remove background pixels from candidate feat vecs, via better dataset parsing (use alpha channel = 0.0)
        model_feat_vecs = model_imgs_feats.reshape(-1, self.encoder.embedding_dim)  # (602112, 384)

        # Reshape cropped image to be the same shape as the feature size
        if self.encoder_name == "DinoV2Encoder":
            H, W, C = ref_img_feats.shape
            obj_mask_cropped_resized = cv2.resize(obj_mask_cropped.astype(np.uint8), (W, H))
        elif self.encoder_name == "CLIPEncoder":
            obj_mask_cropped_resized = obj_mask_cropped
        else:
            raise ValueError(f"Got invalid encoder_name! Valid options: {self.ENCODERS.keys()}, got: {self.encoder_name}")

        # Get set of idxs corresponding to foreground
        foreground_idxs = set(obj_mask_cropped_resized.flatten().nonzero()[0])

        # Match features; compute top-K likely models
        top_k_models = []
        top_k_scores = []
        models_copy = deepcopy(models)
        model_imgs_copy = np.array(model_imgs)
        feat_vecs = np.array(model_feat_vecs)   # (602112, 384)
        imgs = [ref_img_vis, ref_img_masked_vis]

        if self.verbose:
            print(f"{self.__class__.__name__}: Computing top-{n_candidates} candidates using encoder {self.encoder_name}...")

        n_candidates = min(n_candidates, len(models))

        if self.encoder_name == "DinoV2Encoder":
            # Build a single foreground embedding for the reference crop.
            # This is more stable than "patch-NN voting", and avoids bias toward high-frequency textures
            # (e.g., open book pages) when the object should be judged by overall shape.
            fg_mask_ref = np.zeros((H, W), dtype=bool)
            if foreground_idxs:
                fg_mask_ref_flat = np.zeros((H * W,), dtype=bool)
                fg_mask_ref_flat[list(foreground_idxs)] = True
                fg_mask_ref = fg_mask_ref_flat.reshape(H, W)
            ref_vecs_fg = ref_feat_vecs[fg_mask_ref.reshape(-1)] if fg_mask_ref.any() else ref_feat_vecs
            ref_emb = ref_vecs_fg.mean(axis=0, keepdims=True).astype(np.float32)
            ref_emb /= (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-12)

            # Candidate embeddings: mean over non-background patches.
            # Our snapshots render on a white background; drop near-white pixels as background proxy.
            def _candidate_patch_mask(img_rgb: np.ndarray) -> np.ndarray:
                # img_rgb: (H_img, W_img, 3), uint8
                if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
                    return np.ones((H, W), dtype=bool)
                bg = (img_rgb[..., 0] > 245) & (img_rgb[..., 1] > 245) & (img_rgb[..., 2] > 245)
                fg = ~bg
                fg_small = cv2.resize(fg.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                # Ensure non-empty to avoid NaNs.
                return fg_small if fg_small.any() else np.ones((H, W), dtype=bool)

            def _mask_bbox_stats(mask_hw: np.ndarray) -> tuple[float, float]:
                """
                Returns:
                    (aspect_ratio, fill_frac)
                aspect_ratio is w/h of the tight bbox; fill_frac is fg_pixels / bbox_pixels.
                """
                m = mask_hw.astype(bool)
                if not m.any():
                    return 1.0, 0.0
                ys, xs = np.where(m)
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                h = max(1, (y1 - y0 + 1))
                w = max(1, (x1 - x0 + 1))
                bbox_area = float(h * w)
                fill = float(m.sum()) / bbox_area if bbox_area > 0 else 0.0
                ar = float(w) / float(h)
                return ar, fill

            # Reference silhouette stats from the (resized) object mask.
            ref_ar, ref_fill = _mask_bbox_stats(obj_mask_cropped_resized)

            # Precompute per-candidate normalized embeddings.
            n_models = len(models_copy)
            cand_embs = np.zeros((n_models, self.encoder.embedding_dim), dtype=np.float32)
            cand_ar = np.ones((n_models,), dtype=np.float32)
            cand_fill = np.zeros((n_models,), dtype=np.float32)
            for m in range(n_models):
                mask = _candidate_patch_mask(model_imgs_copy[m])
                start = m * H * W
                end = (m + 1) * H * W
                vecs = feat_vecs[start:end]
                vecs_fg = vecs[mask.reshape(-1)] if mask.any() else vecs
                emb = vecs_fg.mean(axis=0).astype(np.float32)
                emb /= (np.linalg.norm(emb) + 1e-12)
                cand_embs[m] = emb
                ar, fill = _mask_bbox_stats(mask)
                cand_ar[m] = float(ar)
                cand_fill[m] = float(fill)

            # Cosine similarity for ranking; higher is better.
            sims = (cand_embs @ ref_emb.T).reshape(-1)

            # Add a lightweight silhouette-geometry penalty to reduce "state" mismatches like open-vs-closed books.
            # This does NOT require explicit state labels; it only uses the object's mask shape.
            # Penalize aspect ratio mismatch and how "filled" the bbox is (open books often have a very different silhouette).
            eps = 1e-6
            ar_diff = np.abs(np.log((cand_ar + eps) / (float(ref_ar) + eps)))
            fill_diff = np.abs(cand_fill - float(ref_fill))
            beta = float(os.getenv("COUSIN_DINO_AR_PENALTY", "0.20"))
            gamma = float(os.getenv("COUSIN_DINO_FILL_PENALTY", "0.10"))
            sims = sims - beta * ar_diff - gamma * fill_diff

            ranked = np.argsort(-sims)[:n_candidates]
            for ridx in ranked.tolist():
                top_k_models.append(models_copy[int(ridx)])
                top_k_scores.append(float(sims[int(ridx)]))
                imgs.append(cv2.resize(model_imgs_copy[int(ridx)], visualize_resolution))

        elif self.encoder_name == "CLIPEncoder":
            # faiss-cpu has no StandardGpuResources / index_cpu_to_gpu; only faiss-gpu does.
            # DinoV2 path above never needed Faiss, but we used to construct GPU Faiss unconditionally and broke CPU-only installs.
            dim = int(self.encoder.embedding_dim)
            cpu_index = faiss.IndexFlatL2(dim)
            search_index = cpu_index
            want_gpu = (
                str(self.device).startswith("cuda")
                and torch.cuda.is_available()
                and hasattr(faiss, "StandardGpuResources")
                and hasattr(faiss, "index_cpu_to_gpu")
            )
            if want_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    search_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                except Exception:
                    search_index = cpu_index
            if hasattr(search_index, "reset"):
                search_index.reset()
            search_index.add(np.ascontiguousarray(feat_vecs, dtype=np.float32))
            dists, idxs = search_index.search(
                np.ascontiguousarray(ref_feat_vecs, dtype=np.float32), n_candidates
            )

            # Store top-k models and distances
            for k, idx in enumerate(idxs[0]):
                top_k_models.append(models_copy[idx])
                # faiss IndexFlatL2: smaller distance is better. Convert to higher-is-better score.
                top_k_scores.append(-float(dists[0][k]))
                imgs.append(cv2.resize(model_imgs_copy[idx], visualize_resolution))

        # Record results
        if self.verbose:
            print(f"Top-{n_candidates} models: {[model.split('/')[-1] for model in top_k_models]}")
        concat_img = np.concatenate(imgs, axis=1)
        Image.fromarray(concat_img).save(f"{save_dir}/{save_prefix}_feature_matcher_results_visualization.png")
        results = {
            "k": n_candidates,
            "candidates": top_k_models,
            "candidate_scores": top_k_scores,
        }
        with open(f"{save_dir}/{save_prefix}_feature_matcher_results.json", "w+") as f:
            json.dump(results, f)

        return results
