# Portions of this file are adapted from Digital Cousins:
# https://github.com/cremebrule/digital-cousins
# Modified by RoboCousin contributors.
# Licensed under Apache-2.0; see licenses/Digital-Cousins-Apache-2.0.txt.
import torch.nn
from .. import constants
import groundingdino
from groundingdino.util import box_ops
from groundingdino.util.inference import load_model, load_image, predict, annotate
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# There seems to be an issue with torch's flash attention compatibility
# (https://github.com/facebookresearch/segment-anything-2/issues/48)
# So we implement the suggested hotfix here
import sam2.modeling.sam.transformer as transformer
transformer.OLD_GPU = True
transformer.USE_FLASH_ATTN = True
transformer.MATH_KERNEL_ON = True


GDINO_CHECKPOINT_PATH = f"{constants.CHECKPOINT_DIR}/groundingdino_swint_ogc.pth"
GSAMV2_CHECKPOINT_PATH = f"{constants.CHECKPOINT_DIR}/sam2_hiera_large.pt"
GSAMV2_CONFIG = "sam2_hiera_l.yaml"


def _patch_transformers_bert_compat():
    """
    Patch transformers BertModel API drift for vendored GroundingDINO.

    GroundingDINO's bertwarper calls:
    - get_head_mask(...)
    - get_extended_attention_mask(attention_mask, input_shape, device)

    In transformers>=5:
    - BertModel.get_head_mask may be missing
    - get_extended_attention_mask expects dtype as the 3rd arg
    """
    try:
        from transformers import BertModel
    except Exception:
        return

    if not getattr(BertModel, "_cousin_head_mask_compat", False):
        if not hasattr(BertModel, "get_head_mask"):
            def _compat_get_head_mask(self, head_mask, num_hidden_layers):
                if head_mask is None:
                    return [None] * num_hidden_layers
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                else:
                    raise ValueError(
                        f"head_mask should have dim 1 or 2, got {head_mask.dim()} with shape {tuple(head_mask.shape)}"
                    )
                return head_mask.to(dtype=torch.float32)

            BertModel.get_head_mask = _compat_get_head_mask
        BertModel._cousin_head_mask_compat = True

    if not getattr(BertModel, "_cousin_ext_mask_compat", False):
        _orig_get_extended_attention_mask = BertModel.get_extended_attention_mask

        def _compat_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
            if isinstance(dtype, torch.device):
                dtype = torch.float32
            return _orig_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)

        BertModel.get_extended_attention_mask = _compat_get_extended_attention_mask
        BertModel._cousin_ext_mask_compat = True


class GroundedSAMv2(torch.nn.Module):

    def __init__(
        self,
        gdino=None,
        box_threshold=0.35,
        text_threshold=0.25,
        device="cuda",
    ):

        super().__init__()

        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        _patch_transformers_bert_compat()

        if gdino is None:
            gdino = load_model(
                f"{groundingdino.__path__[0]}/config/GroundingDINO_SwinT_OGC.py",
                GDINO_CHECKPOINT_PATH,
            )
        self.gdino = gdino
        self.gdino.eval()
        self.gdino.to(self.device)

        sam2 = build_sam2(GSAMV2_CONFIG, GSAMV2_CHECKPOINT_PATH, device=self.device)
        self.gsamv2 = SAM2ImagePredictor(sam2)

    def load_image(self, image_path):
        return load_image(image_path)

    def predict_boxes(self, img, caption):
        # Returns boxes, logits, phrases
        return predict(
            model=self.gdino,
            image=img,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

    def predict_segmentation(self, img_source, boxes, cxcywh=True, multimask_output=False):
        self.gsamv2.set_image(np.array(img_source))
        H, W, _ = img_source.shape

        if isinstance(boxes, np.ndarray):
            boxes = torch.tensor(boxes)

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) if cxcywh else boxes
        boxes_xyxy = boxes_xyxy * torch.tensor([W, H, W, H])
        masks, _, _ = self.gsamv2.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy,
            multimask_output=multimask_output,
        )

        # Make sure masks is always shape 4
        if len(masks.shape) == 3:
            masks = masks.reshape(1, *masks.shape)

        return masks.astype(bool)
