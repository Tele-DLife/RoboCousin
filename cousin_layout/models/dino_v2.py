# Portions of this file are adapted from Digital Cousins:
# https://github.com/cremebrule/digital-cousins
# Modified by RoboCousin contributors.
# Licensed under Apache-2.0; see licenses/Digital-Cousins-Apache-2.0.txt.
import numpy as np

import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode, Normalize
from pathlib import Path

from .visual_encoder import VisualEncoder


class DinoV2Encoder(VisualEncoder):

    BACKBONE_ARCHES = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }

    EMBEDDING_DIMS = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(
            self,
            backbone_size="small",
            aspect_ratio=(4, 3),
            feature_scale=1,
            batch_size=32,
            preprocess_batch_size=None,
            device="cuda",
    ):
        """
        Args:
            backbone_size (str): Size of the backbone model. Valid options are: "small", "base", "large", "giant".
                Default is "small", which we've found to work empirically better than all other models
                (and, conveniently, is also the fastest!)
            aspect_ratio (2-tuple): (W, H) aspect ratio to convert input image into during preprocessing phase
            feature_scale (int): Scaling factor for the outputted encoded feature patches. This will scale the
                dimensions of the outputted feature patch.
            batch_size (None or int): If specified, batch size to use when computing features for a given set of images.
                This is used to avoid OOM errors with limited VRAM
            preprocess_batch_size (None or int): If specified, batch size to use when preprocessing images before
                passing them into the encoder. This is used to avoid OOM errors with limited VRAM
            device (str): device to store tensors on. Default is "cuda"
        """
        # Always run super first
        super().__init__(
            batch_size=batch_size,
            preprocess_batch_size=preprocess_batch_size,
            device=device,
        )

        # Store additional
        self.aspect_ratio = aspect_ratio
        self.feature_scale = int(feature_scale)

        # Sanity check backbone size
        assert backbone_size in self.BACKBONE_ARCHES, \
            f"Got invalid dinov2 backbone size: {backbone_size}. Valid options are: {self.BACKBONE_ARCHES.keys()}"
        backbone_name = f"dinov2_{self.BACKBONE_ARCHES[backbone_size]}"
        self.backbone_size = backbone_size

        # Load the encoder backbone
        # Force local-only torch.hub loading (no GitHub / network).
        def _try_load_local() -> torch.nn.Module | None:
            try:
                hub_dir = Path(torch.hub.get_dir())
            except Exception:
                return None
            candidates = [
                hub_dir / "facebookresearch_dinov2_main",
                hub_dir / "facebookresearch_dinov2_master",
            ]
            for repo_path in candidates:
                if repo_path.is_dir():
                    try:
                        return torch.hub.load(
                            repo_or_dir=str(repo_path),
                            model=backbone_name,
                            source="local",
                        )
                    except Exception:
                        continue
            return None

        backbone = _try_load_local()
        if backbone is None:
            raise RuntimeError(
                "Failed to load DINOv2 from local torch.hub cache. "
                "Expected '~/.cache/torch/hub/facebookresearch_dinov2_main' (or *_master) with hubconf.py present. "
                "Network loading is disabled."
            )
        self.backbone = backbone
        self.backbone.eval()
        self.backbone.to(self.device)

    @property
    def feature_width(self):
        """
        Returns:
            int: Width of outputted feature maps
        """
        return 14 * self.aspect_ratio[0] * self.feature_scale

    @property
    def feature_height(self):
        """
        Returns:
            int: Height of outputted feature maps
        """
        return 14 * self.aspect_ratio[1] * self.feature_scale

    @property
    def embedding_dim(self):
        return self.EMBEDDING_DIMS[self.backbone_size]

    def preprocess(self, x):
        # Standardize shape to be 4-dim (B, H, W, C)
        if len(x.shape) < 4:
            x = x.reshape(1, *x.shape)

        # Convert from range [0, 255] -> [0.0, 1.0]
        x = torch.tensor(x, requires_grad=False) / 255.0

        # Permute, resize, normalize
        new_width = self.feature_width * 14
        new_height = self.feature_height * 14
        return torchvision.transforms.Compose(
            [
                torchvision.ops.Permute([0, 3, 1, 2]),
                Resize((new_height, new_width), interpolation=InterpolationMode.BICUBIC),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )(x)

    def forward(self, x):
        _, _, H, W = x.shape
        return self.backbone.forward_features(x)["x_norm_patchtokens"].view(-1, H // 14, W // 14, self.embedding_dim)
