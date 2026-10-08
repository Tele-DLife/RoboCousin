"""Self-contained subset for RoboTwin merging.

This package can run in an env without installing large deps as wheels by
reusing vendored source trees under `<repo>/deps/` (GroundingDINO, SAM2, etc.).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _maybe_prepend(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def bootstrap_vendored_deps(repo_root: str | None = None) -> None:
    """
    Add vendored dependency roots to sys.path.

    Call early (before importing cousin_layout.models.*) if you rely on deps/ source checkouts.
    """
    root = Path(repo_root).expanduser().resolve() if repo_root else Path(__file__).resolve().parents[1]

    deps = root / "deps"
    # GroundingDINO provides `groundingdino` top-level package
    _maybe_prepend(deps / "GroundingDINO")
    # Segment Anything 2 provides `sam2` top-level package
    _maybe_prepend(deps / "segment-anything-2")
    # DepthAnythingV2 provides `metric_depth` top-level package
    # (so we need the parent directory of `metric_depth/` on sys.path)
    _maybe_prepend(deps / "Depth-Anything-V2")
    # PerspectiveFields repo root provides `PerspectiveFields` python package
    _maybe_prepend(root / "PerspectiveFields")


# Auto-bootstrap unless explicitly disabled.
if os.environ.get("COUSIN_LAYOUT_DISABLE_DEPS_BOOTSTRAP", "").strip().lower() not in ("1", "true", "yes", "y"):
    bootstrap_vendored_deps()

