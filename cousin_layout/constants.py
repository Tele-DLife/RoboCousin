from __future__ import annotations

"""Central paths for the copied `cousin_layout/` tree.

Match the original `digital_cousins/__init__.py` behavior:
- Define a repo root (parent directory of this folder when copied into RoboTwin)
- Default checkpoints/assets live under that repo root

Env vars can still override, but are optional.
"""

import os
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.environ.get("COUSIN_LAYOUT_REPO_DIR", str(ROOT_DIR.parent))).expanduser().resolve()

CONFIG_DEFAULT_YAML = str((ROOT_DIR / "configs" / "default.yaml").resolve())
LOCAL_KEYS_YAML = str((ROOT_DIR / "configs" / "local_keys.yaml").resolve())

def _read_checkpoint_dir_from_config() -> str | None:
    try:
        import yaml  # lazy import

        def _safe_load(path: str) -> dict:
            p = Path(path)
            if not p.exists():
                return {}
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}

        def _deep_merge(base: dict, extra: dict) -> dict:
            merged = dict(base)
            for k, v in extra.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = _deep_merge(merged[k], v)
                else:
                    merged[k] = v
            return merged

        cfg = _deep_merge(_safe_load(CONFIG_DEFAULT_YAML), _safe_load(LOCAL_KEYS_YAML))
        paths = (cfg.get("paths", {}) or {}) if isinstance(cfg, dict) else {}
        v = paths.get("checkpoint_dir", None)
        if isinstance(v, str) and v.strip():
            return str(Path(v).expanduser().resolve())
    except Exception:
        return None
    return None

_CFG_CHECKPOINT_DIR = _read_checkpoint_dir_from_config()

# Priority: env override > config absolute path > repo-relative default
CHECKPOINT_DIR = os.environ.get(
    "COUSIN_LAYOUT_CHECKPOINT_DIR",
    _CFG_CHECKPOINT_DIR or str((REPO_DIR / "checkpoints").resolve()),
)
ASSET_DIR = os.environ.get("COUSIN_LAYOUT_ASSET_DIR", str((REPO_DIR / "assets").resolve()))
OUR_OBJECTS_DIR = os.environ.get("COUSIN_LAYOUT_OUR_OBJECTS_DIR", str((REPO_DIR / "our_objects").resolve()))

