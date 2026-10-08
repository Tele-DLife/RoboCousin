"""
Resolve ``ui_room_layout_path`` from process environment (CLI / wrappers).

Environment (first match wins for layout source):

  **Explicit JSON**
  - ``ROBOTWIN_UI_ROOM_JSON``: path to a layout ``*.json`` (absolute or relative to project root).

  **Preset folder** (if ``ROBOTWIN_UI_ROOM_JSON`` is unset)
  - ``ROBOTWIN_UI_ROOM_TYPE``: ``livingroom`` | ``bedroom`` | ``diningroom`` |
    ``bathroom`` | ``kidsroom`` | ``study`` (or aliases / 中文).
  - ``ROBOTWIN_UI_ROOM_INDEX``: optional int; if unset, pick uniformly at random among ``*.json``.

  **Optional overrides** (after a layout path is chosen)
  - ``ROBOTWIN_UI_ROOM_KEEP_RANDOM_BACKGROUND``: if truthy, do not force
    ``domain_randomization.random_background=false``.
  - ``ROBOTWIN_UI_ROOM_COLLISION``: if set, truthy/falsey sets ``ui_room_layout_collision``.
  - ``ROBOTWIN_UI_ROOM_ALIGN_OFFSET_DEG``: float, sets ``ui_room_layout_align_offset_deg``.
  - ``ROBOTWIN_UI_ROOM_VERBOSE``: truthy/falsey, sets ``ui_room_layout_verbose``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path


def _truthy(s: str | None) -> bool:
    if s is None:
        return False
    return str(s).strip().lower() in ("1", "true", "yes", "on")


def _parse_optional_env_bool(key: str) -> bool | None:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def _parse_optional_env_float(key: str) -> float | None:
    v = os.environ.get(key)
    if v is None or str(v).strip() == "":
        return None
    return float(str(v).strip())


def apply_ui_room_from_process_env(
    cfg: dict,
    project_root: Path,
    *,
    log_prefix: str = "[ui_room_preset]",
) -> bool:
    """
    Mutates ``cfg`` in place. Returns True if ``ui_room_layout_path`` was set.
    """
    project_root = Path(project_root)
    chosen: Path | None = None

    json_raw = (os.environ.get("ROBOTWIN_UI_ROOM_JSON") or "").strip()
    if json_raw:
        p = Path(json_raw)
        if not p.is_file():
            p = project_root / json_raw
        if not p.is_file():
            print(f"{log_prefix} ROBOTWIN_UI_ROOM_JSON not found: {json_raw!r}, ignoring.")
            return False
        chosen = p.resolve()
        print(f"{log_prefix} UI room JSON (env): {chosen}")
    else:
        raw = (os.environ.get("ROBOTWIN_UI_ROOM_TYPE") or "").strip()
        if not raw:
            return False
        key = raw.lower()
        aliases = {
            "livingroom": "livingroom",
            "living": "livingroom",
            "客厅": "livingroom",
            "bedroom": "bedroom",
            "卧室": "bedroom",
            "diningroom": "diningroom",
            "dining": "diningroom",
            "餐厅": "diningroom",
            "bathroom": "bathroom",
            "bath": "bathroom",
            "卫生间": "bathroom",
            "kidsroom": "kidsroom",
            "kids room": "kidsroom",
            "kids": "kidsroom",
            "儿童房": "kidsroom",
            "study": "study",
            "study room": "study",
            "书房": "study",
        }
        sub = aliases.get(key) or aliases.get(raw)
        if not sub:
            print(f"{log_prefix} Unknown ROBOTWIN_UI_ROOM_TYPE={raw!r}, ignoring.")
            return False
        room_dir = project_root / "envs" / "room_config" / sub
        if not room_dir.is_dir():
            print(f"{log_prefix} Room preset dir missing: {room_dir}, ignoring.")
            return False
        jsons = sorted(room_dir.glob("*.json"))
        if not jsons:
            print(f"{log_prefix} No *.json in {room_dir}, ignoring.")
            return False
        env_idx = os.environ.get("ROBOTWIN_UI_ROOM_INDEX")
        if env_idx is not None and str(env_idx).strip() != "":
            try:
                idx = int(str(env_idx).strip()) % len(jsons)
            except ValueError:
                idx = random.randrange(len(jsons))
        else:
            idx = random.randrange(len(jsons))
        chosen = jsons[idx].resolve()
        print(f"{log_prefix} UI room: {sub} -> [{idx}] {chosen.name}")

    cfg["ui_room_layout_path"] = str(chosen)

    if not _truthy(os.environ.get("ROBOTWIN_UI_ROOM_KEEP_RANDOM_BACKGROUND")):
        dr = dict(cfg.get("domain_randomization") or {})
        dr["random_background"] = False
        cfg["domain_randomization"] = dr

    vb = _parse_optional_env_bool("ROBOTWIN_UI_ROOM_COLLISION")
    if vb is not None:
        cfg["ui_room_layout_collision"] = vb
    else:
        cfg.setdefault("ui_room_layout_collision", False)

    ao = _parse_optional_env_float("ROBOTWIN_UI_ROOM_ALIGN_OFFSET_DEG")
    if ao is not None:
        cfg["ui_room_layout_align_offset_deg"] = ao

    vl = _parse_optional_env_bool("ROBOTWIN_UI_ROOM_VERBOSE")
    if vl is not None:
        cfg["ui_room_layout_verbose"] = vl

    return True
