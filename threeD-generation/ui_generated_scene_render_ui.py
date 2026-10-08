"""
UI-friendly wrapper for `script/ui_generated_scene_render.py`.

This mirrors `threeD-generation/preview_cousin_layout_ui.py`:
  - viewer_resolutions
  - viewer_window_placement
  - viewer_minimal_ui

Defaults can be overridden via env vars:
  - ROBOTWIN_VIEWER_RES="1280,720"
  - ROBOTWIN_VIEWER_PLACEMENT="1100,500"  OR  "bottom_left"
  - ROBOTWIN_VIEWER_MINIMAL_UI="1" or "0"

This entrypoint always sets ``use_cousin_coordinate`` to true (via
``ROBOTWIN_USE_COUSIN_COORDINATE=1``) so behavior matches task YAML
``use_cousin_coordinate: true`` when shared code reads that flag.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path


def _parse_resolutions(s: str | None):
    if not s:
        return None
    parts = [p.strip() for p in str(s).strip().lower().replace("x", ",").split(",") if p.strip() != ""]
    if len(parts) >= 2:
        return [int(parts[0]), int(parts[1])]
    return None


def _parse_placement(s: str | None):
    if not s:
        return None
    low = str(s).strip().lower()
    if low == "bottom_left":
        return "bottom_left"
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
    if len(parts) >= 2:
        return [int(parts[0]), int(parts[1])]
    return None


def main():
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=None, help="Override render steps loop.")
    parser.add_argument("--export", action="store_true",default=False, help="Enable export mode.")
    parser.add_argument("--export-path", type=str, default=None, help="Path to export the rendered results.")
    args_ns, _unknown = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parent.parent
    script_dir = repo_root / "script"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Match preview_cousin_layout_ui.py defaults.
    viewer_res = _parse_resolutions(os.getenv("ROBOTWIN_VIEWER_RES")) or [1280, 720]
    viewer_place = _parse_placement(os.getenv("ROBOTWIN_VIEWER_PLACEMENT")) or [1100, 500]
    minimal_ui_env = os.getenv("ROBOTWIN_VIEWER_MINIMAL_UI")
    minimal_ui = True if minimal_ui_env is None else (str(minimal_ui_env).strip() != "0")

    # Force downstream script to see the exact same settings.
    os.environ["ROBOTWIN_VIEWER_RES"] = f"{int(viewer_res[0])},{int(viewer_res[1])}"
    if isinstance(viewer_place, str):
        os.environ["ROBOTWIN_VIEWER_PLACEMENT"] = viewer_place
    else:
        os.environ["ROBOTWIN_VIEWER_PLACEMENT"] = f"{int(viewer_place[0])},{int(viewer_place[1])}"
    os.environ["ROBOTWIN_VIEWER_MINIMAL_UI"] = "1" if minimal_ui else "0"

    # Align with task_config ``use_cousin_coordinate: true`` for this UI workflow.
    os.environ["ROBOTWIN_USE_COUSIN_COORDINATE"] = "1"

    print(
        f"[ui_generated_scene_render_ui] viewer_resolutions={viewer_res} "
        f"viewer_window_placement={viewer_place} viewer_minimal_ui={minimal_ui} "
        f"use_cousin_coordinate=True"
    )

    from script import ui_generated_scene_render as _impl

    # Forward into the implementation. Keep it simple: enable render
    new_args = [sys.argv[0], "--render"]
    if args_ns.steps:
        new_args.extend(["--steps", str(args_ns.steps)])
    if args_ns.export:
        new_args.append("--export")
    if args_ns.export_path:
        new_args.extend(["--export-path", args_ns.export_path])
    
    sys.argv = new_args
    _impl.main()


if __name__ == "__main__":
    main()

