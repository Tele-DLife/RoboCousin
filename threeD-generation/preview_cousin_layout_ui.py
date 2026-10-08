"""
UI-friendly preview wrapper for SAPIEN viewer placement/size.

This script mirrors `script/preview_cousin_layout.py` but additionally injects:
  - viewer_resolutions
  - viewer_window_placement
  - viewer_minimal_ui

so that the GUI window can be made smaller and moved away from the screen center.

Defaults can be overridden via env vars:
  - ROBOTWIN_VIEWER_RES="960,540"
  - ROBOTWIN_VIEWER_PLACEMENT="bottom_left"  OR  "x,y" (top-left corner in screen pixels)
  - ROBOTWIN_VIEWER_MINIMAL_UI="1" or "0"

Note: window moving requires `xdotool` on Linux/X11. If missing, RoboTwin will warn and
the viewer may remain at the window manager's default position.

Optional room background (same presets as asset collection / ``ui_room_layout``):

  - ROBOTWIN_UI_ROOM_TYPE=livingroom|bedroom|diningroom|bathroom|kidsroom|study
    (plus living / dining / bath / kids and Chinese room-type aliases)
  - ROBOTWIN_UI_ROOM_INDEX: optional; if unset, pick a random ``*.json`` under that room folder.

If ROBOTWIN_UI_ROOM_TYPE is empty (e.g. the 3DGENERATION Step 4 empty-room mode),
no UI room meshes are loaded.
"""

from __future__ import annotations

import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml


def _parse_resolutions(s: str | None):
    if not s:
        return None
    parts = [p.strip() for p in str(s).split(",") if p.strip() != ""]
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


def _get_embodiment_defaults(repo_root: Path) -> dict:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    config_path = os.path.join(CONFIGS_PATH, "demo_randomized.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = base_cfg.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type):
        robot_file = embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"Missing embodiment files for type: {emb_type}")
        return robot_file

    if not embodiment_type:
        raise RuntimeError("No default embodiment specified in demo_randomized.yml")

    if len(embodiment_type) == 1:
        left_robot_file = get_embodiment_file(embodiment_type[0])
        right_robot_file = get_embodiment_file(embodiment_type[0])
        dual_arm_embodied = True
    elif len(embodiment_type) == 3:
        left_robot_file = get_embodiment_file(embodiment_type[0])
        right_robot_file = get_embodiment_file(embodiment_type[1])
        dual_arm_embodied = False
    else:
        raise RuntimeError(f"Embodiment config should have 1 or 3 items, got {len(embodiment_type)}")

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    defaults = {
        "left_robot_file": left_robot_file,
        "right_robot_file": right_robot_file,
        "dual_arm_embodied": dual_arm_embodied,
        "left_embodiment_config": get_embodiment_config(left_robot_file),
        "right_embodiment_config": get_embodiment_config(right_robot_file),
    }
    if "camera" in base_cfg:
        defaults["camera"] = base_cfg["camera"]
    if "data_type" in base_cfg:
        defaults["data_type"] = base_cfg["data_type"]
    return defaults


def _resolve_embodiment_from_args(args: dict) -> None:
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    embodiment_type = args.get("embodiment")
    if not embodiment_type:
        raise RuntimeError("YAML must define 'embodiment'.")

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type):
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"missing embodiment files for type {emb_type!r}")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError(f"embodiment should have length 1 or 3, got {len(embodiment_type)}")

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])


def _build_args(
    repo_root: Path,
    *,
    task_config: str | None,
    seed: int,
    episode: int,
    render_freq: int,
    load_robot: bool,
) -> dict:
    if task_config:
        config_path = repo_root / "task_config" / f"{task_config}.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        if not isinstance(args, dict):
            raise RuntimeError(f"Invalid YAML: {config_path}")
        _resolve_embodiment_from_args(args)
    else:
        args = _get_embodiment_defaults(repo_root)
        args.setdefault("use_cousin_coordinate", True)

    args["task_name"] = "preview_cousin_layout"
    args["task_config"] = task_config or "demo_randomized"
    args["save_path"] = str(repo_root / "data" / "_preview_cousin_layout")
    args["save_data"] = False
    args["need_plan"] = False
    args["eval_mode"] = False
    args["now_ep_num"] = episode
    args["seed"] = seed
    args["render_freq"] = render_freq
    args["load_robot"] = bool(load_robot)
    os.makedirs(args["save_path"], exist_ok=True)

    # Viewer placement/size tweaks (UI-friendly defaults).
    viewer_res = _parse_resolutions(os.getenv("ROBOTWIN_VIEWER_RES")) or [1280, 720]
    viewer_place = _parse_placement(os.getenv("ROBOTWIN_VIEWER_PLACEMENT")) or [1100, 500]
    minimal_ui_env = os.getenv("ROBOTWIN_VIEWER_MINIMAL_UI")
    minimal_ui = True if minimal_ui_env is None else (str(minimal_ui_env).strip() != "0")

    args["viewer_resolutions"] = viewer_res
    args["viewer_window_placement"] = viewer_place
    args["viewer_minimal_ui"] = minimal_ui

    return args


def main():
    parser = ArgumentParser()
    parser.add_argument("--render-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="Layout seed; omit for a random seed each run.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task-config", type=str, default=None)
    parser.add_argument("--input_layout", type=str, default=None)
    parser.add_argument("--no-arm", action="store_true", help="Do not load robot arms.")
    args_ns = parser.parse_args()
    seed = args_ns.seed if args_ns.seed is not None else random.randrange(2**31)

    repo_root = Path(__file__).resolve().parent.parent
    script_dir = repo_root / "script"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    if args_ns.input_layout:
        layout_path = os.path.expanduser(str(args_ns.input_layout))
        os.environ["COUSIN_RELATIVE_LAYOUT_JSON"] = layout_path
        # When an explicit cousin layout is provided, always use cousin coordinates.
        os.environ["ROBOTWIN_USE_COUSIN_COORDINATE"] = "1"
        try:
            import cousin_coordinate as _cousin_coordinate

            _cousin_coordinate.DEFAULT_RELATIVE_LAYOUT_PATH = layout_path
        except Exception:
            pass

    from envs.preview_cousin_layout import preview_cousin_layout
    from envs.utils.ui_room_preset_env import apply_ui_room_from_process_env

    cfg = _build_args(
        repo_root,
        task_config=args_ns.task_config,
        seed=seed,
        episode=args_ns.episode,
        render_freq=args_ns.render_freq,
        load_robot=not args_ns.no_arm,
    )
    if args_ns.input_layout:
        cfg["use_cousin_coordinate"] = True
    apply_ui_room_from_process_env(cfg, repo_root, log_prefix="[preview_cousin_layout_ui]")

    env = preview_cousin_layout()
    env.setup_demo(**cfg)

    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        # Hide camera frustum lines in viewport (the "wireframe planes" look),
        # especially when minimal UI is enabled and the checkbox is not visible.
        try:
            cw = getattr(viewer, "control_window", None)
            if cw is not None:
                cw.show_camera_linesets = False
        except Exception:
            pass
        print("Preview (UI wrapper). Close the viewer window to exit.")
        try:
            while not getattr(viewer, "closed", False):
                steps = int(args_ns.render_freq) if args_ns.render_freq is not None else 1
                if steps < 1:
                    steps = 1
                for _ in range(steps):
                    env.scene.step()
                env._update_render()
                viewer.render()
        finally:
            env.close_env(clear_cache=True)


if __name__ == "__main__":
    main()

