"""
Interactive preview: cousin clutter + pick_up-style target object.

**Do not use** `collect_data_complete.py preview_cousin_layout ...` for this —
that path forces `save_data`, traj merge, and per-episode SUCCESS/FAIL semantics.

Run this script instead (no HDF5 / traj / cache pipeline):

  python script/preview_cousin_layout.py --task-config demo_complete --seed 0

Same cousin / embodiment / domain_randomization as the YAML, but always:
  save_data=False, need_plan=False (viewer only).

Without --task-config, embodiment + camera fall back to `task_config/demo_randomized.yml`.
"""
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import yaml


def _get_embodiment_defaults():
    """Embodiment + camera from demo_randomized.yml (when no --task-config)."""
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
            raise RuntimeError("Missing embodiment files for type: {}".format(emb_type))
        return robot_file

    if not embodiment_type:
        raise RuntimeError("No default embodiment specified in demo_randomized.yml")

    if len(embodiment_type) == 1:
        left_robot_file = get_embodiment_file(embodiment_type[0])
        right_robot_file = get_embodiment_file(embodiment_type[0])
        dual_arm_embodied = True
        embodiment_dis = None
    elif len(embodiment_type) == 3:
        left_robot_file = get_embodiment_file(embodiment_type[0])
        right_robot_file = get_embodiment_file(embodiment_type[1])
        embodiment_dis = embodiment_type[2]
        dual_arm_embodied = False
    else:
        raise RuntimeError("Embodiment config should have 1 or 3 items, got {}".format(len(embodiment_type)))

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    left_embodiment_config = get_embodiment_config(left_robot_file)
    right_embodiment_config = get_embodiment_config(right_robot_file)

    defaults = {
        "left_robot_file": left_robot_file,
        "right_robot_file": right_robot_file,
        "dual_arm_embodied": dual_arm_embodied,
        "left_embodiment_config": left_embodiment_config,
        "right_embodiment_config": right_embodiment_config,
    }
    if embodiment_dis is not None:
        defaults["embodiment_dis"] = embodiment_dis

    if "camera" in base_cfg:
        defaults["camera"] = base_cfg["camera"]
    if "data_type" in base_cfg:
        defaults["data_type"] = base_cfg["data_type"]
    for key in ("pcd_down_sample_num", "pcd_crop", "pcd_crop_bbox", "use_seed", "save_freq", "dual_arm"):
        if key in base_cfg:
            defaults[key] = base_cfg[key]

    return defaults


def _resolve_embodiment_from_args(args: dict) -> None:
    """Fill left/right robot paths and configs from args['embodiment'] (mutates args)."""
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    embodiment_type = args.get("embodiment")
    if not embodiment_type:
        raise RuntimeError("YAML must define 'embodiment' (same as collect_data_complete).")

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(emb_type):
        robot_file = _embodiment_types[emb_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files for type %r" % (emb_type,))
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
        raise RuntimeError("embodiment should have length 1 or 3, got %d" % len(embodiment_type))

    def get_embodiment_config(robot_file):
        robot_config_file = os.path.join(robot_file, "config.yml")
        with open(robot_config_file, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])


def _apply_cousin_layout_from_args(args: dict, repo_root: Path) -> None:
    """
    If task YAML provides `cousin_relative_layout_json`, export it to process env
    so script/cousin_coordinate.py resolves layout from config instead of defaults.
    """
    raw = args.get("cousin_relative_layout_json", None)
    if raw is None:
        return
    s = str(raw).strip()
    if not s:
        return
    p = Path(os.path.expanduser(s))
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Invalid cousin_relative_layout_json in task config: {p}")
    os.environ["COUSIN_RELATIVE_LAYOUT_JSON"] = str(p)


def _build_preview_args(
    repo_root: Path,
    *,
    task_config: str | None,
    seed: int,
    episode: int,
    render_freq: int,
    load_robot: bool,
    input_layout: str | None,
    no_physics: bool,
) -> dict:
    """Args dict for preview_cousin_layout.setup_demo — same YAML as collection when task_config is set."""
    script_dir = repo_root / "script"
    if task_config:
        config_path = repo_root / "task_config" / f"{task_config}.yml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Task config not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
        if not isinstance(args, dict):
            raise RuntimeError(f"Invalid YAML: {config_path}")
        _apply_cousin_layout_from_args(args, repo_root)
        # _resolve_embodiment_from_args(args)
    else:
        args = _get_embodiment_defaults()
        args.setdefault("use_cousin_coordinate", True)
        args.setdefault(
            "domain_randomization",
            {
                "cluttered_table": True,
                "random_background": False,
                "clean_background_rate": 0,
                "random_light": False,
                "crazy_random_light_rate": 0,
                "random_table_height": 0,
                "random_head_camera_dis": 0,
            },
        )

    use_cousin = bool(args.get("use_cousin_coordinate", False))
    if use_cousin and str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    # Never run data-collection pipeline from this entrypoint.
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

    if input_layout:
        args["use_cousin_coordinate"] = True
        # A preview of an explicit cousin layout should use the snapshot-estimated
        # yaw unless the loaded task config intentionally overrides it.
        args.setdefault("cousin_random_yaw", False)
        args.setdefault("cousin_snapshot_spin_sign", -1.0)
        args.setdefault("cousin_cam_azimuth_sign", 1.0)
        args.setdefault("cousin_yaw_add_pi", 1.0)
        args.setdefault("cousin_yaw_mirror_mode", "pi_minus")
        args.setdefault("cousin_instance_score_pool_top_k", 3)

    if no_physics:
        # `--no-physics` must freeze the scene from initial spawn, not just stop
        # stepping in the viewer loop after setup_demo() has already settled it.
        args["settle_before_planning"] = False
        args["skip_stable_check"] = True

    os.makedirs(args["save_path"], exist_ok=True)
    return args


def main(
    render_freq: int = 1,
    seed: int = 0,
    episode: int = 0,
    task_config: str | None = None,
    input_layout: str | None = None,
    no_physics: bool = False,
    load_robot: bool = True,
):
    repo_root = Path(__file__).resolve().parent.parent
    script_dir = repo_root / "script"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    if input_layout:
        layout_path = os.path.expanduser(str(input_layout))
        os.environ["COUSIN_RELATIVE_LAYOUT_JSON"] = layout_path
        try:
            import cousin_coordinate as _cousin_coordinate

            _cousin_coordinate.DEFAULT_RELATIVE_LAYOUT_PATH = layout_path
        except Exception:
            # Best-effort: env var is the primary override for cousin_coordinate readers.
            pass

    from envs.preview_cousin_layout import preview_cousin_layout

    args = _build_preview_args(
        repo_root,
        task_config=task_config,
        seed=seed,
        episode=episode,
        render_freq=render_freq,
        load_robot=load_robot,
        input_layout=input_layout,
        no_physics=no_physics,
    )

    env = preview_cousin_layout()
    env.setup_demo(**args)

    viewer = getattr(env, "viewer", None)
    if viewer is not None:
        print("Preview (no data collection). Close the viewer window to exit.")
        if task_config:
            print(f"  task_config={task_config!r}  seed={seed}  episode={episode}")
        print(f"  load_robot={bool(load_robot)}")
        if no_physics:
            print("  no_physics=True (freeze initial placement; no scene.step())")
        try:
            while not getattr(viewer, "closed", False):
                if not no_physics:
                    # Advance physics; otherwise objects won't fall/move.
                    # Use render_freq as "N sim steps per rendered frame" in this preview script.
                    steps = int(render_freq) if render_freq is not None else 1
                    if steps < 1:
                        steps = 1
                    for _ in range(steps):
                        env.scene.step()
                env._update_render()
                viewer.render()
        finally:
            env.close_env(clear_cache=True)
    else:
        for _ in range(100):
            env.scene.step()
            env._update_render()
        env.close_env(clear_cache=True)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Preview cousin layout without collect_data_complete (no HDF5 / traj / episode FAIL logic)."
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        metavar="NAME",
        help="Load task_config/{NAME}.yml (e.g. demo_complete) like collect_data_complete; omit to use demo_randomized embodiment defaults only.",
    )
    parser.add_argument("--render-freq", type=int, default=1, help="Render every N sim steps in viewer.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for cousin_coordinate and env.")
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index passed as now_ep_num (cousin layout selection).",
    )
    parser.add_argument(
        "--input_layout",
        dest="input_layout",
        type=str,
        default=None,
        metavar="PATH",
        help="Override cousin relative layout JSON (replaces cousin_coordinate.DEFAULT_RELATIVE_LAYOUT_PATH).",
    )
    parser.add_argument(
        "--no-arm",
        action="store_true",
        help="Do not load robot arms (table/objects/cameras only).",
    )
    parser.add_argument(
        "--no-physics",
        action="store_true",
        help="Freeze initial placement (skip scene.step()) so you can inspect the layout without dynamics.",
    )
    parsed = parser.parse_args()
    main(
        render_freq=parsed.render_freq,
        seed=parsed.seed,
        episode=parsed.episode,
        task_config=parsed.task_config,
        input_layout=parsed.input_layout,
        no_physics=parsed.no_physics,
        load_robot=not parsed.no_arm,
    )

# python script/preview_cousin_layout.py --task-config demo_complete --seed 0 