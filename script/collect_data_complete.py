import sys

sys.path.append("./")

import sapien.core as sapien
from sapien.render import clear_cache
from collections import OrderedDict
import pdb
from envs import *
import yaml
import importlib
import json
import traceback
import os
import time
import random
import subprocess
from argparse import ArgumentParser
from pathlib import Path

from envs._GLOBAL_CONFIGS import ROOT_PATH
from envs.utils.ui_room_preset_env import apply_ui_room_from_process_env
try:
    from path_config import DATA_INCLUDE_FAILURE_DIR
except ModuleNotFoundError:
    from script.path_config import DATA_INCLUDE_FAILURE_DIR

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

ROOM_DIRS = {
    "bedroom": "bedroom",
    "livingroom": "livingroom",
    "living": "livingroom",
    "diningroom": "diningroom",
    "dining": "diningroom",
    "bathroom": "bathroom",
    "bath": "bathroom",
    "kidsroom": "kidsroom",
    "kids": "kidsroom",
    "study": "study",
    "卧室": "bedroom",
    "客厅": "livingroom",
    "餐厅": "diningroom",
    "卫生间": "bathroom",
    "儿童房": "kidsroom",
    "书房": "study",
}

QWEN_ROOM_PROMPTS = {
    "bedroom": "卧室",
    "livingroom": "客厅",
    "diningroom": "餐厅",
    "bathroom": "卫生间",
    "kidsroom": "儿童房",
    "study": "书房",
}


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def is_intrinsic_dual_arm_embodiment(embodiment_cfg):
    """Whether this embodiment config is a native dual-arm robot."""
    return bool((embodiment_cfg or {}).get("dual_arm", False))


def _resolve_room_layout_path(
    project_root: Path,
    *,
    room_type: str | None,
    room_layout_json: str | None,
    room_layout_index: int | None,
    generate_room_layout: bool,
    room_prompt: str | None,
) -> Path:
    if generate_room_layout:
        gen_script = project_root / "envs" / "utils" / "layout_generator.py"
        if not gen_script.is_file():
            raise FileNotFoundError(f"layout_generator not found: {gen_script}")
        if room_type is None:
            raise SystemExit("--generate-room-layout requires --room-type")
        rk = ROOM_DIRS.get(room_type.strip().lower(), ROOM_DIRS.get(room_type.strip(), room_type.strip()))
        prompt = room_prompt or QWEN_ROOM_PROMPTS.get(rk, room_type)
        print(f"[collect_data_complete] Generating room layout via Qwen, prompt={prompt!r}")
        subprocess.run(
            [sys.executable, str(gen_script), "--prompt", prompt],
            cwd=str(project_root),
            check=True,
        )
        out = project_root / "envs" / "room_config" / "ui_generated_layout.json"
        if not out.is_file():
            raise FileNotFoundError(f"Expected generator output missing: {out}")
        return out.resolve()

    if room_layout_json:
        p = Path(room_layout_json)
        if not p.is_file():
            p = project_root / room_layout_json
        if not p.is_file():
            raise FileNotFoundError(f"--room-layout-json not found: {room_layout_json}")
        return p.resolve()

    if room_type is None:
        raise SystemExit("Specify --room-type, --room-layout-json, or --generate-room-layout.")

    key = room_type.strip()
    sub = ROOM_DIRS.get(key.lower(), ROOM_DIRS.get(key))
    if not sub:
        raise SystemExit(f"Unknown --room-type={room_type!r}")
    room_dir = project_root / "envs" / "room_config" / sub
    if not room_dir.is_dir():
        raise FileNotFoundError(f"No preset layouts under: {room_dir}")
    jsons = sorted(room_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"No *.json in {room_dir}")
    if room_layout_index is None:
        idx = random.randrange(len(jsons))
    else:
        idx = int(room_layout_index) % len(jsons)
    chosen = jsons[idx].resolve()
    print(f"[collect_data_complete] Using room layout {chosen} ([{idx}] / {len(jsons)} files)")
    return chosen


def _apply_ui_room_from_cli(
    cfg: dict,
    project_root: Path,
    *,
    room_type: str | None,
    room_layout_json: str | None,
    room_layout_index: int | None,
    generate_room_layout: bool,
    room_prompt: str | None,
    room_collision: bool,
    room_keep_random_background: bool,
    room_align_offset_deg: float | None,
    room_layout_verbose: bool,
) -> bool:
    has_source = bool(room_type or room_layout_json or generate_room_layout)
    if not has_source:
        return False

    layout_path = _resolve_room_layout_path(
        project_root,
        room_type=room_type,
        room_layout_json=room_layout_json,
        room_layout_index=room_layout_index,
        generate_room_layout=generate_room_layout,
        room_prompt=room_prompt,
    )

    cfg["ui_room_layout_path"] = str(layout_path)
    cfg["ui_room_layout_collision"] = bool(room_collision)
    cfg["ui_room_layout_verbose"] = bool(room_layout_verbose)
    if room_align_offset_deg is not None:
        cfg["ui_room_layout_align_offset_deg"] = float(room_align_offset_deg)

    if not room_keep_random_background:
        dr = dict(cfg.get("domain_randomization") or {})
        dr["random_background"] = False
        cfg["domain_randomization"] = dr

    return True


def _apply_cousin_layout_from_config(args: dict, project_root: Path) -> None:
    """
    If YAML provides `cousin_relative_layout_json`, export it to process env so
    script/cousin_coordinate.py can read a single source of truth.
    """
    raw = args.get("cousin_relative_layout_json", None)
    if raw is None:
        return
    s = str(raw).strip()
    if not s:
        return

    p = Path(os.path.expanduser(s))
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise SystemExit(f"Invalid cousin_relative_layout_json (file not found): {p}")

    os.environ["COUSIN_RELATIVE_LAYOUT_JSON"] = str(p)
    print(f"[collect_data_complete] COUSIN_RELATIVE_LAYOUT_JSON <- {p}")


def main(
    task_name=None,
    task_config=None,
    render_freq_override=None,
    save_on_viewer_close=False,
    use_cousin_coordinate=False,
    room_type=None,
    room_layout_json=None,
    room_layout_index=None,
    generate_room_layout=False,
    room_prompt=None,
    room_collision=False,
    room_keep_random_background=False,
    room_align_offset_deg=None,
    room_layout_verbose=True,
):
    config_path = Path(ROOT_PATH) / "task_config" / f"{task_config}.yml"

    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    _apply_cousin_layout_from_config(args, Path(ROOT_PATH))

    # Optional: pick UI room layout via env (no per-room YAML merge needed).
    apply_ui_room_from_process_env(args, Path(ROOT_PATH), log_prefix="[collect_data_complete]")
    # Optional: direct CLI room controls (override env if both are present).
    _apply_ui_room_from_cli(
        args,
        Path(ROOT_PATH),
        room_type=room_type,
        room_layout_json=room_layout_json,
        room_layout_index=room_layout_index,
        generate_room_layout=generate_room_layout,
        room_prompt=room_prompt,
        room_collision=room_collision,
        room_keep_random_background=room_keep_random_background,
        room_align_offset_deg=room_align_offset_deg,
        room_layout_verbose=room_layout_verbose,
    )

    args['task_name'] = task_name
    # Allow config YAML to enable cousin coordinate if CLI did not
    use_cousin_coordinate = use_cousin_coordinate or args.get("use_cousin_coordinate", False)
    cousin_anchor_layout = bool(args.get("cousin_anchor_layout", False))
    # So tasks can "import cousin_coordinate" when cousin placement is used
    if (use_cousin_coordinate or cousin_anchor_layout) and parent_directory not in sys.path:
        sys.path.insert(0, parent_directory)

    # Optional: override render frequency for interactive visualization (Viewer UI).
    # - 0: no viewer (fastest; default in most task configs)
    # - 1: render every step (slowest)
    # - N: render every N steps
    if render_freq_override is None:
        env_override = os.getenv("ROBOTWIN_RENDER_FREQ")
        if env_override is not None and str(env_override).strip() != "":
            try:
                render_freq_override = int(env_override)
            except ValueError:
                raise SystemExit(f"Invalid ROBOTWIN_RENDER_FREQ={env_override!r}, expected an int.")
    if render_freq_override is not None:
        args["render_freq"] = int(render_freq_override)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "missing embodiment files"
        return robot_file

    if len(embodiment_type) == 1:
        robot_file = get_embodiment_file(embodiment_type[0])
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file

        embodiment_cfg = get_embodiment_config(robot_file)
        intrinsic_dual_arm = is_intrinsic_dual_arm_embodiment(embodiment_cfg)
        task_wants_dual_arm = bool(args.get("dual_arm", True))

        if task_wants_dual_arm and not intrinsic_dual_arm:
            # Task wants dual-arm behavior, but embodiment is single-arm.
            # Build two mirrored single-arm entities (left/right) instead of shared dual-arm entity.
            args["dual_arm_embodied"] = False
            args["embodiment_dis"] = float(args.get("single_embodiment_dual_arm_dis", 0.65))
        else:
            args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print(
        "\033[94mData Save Root:\033[0m "
        + str(DATA_INCLUDE_FAILURE_DIR / str(args["task_name"]) / task_config / embodiment_name)
    )
    if use_cousin_coordinate:
        print("\033[95mUse cousin coordinate (specified placement):\033[0m True (script/cousin_coordinate.py)")
    if cousin_anchor_layout:
        print("\033[95mCousin anchor layout:\033[0m True (base spawn anchor + relative clutter)")
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args['task_config'] = task_config
    # Per-embodiment root: .../<task>/<task_config>/<embodiment>/{success,failure}/...
    args["save_path"] = str(
        DATA_INCLUDE_FAILURE_DIR / str(args["task_name"]) / args["task_config"] / embodiment_name
    )
    args["save_on_viewer_close"] = save_on_viewer_close
    args["use_cousin_coordinate"] = use_cousin_coordinate

    # Strict precheck for cousin target label mapping BEFORE any env setup/viewer initialization.
    if use_cousin_coordinate and task_name == "pick_up" and not cousin_anchor_layout:
        try:
            import cousin_coordinate

            labels = args.get("cousin_target_labels") or ["clock"]
            if isinstance(labels, str):
                labels = [labels]

            for lb in labels:
                model_dir = cousin_coordinate._find_matching_our_object_dir(
                    Path(ROOT_PATH),
                    str(lb),
                    strict=True,
                    actor_only=True,
                )
                if model_dir is None:
                    raise RuntimeError(
                        f"[precheck] cousin target label '{lb}' failed to map to our_assets/actor"
                    )
            print(f"\033[92m[Cousin Precheck] target labels mapped successfully: {labels}\033[0m")
        except Exception as e:
            raise SystemExit(f"[Cousin Precheck] FAILED before env startup: {e}")

    if cousin_anchor_layout and task_name == "pick_up":
        try:
            import cousin_coordinate

            labels = args.get("cousin_target_labels") or args.get("custom_objects") or ["earbud case"]
            if isinstance(labels, str):
                labels = [labels]
            clutter_labels = []
            for spec in args.get("cousin_anchor_clutter") or []:
                if isinstance(spec, dict) and spec.get("label"):
                    clutter_labels.append(str(spec["label"]))
            for lb in list(labels) + clutter_labels:
                class_dir = cousin_coordinate.resolve_custom_asset_class_dir(
                    Path(ROOT_PATH),
                    str(lb),
                    actor_only=(lb in labels),
                )
                if class_dir is None:
                    raise RuntimeError(
                        f"[precheck] cousin anchor label '{lb}' failed to map to our_assets"
                    )
            print(
                "\033[92m[Cousin Anchor Precheck] labels mapped successfully: "
                f"target={labels}, clutter={clutter_labels}\033[0m"
            )
        except Exception as e:
            raise SystemExit(f"[Cousin Anchor Precheck] FAILED before env startup: {e}")

    if use_cousin_coordinate and task_name in ("place_a2b_left", "place_a2b_right") and not cousin_anchor_layout:
        try:
            import cousin_coordinate

            target_labels = args.get("cousin_target_labels") or args.get("custom_objects") or []
            if isinstance(target_labels, str):
                target_labels = [target_labels]
            ref_labels = args.get("cousin_reference_labels") or []
            if isinstance(ref_labels, str):
                ref_labels = [ref_labels]
            if not ref_labels:
                for spec in args.get("cousin_anchor_clutter") or []:
                    if isinstance(spec, dict) and spec.get("label"):
                        ref_labels = [str(spec["label"])]
                        break
            if not target_labels:
                raise RuntimeError("cousin_target_labels is required for place_a2b cousin mode")
            if not ref_labels:
                raise RuntimeError(
                    "cousin_reference_labels or cousin_anchor_clutter is required for place_a2b cousin mode"
                )
            for lb in target_labels:
                model_dir = cousin_coordinate._find_matching_our_object_dir(
                    Path(ROOT_PATH),
                    str(lb),
                    strict=True,
                    actor_only=True,
                )
                if model_dir is None:
                    raise RuntimeError(
                        f"[precheck] cousin place_a2b target label '{lb}' failed to map to our_assets/actor"
                    )
            for lb in ref_labels:
                model_dir = cousin_coordinate._find_matching_our_object_dir(
                    Path(ROOT_PATH),
                    str(lb),
                    strict=True,
                    actor_only=False,
                )
                if model_dir is None:
                    raise RuntimeError(
                        f"[precheck] cousin place_a2b reference label '{lb}' failed to map to our_assets"
                    )
            print(
                "\033[92m[Cousin Precheck] place_a2b labels mapped successfully: "
                f"A={target_labels}, B={ref_labels}\033[0m"
            )
        except Exception as e:
            raise SystemExit(f"[Cousin Precheck] FAILED before env startup: {e}")

    if cousin_anchor_layout and task_name in ("place_a2b_left", "place_a2b_right"):
        try:
            import cousin_coordinate

            target_labels = args.get("cousin_target_labels") or args.get("custom_objects") or []
            if isinstance(target_labels, str):
                target_labels = [target_labels]
            ref_labels = args.get("cousin_reference_labels") or []
            if isinstance(ref_labels, str):
                ref_labels = [ref_labels]
            clutter_labels = []
            for spec in args.get("cousin_anchor_clutter") or []:
                if isinstance(spec, dict) and spec.get("label"):
                    clutter_labels.append(str(spec["label"]))
            if not ref_labels and clutter_labels:
                ref_labels = [clutter_labels[0]]
            if not target_labels:
                raise RuntimeError("cousin_target_labels is required for place_a2b anchor layout")
            if not ref_labels:
                raise RuntimeError(
                    "cousin_reference_labels or cousin_anchor_clutter is required for place_a2b anchor layout"
                )
            for lb in list(target_labels) + list(ref_labels):
                class_dir = cousin_coordinate.resolve_custom_asset_class_dir(
                    Path(ROOT_PATH),
                    str(lb),
                    actor_only=(lb in target_labels),
                )
                if class_dir is None:
                    raise RuntimeError(
                        f"[precheck] cousin anchor place_a2b label '{lb}' failed to map to our_assets"
                    )
            print(
                "\033[92m[Cousin Anchor Precheck] place_a2b labels mapped successfully: "
                f"A={target_labels}, B={ref_labels}\033[0m"
            )
        except Exception as e:
            raise SystemExit(f"[Cousin Anchor Precheck] FAILED before env startup: {e}")

    # Create task env only after precheck passes.
    task = class_decorator(task_name)
    run(task, args)


def run(TASK_ENV, args):
    success_count = 0
    fail_count = 0

    print(f"Task Name: \033[34m{args['task_name']}\033[0m")
    print("\033[93m" + "[Start Complete Data Collection (including failures)]" + "\033[0m")

    # Determine the actual save directory that the task will use
    # Some tasks (like shake_bottle) may modify save_dir during load_actors
    # We do a temporary setup to get the actual save_dir, then close it
    original_save_path = args["save_path"]
    actual_save_path = original_save_path
    
    try:
        # Do a temporary setup to get the actual save_dir that the task will use
        # This is necessary because some tasks modify save_dir during initialization
        temp_args = args.copy()
        temp_args["save_data"] = False  # Don't save data during temp setup
        temp_args["now_ep_num"] = 0
        temp_args["seed"] = 0
        # Probe save_dir only: avoid opening a second viewer before the real episode loop
        # (full scene still runs; user render_freq applies in the loop below).
        temp_args["render_freq"] = 0
        
        TASK_ENV.setup_demo(**temp_args)
        actual_save_path = TASK_ENV.save_dir
        
        # Clean up temporary setup (close_env already closes the viewer when present)
        try:
            TASK_ENV.close_env()
        except:
            pass
            
    except Exception as e:
        # If temp setup fails, use original path
        print(f"\033[93mWarning: Could not determine actual save_dir, using original: {original_save_path}\033[0m")
        actual_save_path = original_save_path
    
    # Update args["save_path"] to use the actual path
    args["save_path"] = actual_save_path
    if actual_save_path != original_save_path:
        print(f"\033[93mNote: Task modified save_dir to: {actual_save_path} (original: {original_save_path})\033[0m")

    # Create save directory: <task>/<config>/<embodiment>/{success,failure}
    os.makedirs(args["save_path"], exist_ok=True)
    print(f"\033[94mEmbodiment data directory:\033[0m {args['save_path']}")
    success_save_dir = os.path.join(args["save_path"], "success")
    failure_save_dir = os.path.join(args["save_path"], "failure")
    for split_dir in (success_save_dir, failure_save_dir):
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(os.path.join(split_dir, "_traj_data"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "video"), exist_ok=True)

    # Initialize scene_info.json
    info_file_path = os.path.join(args["save_path"], "scene_info.json")
    if not os.path.exists(info_file_path):
        with open(info_file_path, "w", encoding="utf-8") as file:
            json.dump({}, file, ensure_ascii=False)
    
    with open(info_file_path, "r", encoding="utf-8") as file:
        info_db = json.load(file)

    # Check existing saved episodes to resume from the last episode + 1.
    # scene_info is only written after data is saved, so episode_num counts saved data.
    last_episode_idx = -1
    for key in info_db.keys():
        if not isinstance(key, str) or not key.startswith("episode_"):
            continue
        try:
            ep_idx = int(key.split("_", 1)[1])
        except (TypeError, ValueError):
            continue
        last_episode_idx = max(last_episode_idx, ep_idx)
    
    # Start from last episode + 1
    start_episode_idx = last_episode_idx + 1
    episode_idx = start_episode_idx
    
    # Find the last used seed from existing episodes to avoid seed conflicts
    last_seed = -1
    if info_db:
        for key, episode_info in info_db.items():
            if isinstance(episode_info, dict) and "seed" in episode_info:
                seed = episode_info["seed"]
                if isinstance(seed, (int, float)):
                    last_seed = max(last_seed, int(seed))
    
    # Start from last seed + 1, or 0 if no previous episodes
    epid = last_seed + 1 if last_seed >= 0 else 0
    
    if start_episode_idx > 0:
        print(f"Resuming from episode {start_episode_idx} (last episode: {last_episode_idx})")
    if last_seed >= 0:
        print(f"Starting from seed {epid} (last used seed: {last_seed})")
    else:
        print(f"Starting from seed {epid}")

    # Set up for data collection
    args["need_plan"] = True  # We need planning for each episode
    args["save_data"] = True
    clear_cache_freq = args["clear_cache_freq"]

    # Main collection loop: collect episode_num trajectories starting from start_episode_idx
    target_episode_idx = start_episode_idx + args["episode_num"]
    while episode_idx < target_episode_idx:
        print(f"\n\033[34mTask name: {args['task_name']}\033[0m")
        print(f"Episode {episode_idx} (target: {target_episode_idx - 1}, remaining: {target_episode_idx - episode_idx}) (seed={epid})")

        # Step 1: Try to setup environment (initialization)
        init_success = False
        try:
            # Setup environment for this episode
            TASK_ENV.setup_demo(now_ep_num=episode_idx, seed=epid, **args)
            init_success = True
            print(f"\033[93m[Episode {episode_idx}] Initialization successful (seed={epid})\033[0m")
        except (UnStableError, Exception) as e:
            print(" -------------")
            print(f"\033[91m[Episode {episode_idx}] Initialization FAILED (seed={epid}) - Skipping episode\033[0m")
            print("Error: ", e)
            if isinstance(e, UnStableError):
                print("(UnStableError: Objects are unstable during initialization)")
            else:
                traceback.print_exc()
            print(" -------------")
            
            # Clean up environment if partially initialized
            try:
                TASK_ENV.close_env()
            except:
                pass
            
            # Skip this episode: don't increment episode_idx, don't save data, don't count success/fail
            # But increment epid since we used this seed
            epid += 1
            time.sleep(0.3)
            continue  # Skip to next iteration without processing this episode
        
        # Step 2: Only proceed if initialization was successful
        if init_success:
            try:
                # Play the episode
                info = TASK_ENV.play_once()
                
                # Check if viewer was closed manually
                viewer_closed = False
                if getattr(TASK_ENV, "viewer", None) is not None:
                    try:
                        viewer_closed = TASK_ENV.viewer.closed
                    except:
                        viewer_closed = False
                
                # If viewer was closed and save_on_viewer_close is False, skip saving
                if viewer_closed and not args.get("save_on_viewer_close", False):
                    print(f"\033[93m[Episode {episode_idx}] Viewer closed manually - Skipping data save (seed={epid})\033[0m")
                    # Clean up environment
                    TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                    TASK_ENV.remove_data_cache()
                    # Only increment epid to try with a new seed, but don't increment episode_idx
                    # so this episode doesn't count towards the total
                    epid += 1
                    continue
                
                skip_save_due_to_no_grasp_cp = (
                    getattr(TASK_ENV, "last_failure_code", None) == "no_valid_grasp_cp_after_filter"
                )
                task_info_success = True
                if isinstance(info, dict):
                    if info.get("success") is False:
                        task_info_success = False
                    elif isinstance(info.get("info"), dict) and info["info"].get("success") is False:
                        task_info_success = False

                success = bool(
                    TASK_ENV.plan_success
                    and TASK_ENV.check_success()
                    and task_info_success
                    and not skip_save_due_to_no_grasp_cp
                )

                if success:
                    print(f"\033[92m[Episode {episode_idx}] SUCCESS (seed={epid})\033[0m")
                else:
                    print(f"\033[91m[Episode {episode_idx}] FAILED (seed={epid})\033[0m")
                skip_save_due_to_initial_plan_fail = (
                    getattr(TASK_ENV, "last_failure_code", None) == "initial_motion_plan_failed"
                )
                split_save_dir = success_save_dir if success else failure_save_dir
                if skip_save_due_to_no_grasp_cp:
                    print(
                        f"\033[93m[Episode {episode_idx}] Skip saving traj/video: "
                        "no valid grasp contact point after filtering. This attempt will not count.\033[0m"
                    )
                    TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                    TASK_ENV.remove_data_cache()
                    epid += 1
                    continue
                elif skip_save_due_to_initial_plan_fail:
                    print(
                        f"\033[93m[Episode {episode_idx}] Skip saving traj/video: "
                        "initial motion planning failed. This attempt will not count.\033[0m"
                    )
                    TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                    TASK_ENV.remove_data_cache()
                    epid += 1
                    continue

                # Only saved episodes count toward episode_num/statistics.
                frames_saved = int(getattr(TASK_ENV, "FRAME_IDX", 0) or 0)
                if frames_saved <= 0:
                    print(
                        f"\033[93m[Episode {episode_idx}] No frames captured "
                        f"(FRAME_IDX=0, plan_success={getattr(TASK_ENV, 'plan_success', None)}). "
                        "Skip saving and do not count this attempt.\033[0m"
                    )
                    TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                    TASK_ENV.remove_data_cache()
                    epid += 1
                    continue

                if success:
                    extra_frames = TASK_ENV.save_post_success_extra_frames()
                    if extra_frames > 0:
                        print(
                            f"\033[92m[Episode {episode_idx}] Saved {extra_frames} post-success frames "
                            f"(total FRAME_IDX={getattr(TASK_ENV, 'FRAME_IDX', 0)})\033[0m"
                        )

                TASK_ENV.save_traj_data(episode_idx, target_save_dir=split_save_dir)
                TASK_ENV.merge_pkl_to_hdf5_video(target_save_dir=split_save_dir)

                # Save scene info with success status
                episode_info = info.copy() if info else {}
                episode_info["success"] = success
                episode_info["seed"] = epid
                episode_info["embodiment"] = args.get("embodiment_name")
                info_db[f"episode_{episode_idx}"] = episode_info
                
                with open(info_file_path, "w", encoding="utf-8") as file:
                    json.dump(info_db, file, ensure_ascii=False, indent=4)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                
                # Close environment and clean cache
                TASK_ENV.close_env(clear_cache=((episode_idx + 1) % clear_cache_freq == 0))
                TASK_ENV.remove_data_cache()

                episode_idx += 1
                epid += 1

            except Exception as e:
                print(" -------------")
                print(f"\033[91m[Episode {episode_idx}] FAILED during execution (seed={epid}) - Exception\033[0m")
                print("Error: ", e)
                traceback.print_exc()
                print(" -------------")
                
                # Check if viewer was closed manually
                viewer_closed = False
                if getattr(TASK_ENV, "viewer", None) is not None:
                    try:
                        viewer_closed = TASK_ENV.viewer.closed
                    except:
                        viewer_closed = False
                
                # If viewer was closed and save_on_viewer_close is False, skip saving
                if viewer_closed and not args.get("save_on_viewer_close", False):
                    print(f"\033[93m[Episode {episode_idx}] Viewer closed manually during execution - Skipping data save (seed={epid})\033[0m")
                    # Clean up environment
                    try:
                        TASK_ENV.close_env()
                    except:
                        pass
                    # Only increment epid to try with a new seed, but don't increment episode_idx
                    # so this episode doesn't count towards the total
                    epid += 1
                    time.sleep(0.3)
                    continue
                
                skip_save_due_to_no_grasp_cp = (
                    getattr(TASK_ENV, "last_failure_code", None) == "no_valid_grasp_cp_after_filter"
                )
                skip_save_due_to_initial_plan_fail = (
                    getattr(TASK_ENV, "last_failure_code", None) == "initial_motion_plan_failed"
                )
                
                # Still save trajectory and data even on error (since initialization was successful)
                try:
                    if skip_save_due_to_no_grasp_cp:
                        print(
                            f"\033[93m[Episode {episode_idx}] Exception path skip saving traj/video: "
                            "no valid grasp contact point after filtering. This attempt will not count.\033[0m"
                        )
                        try:
                            TASK_ENV.remove_data_cache()
                        except:
                            pass
                        try:
                            TASK_ENV.close_env()
                        except:
                            pass
                        epid += 1
                        time.sleep(1)
                        continue
                    elif skip_save_due_to_initial_plan_fail:
                        print(
                            f"\033[93m[Episode {episode_idx}] Exception path skip saving traj/video: "
                            "initial motion planning failed. This attempt will not count.\033[0m"
                        )
                        try:
                            TASK_ENV.remove_data_cache()
                        except:
                            pass
                        try:
                            TASK_ENV.close_env()
                        except:
                            pass
                        epid += 1
                        time.sleep(1)
                        continue
                    elif hasattr(TASK_ENV, 'save_dir') and TASK_ENV.save_dir is not None:
                        frames_saved = int(getattr(TASK_ENV, "FRAME_IDX", 0) or 0)
                        if frames_saved > 0:
                            TASK_ENV.save_traj_data(episode_idx, target_save_dir=failure_save_dir)
                            TASK_ENV.merge_pkl_to_hdf5_video(target_save_dir=failure_save_dir)
                        else:
                            print(
                                f"\033[93m[Episode {episode_idx}] Exception path: no frames captured "
                                f"(FRAME_IDX=0, plan_success={getattr(TASK_ENV, 'plan_success', None)}). "
                                "Skip saving and do not count this attempt.\033[0m"
                            )
                            try:
                                TASK_ENV.remove_data_cache()
                            except:
                                pass
                            try:
                                TASK_ENV.close_env()
                            except:
                                pass
                            epid += 1
                            time.sleep(1)
                            continue
                        TASK_ENV.remove_data_cache()
                    else:
                        print(
                            f"\033[93m[Episode {episode_idx}] Exception path: save_dir is missing. "
                            "Skip saving and do not count this attempt.\033[0m"
                        )
                        try:
                            TASK_ENV.close_env()
                        except:
                            pass
                        epid += 1
                        time.sleep(1)
                        continue
                except Exception as save_error:
                    print(f"Warning: Failed to save data for episode {episode_idx}: {save_error}")
                    try:
                        TASK_ENV.remove_data_cache()
                    except:
                        pass
                    try:
                        TASK_ENV.close_env()
                    except:
                        pass
                    epid += 1
                    time.sleep(1)
                    continue
                
                # Save episode info
                try:
                    episode_info = {
                        "success": False,
                        "seed": epid,
                        "embodiment": args.get("embodiment_name"),
                        "error": "Exception",
                        "error_message": str(e),
                    }
                    info_db[f"episode_{episode_idx}"] = episode_info
                    with open(info_file_path, "w", encoding="utf-8") as file:
                        json.dump(info_db, file, ensure_ascii=False, indent=4)
                except:
                    pass
                fail_count += 1
                
                # Clean up environment
                try:
                    TASK_ENV.close_env()
                except:
                    pass
                
                time.sleep(1)
                
                episode_idx += 1
                epid += 1

    print(f"\n\033[93m" + "="*50 + "\033[0m")
    print(f"\033[93mComplete data collection finished!\033[0m")
    # Only count episodes that were actually run (successful initialization)
    total_run_episodes = success_count + fail_count
    print(f"Total episodes run: {total_run_episodes}")
    print(f"Success: \033[92m{success_count}\033[0m")
    print(f"Failed: \033[91m{fail_count}\033[0m")
    print(f"Success rate: {success_count/total_run_episodes*100:.1f}%" if total_run_episodes > 0 else "N/A")
    print(f"\033[93m" + "="*50 + "\033[0m")

    # Generate episode instructions if language_num is specified
    if args.get("language_num", 0) > 0:
        command = f"cd description && bash gen_episode_instructions.sh {args['task_name']} {args['task_config']} {args['language_num']}"
        os.system(command)


if __name__ == "__main__":
    Sapien_TEST = None
    try:
        from script.test_render import Sapien_TEST as _Sapien_TEST
        Sapien_TEST = _Sapien_TEST
    except Exception:
        try:
            from test_render import Sapien_TEST as _Sapien_TEST
            Sapien_TEST = _Sapien_TEST
        except Exception:
            Sapien_TEST = None
    if Sapien_TEST is not None:
        Sapien_TEST()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--render-freq", type=int, default=None, help="Enable Viewer UI rendering every N steps (0 disables). Overrides task_config and ROBOTWIN_RENDER_FREQ.")
    parser.add_argument("--save-on-viewer-close", action="store_true", default=False, help="Save data even when viewer is manually closed (default: False, skip saving)")
    parser.add_argument("--use-cousin-coordinate", action="store_true", default=False, help="Use specified coordinates from script/cousin_coordinate.py for object placement (default: random)")
    parser.add_argument("--room-type", type=str, default=None, help="Room preset type: livingroom/bedroom/diningroom/bathroom/kidsroom/study (supports 客厅/卧室/餐厅/卫生间/儿童房/书房).")
    parser.add_argument("--room-layout-json", type=str, default=None, help="Explicit room layout JSON path.")
    parser.add_argument("--room-layout-index", type=int, default=None, help="Index into sorted room preset JSON list (default: random).")
    parser.add_argument("--generate-room-layout", action="store_true", default=False, help="Generate room layout first via envs/utils/layout_generator.py.")
    parser.add_argument("--room-prompt", type=str, default=None, help="Prompt text used with --generate-room-layout.")
    parser.add_argument("--room-collision", action="store_true", default=False, help="Enable convex collision for room meshes.")
    parser.add_argument("--room-keep-random-background", action="store_true", default=False, help="Keep domain_randomization.random_background unchanged.")
    parser.add_argument("--room-align-offset-deg", type=float, default=None, help="Set ui_room_layout_align_offset_deg.")
    parser.add_argument("--no-room-layout-log", action="store_true", default=False, help="Disable verbose room layout logs.")
    parsed_args = parser.parse_args()
    task_name = parsed_args.task_name
    task_config = parsed_args.task_config
    render_freq_override = parsed_args.render_freq
    save_on_viewer_close = parsed_args.save_on_viewer_close
    use_cousin_coordinate = parsed_args.use_cousin_coordinate
    room_type = parsed_args.room_type
    room_layout_json = parsed_args.room_layout_json
    room_layout_index = parsed_args.room_layout_index
    generate_room_layout = parsed_args.generate_room_layout
    room_prompt = parsed_args.room_prompt
    room_collision = parsed_args.room_collision
    room_keep_random_background = parsed_args.room_keep_random_background
    room_align_offset_deg = parsed_args.room_align_offset_deg
    room_layout_verbose = not parsed_args.no_room_layout_log

    main(
        task_name=task_name,
        task_config=task_config,
        render_freq_override=render_freq_override,
        save_on_viewer_close=save_on_viewer_close,
        use_cousin_coordinate=use_cousin_coordinate,
        room_type=room_type,
        room_layout_json=room_layout_json,
        room_layout_index=room_layout_index,
        generate_room_layout=generate_room_layout,
        room_prompt=room_prompt,
        room_collision=room_collision,
        room_keep_random_background=room_keep_random_background,
        room_align_offset_deg=room_align_offset_deg,
        room_layout_verbose=room_layout_verbose,
    )
