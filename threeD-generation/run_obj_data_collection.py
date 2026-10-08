#!/usr/bin/env python3
"""
Script to:
1. Generate model_data for threeD-generation/obj using mix strategy
2. Collect data for place_bread_basket task with threeD-generation/obj as the object
3. Save data to threeD-generation/data
"""

import sys
import os
import shutil
import subprocess
import argparse
import yaml
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser(description="Run RoboTwin object data collection demo.")
    parser.add_argument("--room-type", type=str, default=None, help="Room preset type, e.g. livingroom/bedroom/diningroom/bathroom/kidsroom/study.")
    parser.add_argument("--room-layout-json", type=str, default=None, help="Explicit room layout JSON path.")
    parser.add_argument("--room-layout-index", type=int, default=None, help="Preset JSON index under room type.")
    return parser.parse_args()

# Add RoboTwin repo root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
os.chdir(project_root)
cli_args = _parse_args()
from script.path_config import THREE_D_GENERATION_DATA_DIR

# Step 1: Generate model_data for threeD-generation/obj using mix strategy
print("=" * 60)
print("Step 1: Generating model_data for threeD-generation/obj using mix strategy")
print("=" * 60)

result = subprocess.run([
    sys.executable,
    "script/auto_generate_contact_points.py",
    "--object_dir", "threeD-generation/obj",
    "--strategy", "auto",
    "--height_axis", "Y",
    "--height_ratio", "0.6",
    "--vertical_ratio", "0.6",
    "--no_skip_existing_model_data",
], check=True, cwd=project_root)

# Verify model_data.json was created
model_data_path = project_root / "threeD-generation" / "obj" / "model_data.json"
if not model_data_path.exists():
    raise FileNotFoundError(f"Error: model_data.json was not created at {model_data_path}")
print(f"✓ model_data.json created successfully at {model_data_path}")

# Step 2: Define task config for data collection
print("\n" + "=" * 60)
print("Step 2: Preparing task config for data collection")
print("=" * 60)

# Define configuration directly (no need to create file)
config_content = {
    "render_freq": 3,
    "dual_arm": False,
    "episode_num": 50,
    "use_seed": False,
    "save_freq": 15,
    "embodiment": ["aloha-agilex"],
    "language_num": 0,
    "domain_randomization": {
        "random_background": False,
        "cluttered_table": True,
        "clean_background_rate": 0,
        "random_head_camera_dis": 0,
        "random_table_height": 0.03,
        "random_light": False,
        "crazy_random_light_rate": 0
    },
    "camera": {
        "head_camera_type": "D435",
        "wrist_camera_type": "D435",
        "collect_head_camera": True,
        "collect_wrist_camera": True
    },
    "data_type": {
        "rgb": True,
        "third_view": False,
        "depth": False,
        "pointcloud": False,
        "observer": False,
        "endpose": True,
        "qpos": True,
        "mesh_segmentation": False,
        "actor_segmentation": False
    },
    "pcd_down_sample_num": 1024,
    "pcd_crop": True,
    "save_path": str(THREE_D_GENERATION_DATA_DIR),
    "clear_cache_freq": 5,
    "collect_data": True,
    "eval_video_log": True,
    "use_custom_objects": True,
    "use_custom_env_objects": False,
    "custom_env_objects_dir": "our_assets/actor",
    "custom_objects_dir": "threeD-generation",
    "custom_objects": ["obj"],
    "custom_object_scale": 1,
    "custom_object_fix_root": False,
    #"custom_object_mass": 0.05,
    "custom_object_collision": "mesh",
    "skip_stable_check": False,
    "custom_object_upright_only": True,
    "custom_object_spawn_height": 0.0,
    #"custom_object_base_qpos": [1, 0, 0, 0],
    "custom_object_base_qpos": [0.7071, 0.7071, 0.0, 0.0],
    "debug_grasp_cp": False,
    "grasp_preference_by_object": {
        "obj": "horizontal"
    },
    "require_grasp_pose_reachable": True,
    "debug_held_object_collision": False,
    "basket_pre_dis": 0.12,
    "basket_place_dis": 0.02,
    "basket_above_target": 0.10,
    "basket_rim_clearance_factor": 0.6,
    "basket_rim_clearance_margin": 0.04,
    "basket_rim_clearance_max": None,
    "basket_drop_below_rim": 0.0,
    # Viewer: slightly larger than 640x360; bottom-left via xdotool; hide docked tool panels.
    "viewer_resolutions": [960, 540],
    "viewer_window_placement": "bottom_left",
    "viewer_minimal_ui": True,
}

if config_content.get("viewer_window_placement") and config_content.get("render_freq"):
    if not shutil.which("xdotool"):
        print(
            "Notice: viewer_window_placement is set, but xdotool was not found; "
            "the window cannot be moved to the bottom-left corner. "
            "Install it with: sudo apt install xdotool",
            file=sys.stderr,
        )

from envs.utils.ui_room_preset_env import apply_ui_room_from_process_env

if cli_args.room_layout_json:
    os.environ["ROBOTWIN_UI_ROOM_JSON"] = str(cli_args.room_layout_json)
    os.environ.pop("ROBOTWIN_UI_ROOM_TYPE", None)
    os.environ.pop("ROBOTWIN_UI_ROOM_INDEX", None)
elif cli_args.room_type:
    os.environ["ROBOTWIN_UI_ROOM_TYPE"] = str(cli_args.room_type)
    os.environ.pop("ROBOTWIN_UI_ROOM_JSON", None)
    if cli_args.room_layout_index is not None:
        os.environ["ROBOTWIN_UI_ROOM_INDEX"] = str(cli_args.room_layout_index)
    else:
        os.environ.pop("ROBOTWIN_UI_ROOM_INDEX", None)

apply_ui_room_from_process_env(config_content, project_root, log_prefix="[run_obj_data_collection]")

print(f"✓ Task config prepared")

# Step 3: Collect data using collect_data_complete.py with custom save_path
print("\n" + "=" * 60)
print("Step 3: Collecting data for place_bread_basket task")
print("=" * 60)

# Import collect_data_complete module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "collect_data_complete",
    project_root / "script" / "collect_data_complete.py"
)
collect_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect_module)

# Monkey patch the main function to override save_path
original_main = collect_module.main

def patched_main(task_name=None, task_config=None, render_freq_override=None, config_dict=None):
    """Patched main that uses config_dict directly instead of reading from file"""
    task = collect_module.class_decorator(task_name)
    
    # Use config_dict directly if provided, otherwise fall back to file reading
    if config_dict is not None:
        args = config_dict.copy()
    else:
        # Fallback: read from file if config_dict not provided
        config_path = project_root / "task_config" / f"{task_config}.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['task_name'] = task_name
    # Set task_config in args (use provided value or default from config_dict)
    if task_config is None:
        task_config = args.get('task_config', 'obj_bread_basket_collection')
    args['task_config'] = task_config
    
    # Handle render_freq_override
    if render_freq_override is None:
        env_override = os.getenv("ROBOTWIN_RENDER_FREQ")
        if env_override is not None and str(env_override).strip() != "":
            try:
                render_freq_override = int(env_override)
            except ValueError:
                raise SystemExit(f"Invalid ROBOTWIN_RENDER_FREQ={env_override!r}, expected an int.")
    if render_freq_override is not None:
        args["render_freq"] = int(render_freq_override)
    
    # Get embodiment config
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH
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
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "number of embodiment config parameters should be 1 or 3"
    
    args["left_embodiment_config"] = collect_module.get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = collect_module.get_embodiment_config(args["right_robot_file"])
    
    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    
    # Show config
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
    print("\n==================================")
    
    args["embodiment_name"] = embodiment_name
    
    # Override save_path from config file (not hardcoded)
    # Use save_path from config if specified, otherwise use custom path
    if 'save_path' in args and args['save_path']:
        args["save_path"] = args["save_path"]
    else:
        args["save_path"] = os.path.join(
            str(THREE_D_GENERATION_DATA_DIR),
            str(args["task_name"]),
            args["task_config"],
        )
    
    print(f"\033[93mSave path:\033[0m {args['save_path']}")
    
    # Run the task
    collect_module.run(task, args)

# Replace main function
collect_module.main = patched_main

# Import render preflight and set multiprocessing
from script.test_render import Sapien_TEST
Sapien_TEST()

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

# Call the patched main function
print(f"\nStarting data collection...")
print(f"Task: place_bread_basket")
print(f"Object: threeD-generation/obj")
print(f"Save path: {THREE_D_GENERATION_DATA_DIR}\n")

collect_module.main(task_name="pick_up",#place_bread_basket", 
                    task_config=None,  # Not needed when using config_dict
                    render_freq_override=None,
                    config_dict=config_content)

print("\n" + "=" * 60)
print("Data collection completed!")
print(f"Data saved to: {THREE_D_GENERATION_DATA_DIR}")
print("=" * 60)
