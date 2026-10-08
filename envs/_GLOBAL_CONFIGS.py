# global configs
import os

ROOT_PATH = os.path.abspath(__file__)
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/")]
ROOT_PATH = ROOT_PATH[:ROOT_PATH.rfind("/") + 1]

ASSETS_PATH = os.path.join(ROOT_PATH, "assets/")
EMBODIMENTS_PATH = os.path.join(ASSETS_PATH, "embodiments/")
TEXTURES_PATH = os.path.join(ASSETS_PATH, "background_texture/")
CONFIGS_PATH = os.path.join(ROOT_PATH, "task_config/")
SCRIPT_PATH = os.path.join(ROOT_PATH, "script/")
DESCRIPTION_PATH = os.path.join(ROOT_PATH, "description/")

# Euler angles in world coordinates
# t3d.euler.quat2euler(quat) returns (theta_x, theta_y, theta_z)
# theta_y controls the pitch, and theta_z controls rotation around the axis perpendicular to the tabletop plane
GRASP_DIRECTION_DIC = {
    "left": [0, 0, 0, -1],
    "front_left": [-0.383, 0, 0, -0.924],
    "front": [-0.707, 0, 0, -0.707],
    "front_right": [-0.924, 0, 0, -0.383],
    "right": [-1, 0, 0, 0],
    "top_down": [-0.5, 0.5, -0.5, -0.5],
    "down_right": [-0.707, 0, -0.707, 0],
    "down_left": [0, 0.707, 0, -0.707],
    "top_down_little_left": [-0.353523, 0.61239, -0.353524, -0.61239],
    "top_down_little_right": [-0.61239, 0.353523, -0.61239, -0.353524],
    "left_arm_perf": [-0.853532, 0.146484, -0.353542, -0.3536],
    "right_arm_perf": [-0.353518, 0.353564, -0.14642, -0.853568],
}

WORLD_DIRECTION_DIC = {
    "left": [0, -0.707, 0, 0.707],  # -z  -y  -x
    "front": [0.5, -0.5, 0.5, 0.5],  # y   z   -x
    "right": [0.707, 0, 0.707, 0],  # z   y   -x
    "top_down": [0, 0.707, -0.707, 0],  # -x  -y  -z
}

ROTATE_NUM = 10

# Single place to tune mass for ``create_custom_object_actor`` and custom our_assets clutter spawns.
DEFAULT_CUSTOM_DYNAMIC_MASS_KG = 0.75


def is_mobile_arx_embodiment_urdf(urdf_path) -> bool:
    """True when URDF path contains ``assets/embodiments/mobile_ARX`` (case-insensitive)."""
    low = str(urdf_path or "").lower()
    return "mobile_arx" in low


# Embodiments allowed to use mixed-strategy height thresholds (force horizontal/vertical CP subsets).
DEFAULT_MIXED_GRASP_HEIGHT_FORCE_EMBODIMENTS = ("aloha-agilex",)


def embodiment_matches_any(name_or_path, allowed_patterns) -> bool:
    """True when any allowed pattern is a substring of ``name_or_path`` (case-insensitive)."""
    low = str(name_or_path or "").lower()
    if not low:
        return False
    for pat in allowed_patterns or ():
        if str(pat).lower() in low:
            return True
    return False
