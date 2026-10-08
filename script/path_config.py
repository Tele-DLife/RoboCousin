import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PATH_CONFIG = {
    "data_include_failure_dir": None,
    "room_scenes_export_dir": None,
    "three_d_generation_data_dir": None,
    "cousin_layout_desk_layout_dir": None,
    "hf_cache_root": None,
    "digital_cousins_our_objects_dir": None,
}


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _get_configured_path(config_key: str, env_key: str, default_relative: str) -> Path:
    raw_value = PATH_CONFIG.get(config_key)
    if raw_value is None:
        raw_value = os.getenv(env_key)
    if raw_value is None or str(raw_value).strip() == "":
        return (REPO_ROOT / default_relative).resolve()
    return _resolve_path(str(raw_value))


UI_GENERATED_LAYOUT_JSON = (REPO_ROOT / "envs" / "room_config" / "ui_generated_layout.json").resolve()
ROOM_CONFIG_50ROOMS_DIR = (REPO_ROOT / "envs" / "room_config" / "50rooms").resolve()
DATA_INCLUDE_FAILURE_DIR = _get_configured_path(
    "data_include_failure_dir",
    "ROBOTWIN_DATA_INCLUDE_FAILURE_DIR",
    "/data/RoboCousin/data_include_failure",
)
ROOM_SCENES_EXPORT_DIR = _get_configured_path(
    "room_scenes_export_dir",
    "ROBOTWIN_ROOM_SCENES_DIR",
    "room_scenes",
)
THREE_D_GENERATION_DATA_DIR = _get_configured_path(
    "three_d_generation_data_dir",
    "ROBOTWIN_THREED_DATA_DIR",
    "threeD-generation/data",
)
COUSIN_LAYOUT_DESK_LAYOUT_DIR = _get_configured_path(
    "cousin_layout_desk_layout_dir",
    "ROBOTWIN_COUSIN_DESK_LAYOUT_DIR",
    "cousin_layout/desk_layout",
)
HF_CACHE_ROOT = _get_configured_path(
    "hf_cache_root",
    "ROBOTWIN_HF_CACHE_DIR",
    "/data/huggingface_cache",
)
DIGITAL_COUSINS_OUR_OBJECTS_DIR = _get_configured_path(
    "digital_cousins_our_objects_dir",
    "ROBOTWIN_DIGITAL_COUSINS_OBJECTS_DIR",
    "our_assets/actor",
)
