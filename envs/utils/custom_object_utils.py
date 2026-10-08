"""
Utility functions for loading custom objects from ``our_assets/actor`` (manipulated)
and, for clutter, from ``our_assets/actor`` + ``our_assets/non-actor`` (see
``rand_create_cluttered_actor.get_all_cluttered_objects``).

Provides reusable functions for:
- Reading URDF properties (mass, mesh scale)
- Calculating z_offset for stable placement
- Creating custom object actors with proper physics
"""
from pathlib import Path
from copy import deepcopy
import json
import xml.etree.ElementTree as ET
import numpy as np
import transforms3d as t3d
from typing import Optional, Tuple, Dict, Any
import sapien
import os

from .create_actor import find_urdf_in_dir, create_urdf_obj_from_path
from .actor_utils import Actor, ArticulationActor
from .._GLOBAL_CONFIGS import DEFAULT_CUSTOM_DYNAMIC_MASS_KG, ROOT_PATH

# Default directory for task manipulable (actor) custom objects, relative to repo root.
DEFAULT_CUSTOM_ACTOR_OBJECTS_DIR = "our_assets/actor"
DEFAULT_CUSTOM_NONACTOR_OBJECTS_DIR = "our_assets/non-actor"

# Bottle-like actor classes allowed for shake_bottle / shake_bottle_horizontally tasks.
SHAKE_BOTTLE_CUSTOM_OBJECT_NAMES = [
    "baby bottle",
    "bottle drink",
    "canned drink",
    "glass drink",
    "tumbler",
]

# Custom our_assets mesh default: lying flat (local Y horizontal). Upright spawn uses Rx(90°).
CUSTOM_OBJECT_UPRIGHT_BASE_QPOS = [0.7071, 0.7071, 0.0, 0.0]
CUSTOM_OBJECT_HORIZONTAL_BASE_QPOS = [1.0, 0.0, 0.0, 0.0]
# Builtin assets/objects (e.g. 001_bottle): local Y is long axis; Rx(90°) upright, Rz(180°) horizontal.
BUILTIN_OBJECT_UPRIGHT_BASE_QPOS = [0.7071, 0.7071, 0.0, 0.0]
BUILTIN_OBJECT_HORIZONTAL_BASE_QPOS = [0.0, 0.0, 1.0, 0.0]


def mixed_horizontal_contact_point_ids(cp_ids: list) -> list:
    """
    Horizontal-only contact point indices for mixed-strategy model_data layouts.
    New format: 8 horizontal (0-7) + 4 vertical (8-11); older layouts use proportionally fewer horizontals.
    """
    n = len(cp_ids)
    if n >= 12:
        return cp_ids[:8]
    if n >= 10:
        return cp_ids[:6]
    if n >= 5:
        return cp_ids[:4]
    return list(cp_ids)


def is_custom_objects_clutter_root(root: Optional[str]) -> bool:
    """True if ``root`` is a ``our_assets/...`` clutter registry path."""
    return str(root or "").startswith("our_assets/")


def should_randomize_object_yaw(task_args: Optional[Dict[str, Any]]) -> bool:
    """
    Whether a task **target** actor should get random table yaw this episode.

    ``cousin_random_yaw`` applies outside cousin layout mode too (e.g. pick_up grasp target).
    When absent from ``task_args``, fall back to ``custom_object_upright_only``.
    """
    args = task_args or {}
    if "cousin_random_yaw" in args:
        return bool(args.get("cousin_random_yaw"))
    return not bool(args.get("custom_object_upright_only", True))


def sample_object_yaw_rad(task_args: Optional[Dict[str, Any]], rng=None) -> float:
    """Sample Ry yaw (radians) for custom-object placement; 0 when yaw is fixed."""
    import math
    import random

    if not should_randomize_object_yaw(task_args):
        return 0.0
    r = rng if rng is not None else random
    if hasattr(r, "uniform"):
        return float(r.uniform(-math.pi, math.pi))
    return float(np.random.uniform(-math.pi, math.pi))


def compose_custom_object_quat(base_qpos, yaw_rad: float = 0.0) -> np.ndarray:
    """``qmult(base, Ry(yaw))`` — upright our_assets convention (cousin clutter)."""
    return compose_object_table_yaw_quat(base_qpos, yaw_rad, axis="y")


def compose_object_table_yaw_quat(
    base_qpos,
    yaw_rad: float = 0.0,
    *,
    axis: str = "y",
) -> np.ndarray:
    """Apply in-plane table rotation on top of ``base_qpos`` (``y``=Ry, ``z``=Rz)."""
    base_q = np.array(base_qpos, dtype=np.float32)
    if abs(float(yaw_rad)) < 1e-12:
        return base_q
    yaw = float(yaw_rad)
    if str(axis).lower() == "z":
        yaw_q = np.array(t3d.euler.euler2quat(0, 0, yaw), dtype=np.float32)
    else:
        yaw_q = np.array(t3d.euler.euler2quat(0, yaw, 0), dtype=np.float32)
    return np.array(t3d.quaternions.qmult(base_q, yaw_q), dtype=np.float32)


def pick_up_spawn_table_yaw_axis(task_args: Optional[Dict[str, Any]]) -> str:
    """``z`` when lying flat (spin heading on table); ``y`` when upright."""
    return "z" if pick_up_spawn_horizontal_enabled(task_args) else "y"


def custom_object_spawn_quat(task_args: Optional[Dict[str, Any]], base_qpos, rng=None) -> np.ndarray:
    """Final spawn quaternion for a custom object (base pose + optional random Ry yaw)."""
    return compose_custom_object_quat(base_qpos, sample_object_yaw_rad(task_args, rng=rng))


def pick_up_spawn_horizontal_enabled(task_args: Optional[Dict[str, Any]]) -> bool:
    """True when pick_up should spawn grasp targets lying flat with random table yaw."""
    args = task_args or {}
    if "pick_up_spawn_horizontal" in args:
        return bool(args.get("pick_up_spawn_horizontal"))
    return bool(args.get("pick_up_custom_object_horizontal", False))


def pick_up_custom_object_horizontal_enabled(task_args: Optional[Dict[str, Any]]) -> bool:
    """Backward-compatible alias for :func:`pick_up_spawn_horizontal_enabled`."""
    return pick_up_spawn_horizontal_enabled(task_args)


def resolve_pick_up_spawn_base_qpos(task_args: Optional[Dict[str, Any]], use_custom: bool) -> list:
    """Base spawn quaternion for pick_up grasp targets (custom or builtin)."""
    args = task_args or {}
    if pick_up_spawn_horizontal_enabled(args):
        if use_custom:
            return list(CUSTOM_OBJECT_HORIZONTAL_BASE_QPOS)
        return list(BUILTIN_OBJECT_HORIZONTAL_BASE_QPOS)
    if use_custom:
        return list(args.get("custom_object_base_qpos", CUSTOM_OBJECT_UPRIGHT_BASE_QPOS))
    return list(BUILTIN_OBJECT_UPRIGHT_BASE_QPOS)


def resolve_custom_object_spawn_base_qpos(task_args: Optional[Dict[str, Any]]) -> list:
    """Base spawn quaternion for our_assets custom clutter / cousin placement."""
    return resolve_pick_up_spawn_base_qpos(task_args, use_custom=True)


def pick_up_spawn_quat(
    task_args: Optional[Dict[str, Any]],
    use_custom: bool,
    rng=None,
) -> np.ndarray:
    """Final spawn quaternion for pick_up grasp targets (custom or builtin)."""
    args = dict(task_args or {})
    base_qpos = resolve_pick_up_spawn_base_qpos(args, use_custom)
    if pick_up_spawn_horizontal_enabled(args):
        args["cousin_random_yaw"] = True
    yaw = sample_object_yaw_rad(args, rng=rng)
    axis = pick_up_spawn_table_yaw_axis(args)
    return compose_object_table_yaw_quat(base_qpos, yaw, axis=axis)


def pick_up_custom_object_spawn_quat(task_args: Optional[Dict[str, Any]], rng=None) -> np.ndarray:
    """Backward-compatible alias: custom-object pick_up spawn quaternion."""
    return pick_up_spawn_quat(task_args, use_custom=True, rng=rng)


def apply_pick_up_horizontal_spawn_physics(actor, task_args: Optional[Dict[str, Any]] = None) -> None:
    """
    Reduce rolling/sliding for horizontally spawned pick_up grasp targets.

    Uses per-actor linear/angular damping (same approach as shake_bottle). Scene-level
    ``static_friction`` / ``dynamic_friction`` in task config also affect contact.
    """
    if actor is None or not pick_up_spawn_horizontal_enabled(task_args):
        return
    args = task_args or {}
    lin = float(args.get("pick_up_spawn_horizontal_linear_damping", 2.0))
    ang = float(args.get("pick_up_spawn_horizontal_angular_damping", 10.0))
    try:
        actor.set_damping(linear_damping=lin, angular_damping=ang)
    except Exception:
        pass


def custom_object_placement_quat(
    task_args: Optional[Dict[str, Any]],
    yaw_rad: float,
    *,
    spawn_as_actor_only: bool = False,
) -> np.ndarray:
    """
    Compose base pose + Ry yaw for our_assets placement.

    When ``pick_up_spawn_horizontal`` (or legacy ``pick_up_custom_object_horizontal``)
    is enabled for a pick_up grasp target (``spawn_as_actor_only``), use horizontal
    base pose and resample yaw uniformly.
    """
    args = dict(task_args or {})
    base_qpos = resolve_custom_object_spawn_base_qpos(args)
    yaw = float(yaw_rad)
    if (
        spawn_as_actor_only
        and str(args.get("task_name") or "") == "pick_up"
        and pick_up_spawn_horizontal_enabled(args)
    ):
        args["cousin_random_yaw"] = True
        yaw = sample_object_yaw_rad(args)
        return compose_object_table_yaw_quat(base_qpos, yaw, axis="z")
    return compose_custom_object_quat(base_qpos, yaw)


def resolve_custom_clutter_instance_dir(root_path: str, instance_id: str) -> Path:
    """Instance directory: ``our_assets/<bucket>/<class>/<id>`` under repo root."""
    rp = str(root_path or "").strip().strip("/")
    base = Path(ROOT_PATH)
    if not rp.startswith("our_assets/"):
        raise ValueError(f"resolve_custom_clutter_instance_dir: expected our_assets/..., got {root_path!r}")
    d = base.joinpath(*rp.split("/"), str(instance_id))
    if d.is_dir():
        return d
    return base.joinpath(*rp.split("/"))


def read_urdf_properties(urdf_path: Path) -> Dict[str, Any]:
    """
    Read physical properties from URDF file.
    
    Args:
        urdf_path: Path to URDF file
        
    Returns:
        Dictionary with keys: 'mesh_scale', 'mass', 'rpy' (if found)
    """
    result = {
        'mesh_scale': (1.0, 1.0, 1.0),
        'mass': None,
        'rpy': (0.0, 0.0, 0.0),
    }
    
    if not urdf_path or not urdf_path.exists():
        return result
    
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Read mesh scale
        mesh = root.find(".//link/visual/geometry/mesh")
        if mesh is not None and mesh.get("scale"):
            result['mesh_scale'] = tuple(float(v) for v in mesh.get("scale").split())
        
        # Read mass
        mass_node = root.find(".//link/inertial/mass")
        if mass_node is not None and mass_node.get("value"):
            result['mass'] = float(mass_node.get("value"))
        
        # Read RPY (optional, for future use)
        origin = root.find(".//link/visual/origin")
        if origin is not None and origin.get("rpy"):
            result['rpy'] = tuple(float(v) for v in origin.get("rpy").split())
    except Exception:
        pass
    
    return result


def calculate_z_offset_from_mesh(
    collision_file: Path,
    scale: np.ndarray,
    base_quat: np.ndarray,
    default_offset: float = 0.005,
) -> float:
    """
    Calculate z_offset to place object bottom on table surface.
    Uses a small clearance (0.005m = 5mm) to ensure objects are stable without floating.
    
    Args:
        collision_file: Path to collision mesh file
        scale: Scale array (3D)
        base_quat: Base quaternion for rotation [qw, qx, qy, qz]
        default_offset: Default offset if calculation fails (small clearance to avoid penetration)
        
    Returns:
        z_offset value (object origin z = table_height + z_offset)
    """
    try:
        import trimesh
        mesh = trimesh.load(str(collision_file), force="mesh")
        bounds = mesh.bounds
        mins = bounds[0] * scale
        maxs = bounds[1] * scale
        
        # Get all 8 corners of the bounding box
        corners = np.array(
            [[x, y, z] for x in [mins[0], maxs[0]]
                       for y in [mins[1], maxs[1]]
                       for z in [mins[2], maxs[2]]],
            dtype=np.float32,
        )
        
        # Apply rotation for z_offset calculation
        rot_matrix = t3d.quaternions.quat2mat(base_quat)
        corners_rotated = (rot_matrix @ corners.T).T
        
        min_z = float(corners_rotated[:, 2].min())
        
        # Align the lowest point to the table surface with small clearance
        # Smaller clearance (0.005m) ensures objects are stable and don't float
        z_offset = -min_z + default_offset
        return z_offset
    except Exception as e:
        # Fallback to default if calculation fails
        return default_offset


def find_custom_object_mesh_files(obj_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find collision and visual mesh files in custom object directory.
    
    Args:
        obj_dir: Custom object directory path
        
    Returns:
        Tuple of (collision_file, visual_file) paths, or (None, None) if not found
    """
    mesh_dir = obj_dir / "mesh"
    collision_file = None
    visual_file = None
    
    if mesh_dir.exists():
        # Find collision file
        for name in ["sample_collision.obj", "sample_collision.glb"]:
            candidate = mesh_dir / name
            if candidate.exists():
                collision_file = candidate
                break
        
        # Find visual file
        for name in ["sample.obj", "sample.glb"]:
            candidate = mesh_dir / name
            if candidate.exists():
                visual_file = candidate
                break
    
    return collision_file, visual_file


def load_model_data(obj_dir: Path) -> Tuple[Optional[Dict], np.ndarray]:
    """
    Load model_data.json from custom object directory.
    
    Args:
        obj_dir: Custom object directory path
        
    Returns:
        Tuple of (model_data dict, scale array)
    """
    model_data = None
    scale = (1, 1, 1)
    
    try:
        json_file_path = obj_dir / "model_data.json"
        if json_file_path.exists():
            with open(json_file_path, "r") as file:
                model_data = json.load(file)
                scale = model_data.get("scale", (1, 1, 1))
    except Exception:
        model_data = None
        scale = (1, 1, 1)
    
    return model_data, scale


def create_custom_object_actor(
    scene,
    obj_dir: Path,
    pose: sapien.Pose,
    custom_scale: float = 1.0,
    custom_collision: str = "mesh",
    custom_mass: Optional[float] = None,
    default_mass: float = 0.05,
    table_height: float = 0.74,
    table_z_bias: float = 0.0,
    custom_spawn_height: float = 0.0,
    custom_base_qpos: list = [0.7071, 0.7071, 0.0, 0.0],
    calculate_z_offset: bool = True,
) -> Optional[Actor]:
    """
    Create a custom object Actor from a per-instance directory (e.g. our_assets/actor/cls/0).
    
    This function handles:
    - Loading mesh files (collision and visual)
    - Reading model_data.json
    - Reading URDF mesh scale; rigid-body mass uses ``DEFAULT_CUSTOM_DYNAMIC_MASS_KG``
    - Calculating proper z_offset for stable placement
    - Creating Actor with proper physics properties
    
    Args:
        scene: SAPIEN scene object
        obj_dir: Path to custom object instance dir (e.g., our_assets/actor/apple/0)
        pose: Initial pose (x, y will be used, z will be adjusted)
        custom_scale: Additional scale factor
        custom_collision: Collision type ("mesh" or "box")
        custom_mass: Unused; mass is ``DEFAULT_CUSTOM_DYNAMIC_MASS_KG``.
        default_mass: Unused; mass is ``DEFAULT_CUSTOM_DYNAMIC_MASS_KG``.
        table_height: Table height in meters
        table_z_bias: Table height bias
        custom_spawn_height: Additional spawn height offset
        custom_base_qpos: Base quaternion [qw, qx, qy, qz]
        calculate_z_offset: Whether to calculate z_offset from mesh bounds
        
    Returns:
        Actor object or None if creation fails
    """
    # Find mesh files
    collision_file, visual_file = find_custom_object_mesh_files(obj_dir)
    
    # Load model_data.json
    model_data, scale = load_model_data(obj_dir)
    
    # Read URDF properties
    urdf_path = find_urdf_in_dir(obj_dir)
    urdf_props = read_urdf_properties(urdf_path) if urdf_path else {}
    
    # Calculate final scale
    if isinstance(scale, (int, float, np.floating)):
        scale = (float(scale), float(scale), float(scale))
    scale = np.array(scale, dtype=np.float32)
    scale = scale * np.array(urdf_props['mesh_scale'], dtype=np.float32) * float(custom_scale)
    
    # IMPORTANT:
    # `pose.q` is already produced by caller-side rand_pose(qpos=custom_base_qpos, ...).
    # Do NOT multiply custom_base_qpos again here, otherwise orientation gets applied twice.
    # Only apply model_data.transform_matrix (if non-identity) on top of pose.q.
    try:
        pose_q = np.array(pose.q, dtype=np.float32).reshape(4)
        final_quat = pose_q
        _applied_align = False
        if isinstance(model_data, dict):
            T = np.array(model_data.get("transform_matrix", np.eye(4)), dtype=np.float32).reshape(4, 4)
            R = T[:3, :3]
            if np.max(np.abs(R - np.eye(3, dtype=np.float32))) > 1e-6:
                q_align = np.array(t3d.quaternions.mat2quat(R), dtype=np.float32).reshape(4)
                # local correction should be right-multiplied: final = pose_q * q_align
                final_quat = np.array(t3d.quaternions.qmult(pose_q, q_align), dtype=np.float32).reshape(4)
                _applied_align = True
    except Exception:
        final_quat = np.array(custom_base_qpos, dtype=np.float32).reshape(4)
        _applied_align = False

    # Minimal debug: verify whether create_custom_object_actor introduces roll/pitch.
    # Enabled only when explicitly requested (avoid noisy logs).
    if os.environ.get("ROBOTWIN_DEBUG_POSE_UP", "").strip() == "1":
        try:
            up_w = t3d.quaternions.rotate_vector([0.0, 1.0, 0.0], final_quat.tolist())
        except Exception:
            up_w = None
        print(f"{obj_dir.resolve()} final_q={final_quat.tolist()} up_w={up_w} applied_align={_applied_align}")

    # Calculate z_offset if needed (must use the same orientation we will actually set)
    z_offset = 0.02  # Default
    if calculate_z_offset and collision_file is not None:
        z_offset = calculate_z_offset_from_mesh(collision_file, scale, final_quat)
        # Adjust pose z coordinate only if we calculated z_offset
        final_table_height = table_height + table_z_bias
        adjusted_pose = sapien.Pose(
            [pose.p[0], pose.p[1], final_table_height + z_offset + custom_spawn_height],
            final_quat
        )
    else:
        # Use pose as-is if z_offset was already calculated externally.
        # However, if we changed orientation (e.g. applied model_data.transform_matrix),
        # the previously computed z_offset may no longer be correct. Compensate by the delta
        # between required offsets under the old vs new quaternion.
        z = float(pose.p[2])
        try:
            if collision_file is not None:
                old_quat = np.array(pose.q, dtype=np.float32).reshape(4)
                old_offset = calculate_z_offset_from_mesh(collision_file, scale, old_quat)
                new_offset = calculate_z_offset_from_mesh(collision_file, scale, final_quat)
                dz = float(new_offset - old_offset)
                if np.isfinite(dz) and abs(dz) > 1e-8:
                    z = z + dz
        except Exception:
            pass
        adjusted_pose = sapien.Pose([pose.p[0], pose.p[1], z], final_quat)
    
    # If no collision file, try URDF fallback
    if collision_file is None:
        if urdf_path:
            actor = create_urdf_obj_from_path(
                scene=scene,
                pose=adjusted_pose,
                urdf_path=urdf_path,
                scale=custom_scale,
                fix_root_link=False,
                name=obj_dir.name,
            )
            if actor is not None:
                _m = float(DEFAULT_CUSTOM_DYNAMIC_MASS_KG)
                if isinstance(actor, ArticulationActor):
                    links = list(actor.actor.get_links())
                    n = len(links) or 1
                    for L in links:
                        L.set_mass(_m / n)
                else:
                    actor.set_mass(_m)
            return actor
        return None
    
    # Back-fill model_data from mesh bounds if needed
    try:
        import trimesh
        mesh = trimesh.load(str(collision_file), force="mesh")
        bounds = mesh.bounds
        if model_data is None:
            model_data = {}
        if isinstance(model_data, dict):
            model_data = deepcopy(model_data)
            model_data.setdefault("extents", (bounds[1] - bounds[0]).tolist())
            model_data.setdefault("center", ((bounds[0] + bounds[1]) * 0.5).tolist())
            model_data["scale"] = scale.tolist()
    except Exception:
        if model_data is not None and isinstance(model_data, dict):
            model_data = deepcopy(model_data)
            model_data["scale"] = scale.tolist()
    
    # Create actor builder
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    collision_pose = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
    
    # Add collision
    use_box_collision = (str(custom_collision).lower() == "box")
    if use_box_collision and model_data and model_data.get("extents") is not None:
        extents = model_data.get("extents")
        center = model_data.get("center", [0.0, 0.0, 0.0])
        half_size = [
            float(extents[0]) * scale[0] * 0.5,
            float(extents[1]) * scale[1] * 0.5,
            float(extents[2]) * scale[2] * 0.5,
        ]
        box_pose = sapien.Pose(
            [float(center[0]) * scale[0], float(center[1]) * scale[1], float(center[2]) * scale[2]],
            collision_pose.q,
        )
        builder.add_box_collision(pose=box_pose, half_size=half_size)
    else:
        builder.add_multiple_convex_collisions_from_file(
            filename=str(collision_file),
            scale=scale,
            pose=collision_pose,
        )
    
    # Add visual
    if visual_file is None:
        visual_file = collision_file
    builder.add_visual_from_file(filename=str(visual_file), scale=scale, pose=collision_pose)
    
    # Build actor
    mesh_actor = builder.build(name=obj_dir.name)
    mesh_actor.set_pose(adjusted_pose)
    actor = Actor(mesh_actor, model_data)
    actor.set_mass(float(DEFAULT_CUSTOM_DYNAMIC_MASS_KG))
    return actor


def get_custom_object_strategy(obj_dir: Path) -> Optional[str]:
    """Return normalized ``strategy`` from model_data.json (e.g. ``mixed``, ``obb``), or None."""
    model_data, _ = load_model_data(Path(obj_dir))
    if not isinstance(model_data, dict):
        return None
    strategy = model_data.get("strategy")
    if strategy is None:
        return None
    return str(strategy).lower()


def is_mixed_strategy_custom_object(obj_dir: Path) -> bool:
    """True when model_data strategy is ``mixed`` (horizontal + vertical CP layout)."""
    return get_custom_object_strategy(obj_dir) == "mixed"


def estimate_scaled_object_height(
    obj_dir: Path,
    custom_scale: float = 1.0,
    base_quat: Optional[np.ndarray | list] = None,
) -> Optional[float]:
    """
    Estimate object height (meters) along world Z after model/URDF/custom scaling and base rotation.
    Uses collision mesh AABB corners rotated by ``base_quat`` (same convention as z_offset placement).
    """
    collision_file, _ = find_custom_object_mesh_files(Path(obj_dir))
    if collision_file is None:
        return None

    _, scale = load_model_data(Path(obj_dir))
    urdf_path = find_urdf_in_dir(Path(obj_dir))
    urdf_props = read_urdf_properties(urdf_path) if urdf_path else {}

    if isinstance(scale, (int, float, np.floating)):
        scale = (float(scale), float(scale), float(scale))
    scale = np.array(scale, dtype=np.float32)
    scale = scale * np.array(urdf_props["mesh_scale"], dtype=np.float32) * float(custom_scale)

    if base_quat is None:
        quat = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)
    else:
        quat = np.array(base_quat, dtype=np.float32).reshape(4)

    try:
        import trimesh

        mesh = trimesh.load(str(collision_file), force="mesh")
        bounds = mesh.bounds
        mins = bounds[0] * scale
        maxs = bounds[1] * scale
        corners = np.array(
            [[x, y, z] for x in [mins[0], maxs[0]]
                       for y in [mins[1], maxs[1]]
                       for z in [mins[2], maxs[2]]],
            dtype=np.float32,
        )
        rot_matrix = t3d.quaternions.quat2mat(quat)
        corners_rotated = (rot_matrix @ corners.T).T
        height = float(corners_rotated[:, 2].max() - corners_rotated[:, 2].min())
        if np.isfinite(height) and height > 0:
            return height
    except Exception:
        pass
    return None


def get_custom_object_directories(
    custom_obj_dir: str,
    custom_obj_names: Optional[list] = None,
) -> list[Path]:
    """
    Get list of valid custom object instance directories.

    Supports instance layouts under any root:
    - <root>/<object>/<instance_id>/... (numeric instance dirs), or
    - legacy single-level instance dir under <object>.
    """
    custom_root = Path(custom_obj_dir)
    if not custom_root.is_absolute():
        custom_root = Path(ROOT_PATH) / custom_root
    
    if not custom_root.exists():
        return []
    
    if isinstance(custom_obj_names, str):
        custom_obj_names = [custom_obj_names]

    def _is_instance_dir(p: Path) -> bool:
        if not p.is_dir():
            return False
        if (p / "mesh").exists():
            return True
        return any(p.glob("*.urdf"))

    def _collect_instances(object_dir: Path) -> list[Path]:
        if not object_dir.exists() or not object_dir.is_dir():
            return []
        # Prefer explicit instance subdirs (0/1/2/...)
        instance_subdirs = [p for p in object_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if instance_subdirs:
            return sorted(instance_subdirs, key=lambda p: int(p.name))
        # Backward compatibility: object dir itself is an instance dir.
        if _is_instance_dir(object_dir):
            return [object_dir]
        return []

    obj_dirs: list[Path] = []
    if custom_obj_names:
        for name in custom_obj_names:
            obj_dirs.extend(_collect_instances(custom_root / name))
    else:
        for object_dir in custom_root.iterdir():
            if object_dir.is_dir():
                obj_dirs.extend(_collect_instances(object_dir))

    # Deduplicate while preserving order.
    deduped = []
    seen = set()
    for p in obj_dirs:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    return deduped


def get_custom_object_label(obj_dir: Path, custom_root: Optional[Path] = None) -> str:
    """
    Build a stable repo-relative label for a selected instance dir
    (e.g. ``our_assets/actor/orange/0``).
    """
    obj_dir = Path(obj_dir).resolve()
    repo = Path(ROOT_PATH).resolve()
    try:
        return obj_dir.relative_to(repo).as_posix()
    except ValueError:
        pass
    if custom_root is not None:
        custom_root = Path(custom_root)
        if not custom_root.is_absolute():
            custom_root = Path(ROOT_PATH) / custom_root
        custom_root = custom_root.resolve()
        try:
            return obj_dir.relative_to(custom_root).as_posix()
        except ValueError:
            pass
    return obj_dir.name
