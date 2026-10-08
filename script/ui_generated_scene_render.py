import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import trimesh
import sapien.core as sapien
from sapien.utils.viewer import Viewer
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat
try:
    from path_config import ROOM_CONFIG_50ROOMS_DIR, ROOM_SCENES_EXPORT_DIR, UI_GENERATED_LAYOUT_JSON
except ModuleNotFoundError:
    from script.path_config import ROOM_CONFIG_50ROOMS_DIR, ROOM_SCENES_EXPORT_DIR, UI_GENERATED_LAYOUT_JSON


# Ensure RoboTwin repo root is importable when running as a script:
# - `python script/foo.py` sets sys.path[0] to ".../script", so "envs" is not found.
repo_root = Path(__file__).resolve().parent.parent
script_dir = repo_root / "script"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from envs.utils.create_actor import create_box, create_table

ALIGN_OFFSET = 90


def _resolve_repo_path(path_like: str | os.PathLike[str]) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _repo_relative_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root))
    except Exception:
        return str(path.resolve())


def export_scene_from_source_json(json_path: str | None = None, out_glb: str = "scene_export_from_sources.glb"):
    """Assemble a GLB by loading the original source GLB files listed in the
    room JSON, placing them with the same placement logic used for the SAPIEN
    scene. This preserves original materials/textures instead of sampling
    render meshes from SAPIEN.

    If json_path is None, pick the most-recent JSON under envs/room_config/50rooms.
    """
    import json
    tm_scene = trimesh.Scene()

    cfg_dir = ROOM_CONFIG_50ROOMS_DIR
    if json_path is None:
        files = list(cfg_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No room JSON found in {cfg_dir}")
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        json_path = str(files[0])

    with open(json_path, 'r', encoding='utf-8') as f:
        room_config = json.load(f)

    # Precompute metadata and final rotations (same logic as placement in _create_custom_scene)
    asset_metadata = {}
    for item in room_config:
        model_path = _resolve_repo_path(item["path"])
        center, raw_extents = get_model_metadata(None, str(model_path))
        temp_pose = get_aligned_pose(0, 0, 0, ALIGN_OFFSET + item.get("rotation_deg", 0), fix_axis=item.get("fix_axis"))
        world_ext = get_world_extents(raw_extents, temp_pose.q)
        asset_metadata[item["name"]] = {"world_extents": world_ext, "final_quat": temp_pose.q}

    placed_objects = {}
    for item in room_config:
        name = item["name"]
        x, y, z = calculate_semantic_pos(item, asset_metadata, placed_objects)
        pose = sapien.Pose(p=[x, y, z], q=asset_metadata[name]["final_quat"])
        placed_objects[name] = {'pos_2d': (x, y), 'pos_z': z}

        # Load source file and add its geometry into a trimesh scene with transform
        try:
            loaded = trimesh.load(str(_resolve_repo_path(item["path"])), force='scene')
        except Exception:
            try:
                loaded = trimesh.load(str(_resolve_repo_path(item["path"])), force='mesh')
            except Exception as e:
                print(f"[warn] failed to load {item['path']}: {e}")
                continue

        # compute transform matrix
        tm = pose.to_transformation_matrix()

        # If loaded is a Scene, iterate its geometries; if Mesh, add directly
        if isinstance(loaded, trimesh.Scene):
            for gname, geom in loaded.geometry.items():
                try:
                    gm = geom.copy()
                    gm.apply_transform(tm)
                    node = f"{name}_{gname}"
                    tm_scene.add_geometry(gm, node_name=node)
                except Exception as e:
                    print(f"[warn] add geometry failed for {item['path']}:{gname} => {e}")
        else:
            try:
                gm = loaded.copy()
                gm.apply_transform(tm)
                tm_scene.add_geometry(gm, node_name=name)
            except Exception as e:
                print(f"[warn] add mesh failed for {item['path']} => {e}")

    if len(tm_scene.geometry) == 0:
        raise RuntimeError("No geometry collected from source files.")

    tm_scene.export(out_glb)
    print(f"--- Successfully exported GLB from source files: {out_glb} ---")
    return out_glb

def _setup_scene(render: bool) -> tuple[sapien.Engine, sapien.Scene, Viewer | None]:
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 250)
    scene.add_ground(0)
    scene.default_physical_material = scene.create_physical_material(0.5, 0.5, 0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 0, 1.8], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1.8], [1, 1, 1], shadow=True)

    viewer = None
    if render:
        # Reuse RoboTwin's proven UI placement machinery (same as preview_cousin_layout_ui):
        # - hide RenderWindow immediately after __init__
        # - xdotool moves while hidden
        # - show and re-apply move (WMs may ignore pre-map moves)
        from envs import _base_task as _bt

        viewer_res = (1280, 720)
        env_res = os.environ.get("ROBOTWIN_VIEWER_RES") or os.environ.get("ROBOTWIN_VIEWER_RESOLUTIONS")
        if env_res and str(env_res).strip():
            try:
                s = str(env_res).strip().lower().replace("x", ",")
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) >= 2:
                    viewer_res = (int(parts[0]), int(parts[1]))
            except Exception:
                pass

        # Match preview_cousin_layout_ui.py parsing semantics.
        def _parse_placement(s: str | None):
            if not s:
                return None
            low = str(s).strip().lower()
            if low == "bottom_left":
                return "bottom_left"
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            if len(parts) >= 2:
                try:
                    return [int(parts[0]), int(parts[1])]
                except Exception:
                    return None
            return None

        placement = _parse_placement(os.environ.get("ROBOTWIN_VIEWER_PLACEMENT"))
        minimal_ui_env = os.environ.get("ROBOTWIN_VIEWER_MINIMAL_UI")
        minimal_ui = True if minimal_ui_env is None else (str(minimal_ui_env).strip() != "0")

        viewer_kw = {}
        if minimal_ui:
            viewer_kw["plugins"] = _bt._viewer_plugins_minimal()

        _hid_viewer_for_placement = False
        if placement and _bt.shutil.which("xdotool"):
            _bt._install_render_window_hide_patch()
            _bt._robotwin_hide_next_render_window = True
            _hid_viewer_for_placement = True
            _bt._robotwin_viewer_target_placement = placement
            _bt._robotwin_viewer_target_resolutions = viewer_res

        viewer = Viewer(renderer, resolutions=viewer_res, **viewer_kw)
        _bt._robotwin_hide_next_render_window = False
        if _hid_viewer_for_placement:
            _bt._robotwin_viewer_target_placement = None
            _bt._robotwin_viewer_target_resolutions = None

        viewer.set_scene(scene)
        viewer.set_camera_xyz(x=0.4, y=0.22, z=1.5)
        viewer.set_camera_rpy(r=0, p=-0.8, y=2.45)

        if placement:
            dummy = _bt.Base_Task()
            placement_x11_wid = None
            try:
                placement_x11_wid = dummy._apply_viewer_window_placement(placement, viewer_res)
            finally:
                if _hid_viewer_for_placement:
                    use_opacity_bridge = False
                    if placement_x11_wid and _bt.shutil.which("xprop"):
                        use_opacity_bridge = _bt._x11_set_net_wm_window_opacity(
                            placement_x11_wid, fully_transparent=True
                        )
                    try:
                        viewer.window.show()
                    except Exception:
                        pass
                    dummy._apply_viewer_window_placement(placement, viewer_res, post_show_only=True)
                    if use_opacity_bridge:
                        _bt._x11_restore_net_wm_window_opacity(placement_x11_wid)
    return engine, scene, viewer

def get_aligned_pose(x, y, z, angle_deg, fix_axis=None):
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    if fix_axis == 'x_to_z':
        half_angle = -np.pi / 4.0
        qw = np.cos(half_angle)
        qy = np.sin(half_angle)
    elif fix_axis == 'y_to_z':
        half_angle = np.pi / 4.0
        qw = np.cos(half_angle)
        qx = np.sin(half_angle)
    elif fix_axis == 'neg_x_to_z':
        half_angle = np.pi / 4.0
        qw = np.cos(half_angle)
        qy = np.sin(half_angle)
    elif fix_axis == 'neg_y_to_z':
        half_angle = -np.pi / 4.0
        qw = np.cos(half_angle)
        qx = np.sin(half_angle)

    quat_fix = [qw, qx, qy, qz]
    angle_rad = np.deg2rad(angle_deg)
    quat_rotate_z = [np.cos(angle_rad / 2.0), 0.0, 0.0, np.sin(angle_rad / 2.0)]

    def quat_multiply(q2, q1):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w2*w1 - x2*x1 - y2*y1 - z2*z1,
            w2*x1 + x2*w1 + y2*z1 - z2*y1,
            w2*y1 - x2*z1 + y2*w1 + z2*x1,
            w2*z1 + x2*y1 - y2*x1 + z2*w1
        ]

    final_quat = quat_multiply(quat_rotate_z, quat_fix)
    return sapien.Pose(p=[x, y, z], q=final_quat)

def get_model_metadata(scene, model_path: str, scale: float = 1.0):
    mesh = trimesh.load(str(model_path), force='mesh')
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    extents = (bounds[1] - bounds[0]) * scale

    return center, extents

def get_world_extents(raw_extents, quat):
    """Compute world-space AABB dimensions for an object at a given rotation."""
    mat = quat2mat(quat)
    half = np.array(raw_extents) / 2.0
    # Construct the eight local-space corners.
    corners = np.array([
        [i, j, k] for i in [-half[0], half[0]] 
                  for j in [-half[1], half[1]] 
                  for k in [-half[2], half[2]]
    ])
    # Transform to world space.
    world_corners = corners @ mat.T
    world_min = np.min(world_corners, axis=0)
    world_max = np.max(world_corners, axis=0)
    return world_max - world_min

def calculate_semantic_pos(item, asset_metadata, placed_objects):
    # asset_metadata[name]['extents'] already stores rotated world dimensions.
    self_ext = asset_metadata[item['name']]['world_extents']
    floor_h = 0.1 
    
    if item.get("anchor") is None:
        x, y = item["pos_abs"]
        z = self_ext[2] / 2.0 + floor_h
        return x, y, z
    
    anchor_name = item["anchor"]
    if anchor_name not in placed_objects:
        raise KeyError(f"Anchor object not found: {anchor_name}")

    ax, ay = placed_objects[anchor_name]['pos_2d']
    az = placed_objects[anchor_name]['pos_z']
    a_ext = asset_metadata[anchor_name]['world_extents']
    
    side = item.get("rel_side", "left")
    gap = item.get("gap", 0.0)

    anchor_bottom_z = az - (a_ext[2] / 2.0)
    anchor_top_z = az + (a_ext[2] / 2.0)

    if side == "top":
        x = ax + item.get("x_extra", 0.0)
        y = ay + item.get("y_extra", 0.0)
        z = anchor_top_z + (self_ext[2] / 2.0) + gap
    else:
        z = anchor_bottom_z + (self_ext[2] / 2.0)
        if ALIGN_OFFSET == -90:
            if side == "left":
                x = ax - (a_ext[0]/2 + self_ext[0]/2 + gap)
                y = ay
            elif side == "right":
                x = ax + (a_ext[0]/2 + self_ext[0]/2 + gap)
                y = ay
            elif side == "back":
                x = ax
                y = ay + (a_ext[1]/2 + self_ext[1]/2 + gap)
            elif side == "front":
                x = ax
                y = ay - (a_ext[1]/2 + self_ext[1]/2 + gap)
            else:
                x, y = ax, ay
        elif ALIGN_OFFSET == 90:
            if side == "left":
                x = ax + (a_ext[0]/2 + self_ext[0]/2 + gap)
                y = ay
            elif side == "right":
                x = ax - (a_ext[0]/2 + self_ext[0]/2 + gap)
                y = ay
            elif side == "back":
                x = ax
                y = ay - (a_ext[1]/2 + self_ext[1]/2 + gap)
            elif side == "front":
                x = ax
                y = ay + (a_ext[1]/2 + self_ext[1]/2 + gap)
            else:
                x, y = ax, ay
    
            
    return x, y, z


def load_custom_asset(scene, pose, model_path, scale=1.0, is_static=True, name="custom_object"):
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if is_static else "dynamic")
    builder.add_visual_from_file(filename=model_path, pose=sapien.Pose(), scale=[scale]*3)
    builder.add_convex_collision_from_file(filename=model_path, pose=sapien.Pose(), scale=[scale]*3, material=scene.default_physical_material)
    builder.set_initial_pose(pose)
    return builder.build(name=name)

def _create_custom_scene(scene: sapien.Scene, table_height: float, export: bool, export_path: Path) -> None:
    # --- Walls and floor (unchanged) ---
    hs = [3.5, 0.05, 2.0]
    d = hs[0] - hs[1]
    wall_poses = [
        sapien.Pose(p=[d, -1.5, 1.5], q=euler2quat(0, 0, np.pi/2)), 
        sapien.Pose(p=[0, d-1.5, 1.5], q=euler2quat(0, 0, 0)),
        sapien.Pose(p=[-d, -1.5, 1.5], q=euler2quat(0, 0, np.pi/2)),
        sapien.Pose(p=[0, -d-1.5, 1.5], q=euler2quat(0, 0, 0))
    ]
    for i, pose in enumerate(wall_poses):
        create_box(scene, pose, hs, color=(1, 1, 1), name=f"wall_{i}", is_static=True, texture_id="ours_wall/1")
    create_box(scene, sapien.Pose(p=[0, 0, 0]), [5.0, 5.0, 0.1], color=(1, 1, 1), name="floor", is_static=True, texture_id="ours_floor/2")
    create_box(scene, sapien.Pose(p=[0, 0, 3.3]), [5.0, 5.0, 0.1], color=(1, 1, 1), name="ceiling", is_static=True)

    # Path configuration.
    input_json = UI_GENERATED_LAYOUT_JSON
    output_dir = ROOM_CONFIG_50ROOMS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists.

    if not input_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")
    
    import json
    with open(input_json, 'r', encoding='utf-8') as f:
        room_config = json.load(f)

    print("\n--- Step 0: Randomly select furniture assets and save the configuration ---")
    random_log = {}
    
    for item in room_config:
        original_path = _resolve_repo_path(item["path"])
        # Layout: .../category/index/mesh/sample.glb
        category_dir = original_path.parents[2]
        
        if category_dir.exists():
            # Scan all numeric directories as candidates.
            candidate_indices = [d.name for d in category_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if candidate_indices:
                chosen_index = random.choice(candidate_indices)
                # Construct the new path.
                new_path = category_dir / chosen_index / original_path.parent.name / original_path.name
                item["path"] = _repo_relative_or_abs(new_path)
                random_log[item["name"]] = chosen_index
            else:
                print(f"[warn] No candidate index found for {item['name']}; using the default.")
    
    # --- Save the updated JSON ---
    # Count existing JSON files in the directory to determine the next filename.
    existing_files = list(output_dir.glob("*.json"))
    new_file_index = len(existing_files)
    save_path = output_dir / f"{new_file_index}.json"
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(room_config, f, indent=2, ensure_ascii=False)
    
    print(f"Randomized configuration saved to: {save_path}")
    print(f"Sampling record: {random_log}")
    print("-" * 30)

    # --- Steps 1 and 2: Physical placement (same logic as before) ---
    asset_metadata = {}
    placed_objects = {}
    align_offset = ALIGN_OFFSET

    for item in room_config:
        model_path = _resolve_repo_path(item["path"])
        center, raw_extents = get_model_metadata(scene, str(model_path))
        temp_pose = get_aligned_pose(0, 0, 0, align_offset + item["rotation_deg"], fix_axis=item.get("fix_axis"))
        world_ext = get_world_extents(raw_extents, temp_pose.q)
        asset_metadata[item["name"]] = {"world_extents": world_ext, "final_quat": temp_pose.q}

    for item in room_config:
        try:
            name = item["name"]
            x, y, z = calculate_semantic_pos(item, asset_metadata, placed_objects)
            pose = sapien.Pose(p=[x, y, z], q=asset_metadata[name]["final_quat"])
            load_custom_asset(scene=scene, pose=pose, model_path=str(_resolve_repo_path(item["path"])), 
                              is_static=item.get("is_static", True), name=name)
            placed_objects[name] = {'pos_2d': (x, y), 'pos_z': z}
        except Exception as e:
            print(f"[error] Failed to place {item['name']}: {e}")
    
    # Export GLB when requested.
    if export and export_path:
        export_path = export_path / f"{new_file_index}.glb"
        export_scene_from_source_json(save_path, str(export_path))


def load_glb_into_scene(scene: sapien.Scene, glb_path: str, is_static: bool = True, name: str = "glb_scene") -> None:
    """Load a GLB/GLTF file into the given SAPIEN scene as a single actor.

    This uses SAPIEN's actor builder to add visual geometry (and attempts a convex
    collision) from the file. If the GLB contains multiple nodes, they will be
    loaded together by the renderer as a single actor.
    """
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if is_static else "dynamic")
    # Add visual geometry from file (SAPIEN supports gltf/glb)
    builder.add_visual_from_file(filename=glb_path, pose=sapien.Pose(), scale=[1.0, 1.0, 1.0])
    # Try to add convex collision if possible (best-effort)
    try:
        builder.add_convex_collision_from_file(filename=glb_path, pose=sapien.Pose(), scale=[1.0, 1.0, 1.0], material=scene.default_physical_material)
    except Exception:
        # Some GLBs don't have suitable collision; ignore if it fails
        pass

    builder.set_initial_pose(sapien.Pose())
    builder.build(name=name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=str, default="our_assets/actor/")
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--table-height", type=float, default=0.74)

    parser.add_argument("--load-glb", type=str, default=None, help="Path to load a GLB/GLTF file into the scene")
    parser.add_argument("--export", action="store_true", default=False, help="Assemble a GLB from the original source GLB files listed in the JSON")
    parser.add_argument("--sources-json", type=str, default=None, help="Optional path to the room JSON to use when exporting from sources")
    parser.add_argument("--export-path", type=str, default=str(ROOM_SCENES_EXPORT_DIR), help="Path to export the current scene to a GLB file")
    args = parser.parse_args()

    # repo_root = Path(__file__).resolve().parents[1]
    _, scene, viewer = _setup_scene(render=args.render)

    # Load a GLB directly when requested; otherwise build the custom scene.
    if args.load_glb:
        load_glb_into_scene(scene, args.load_glb, is_static=True, name="loaded_glb")
    else:
        _create_custom_scene(scene, table_height=args.table_height, export=args.export, export_path=Path(args.export_path))

    for step in range(args.steps):
        scene.step()
        if viewer:
            viewer.render()

if __name__ == "__main__":
    main()