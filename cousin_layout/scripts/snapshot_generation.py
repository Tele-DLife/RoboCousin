import os
# Force EGL mode for headless servers.
# NOTE: pyrender/pyglet/OpenGL backend selection must happen before importing pyrender/PyOpenGL.
# PYRENDER_BACKEND alone is sometimes insufficient (pyglet may still try to connect to DISPLAY=None).
os.environ.setdefault("PYRENDER_BACKEND", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import trimesh
import pyrender
import numpy as np
import cv2
import argparse
import json
from pathlib import Path
from typing import Optional

try:
    from script.path_config import DIGITAL_COUSINS_OUR_OBJECTS_DIR
except ModuleNotFoundError:
    DIGITAL_COUSINS_OUR_OBJECTS_DIR = (Path(__file__).resolve().parents[2] / "our_assets" / "actor").resolve()

def _load_mesh_with_model_transform(model_path: str):
    # 1. Load the model.
    tm_scene = trimesh.load(model_path)

    # Compute model size automatically to adjust the rendering distance.
    # Larger models require a larger distance.
    if isinstance(tm_scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(tm_scene.dump())
    else:
        mesh = tm_scene

    # Before rendering, apply the transform_matrix in the adjacent
    # model_data.json to the mesh. Use the identity matrix when it is missing
    # or cannot be read.
    transform = np.eye(4, dtype=float)
    transform_is_identity = True
    model_data_path: Optional[Path] = None
    try:
        model_path_obj = Path(model_path)
        # model_path: .../<instance>/mesh/sample.(glb|obj)
        # model_data.json is sibling of "mesh" directory: .../<instance>/model_data.json
        model_data_path = model_path_obj.parent.parent / "model_data.json"
        if model_data_path.exists():
            with open(model_data_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
            tm_raw = model_data.get("transform_matrix", None)
            tm = np.array(tm_raw, dtype=float) if tm_raw is not None else None
            if tm is not None and tm.shape == (4, 4) and np.isfinite(tm).all():
                transform = tm
                transform_is_identity = bool(np.allclose(transform, np.eye(4, dtype=float), atol=1e-8, rtol=1e-6))
    except Exception as e:
        src = str(model_data_path) if model_data_path is not None else "<unknown model_data.json>"
        print(f"Warning: failed to read transform_matrix from {src}; using the identity matrix: {e}")

    # Only apply if transform_matrix is not identity.
    if not transform_is_identity:
        mesh.apply_transform(transform)
    return mesh


def _compute_eye_from_cam_pose_world(cam_pose_world, render_distance: float, fallback_height: float) -> np.ndarray:
    """
    Use the position in cam_pose_world to determine the viewing direction and
    elevation. The input world frame is z-up, while the rendering camera is
    y-up. Convert coordinates first, then normalize only the horizontal radius
    to render_distance while preserving the real height component so small
    models do not lower the camera height.
    """
    if not isinstance(cam_pose_world, dict):
        return np.array([0.0, fallback_height, render_distance], dtype=float)
    pos = cam_pose_world.get("position", None)
    if not isinstance(pos, (list, tuple)) or len(pos) < 3:
        return np.array([0.0, fallback_height, render_distance], dtype=float)
    try:
        # world(z-up) -> render(y-up): y' = z, z' = -y
        p_world = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=float)
        p = np.array([p_world[0], p_world[2], -p_world[1]], dtype=float)
    except Exception:
        return np.array([0.0, fallback_height, render_distance], dtype=float)
    if not np.isfinite(p).all():
        return np.array([0.0, fallback_height, render_distance], dtype=float)

    # Keep camera height absolute; only normalize horizontal direction/radius.
    height = float(p[1]) if np.isfinite(p[1]) else float(fallback_height)
    horizontal = np.array([p[0], 0.0, p[2]], dtype=float)
    h_norm = float(np.linalg.norm(horizontal))
    if h_norm < 1e-8:
        eye = np.array([0.0, height, render_distance], dtype=float)
    else:
        eye = horizontal / h_norm * float(render_distance)
        eye[1] = height
    return eye


def resolve_model_path_from_asset_instance(asset_instance_dir: str) -> Path | None:
    """
    Find a renderable model in an asset instance directory
    (.../<class>/<id>). Supports sample.glb, sample.obj, and
    sample_collision.obj.
    """
    inst = Path(asset_instance_dir)
    mesh_dir = inst / "mesh"
    candidates = [
        mesh_dir / "sample.glb",
        mesh_dir / "sample.obj",
        mesh_dir / "sample_collision.obj",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def render_calibrated_views(
    model_path,
    output_folder="renders",
    distance_factor=2.0,
    camera_height=0.2,
    step_degrees=22.5,
    fixed_camera_pose_world=None,
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mesh = _load_mesh_with_model_transform(model_path=model_path)

    # Get bounding-box extents.
    model_scale = np.max(mesh.extents)
    # Effective rendering distance = model size * distance factor.
    render_distance = model_scale * distance_factor

    # Fixed-camera mode: use the cam_pose_world direction (with radius
    # normalized to render_distance) and rotate the object instead of orbiting
    # the camera to generate multi-view snapshots.
    fixed_eye = _compute_eye_from_cam_pose_world(
        cam_pose_world=fixed_camera_pose_world,
        render_distance=render_distance,
        fallback_height=float(camera_height),
    ) if fixed_camera_pose_world is not None else None

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

    # Set up the renderer.
    r = pyrender.OffscreenRenderer(800, 800)

    # Render each view.
    angles_deg = np.arange(0, 360, float(step_degrees))
    for i, angle_deg in enumerate(angles_deg):
        if fixed_eye is None:
            # Default: orbit the camera around the object.
            angle_rad = np.radians(angle_deg)
            x = render_distance * np.sin(angle_rad)
            z = render_distance * np.cos(angle_rad)
            eye = np.array([x, camera_height, z], dtype=float)
            mesh_frame = mesh
        else:
            # Fixed camera: rotate the object around the y axis.
            eye = fixed_eye.copy()
            mesh_frame = mesh.copy()
            theta = np.radians(angle_deg)
            c, s = np.cos(theta), np.sin(theta)
            rot_y = np.array(
                [
                    [c, 0.0, s, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-s, 0.0, c, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            mesh_frame.apply_transform(rot_y)

        scene = pyrender.Scene.from_trimesh_scene(trimesh.Scene(mesh_frame), bg_color=[1.0, 1.0, 1.0])
        scene.ambient_light = [0.5, 0.5, 0.5]
        
        # Construct the LookAt matrix.
        target = np.array([0, 0, 0])
        up = np.array([0, 1, 0])
        z_axis = (eye - target) / np.linalg.norm(eye - target)
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        
        cp = np.eye(4)
        cp[:3, 0] = x_axis
        cp[:3, 1] = y_axis
        cp[:3, 2] = z_axis
        cp[:3, 3] = eye

        # Render this frame.
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_node = scene.add(camera, pose=cp)
        light_node = scene.add(light, pose=cp)  # Move the light with the camera.
        
        color, _ = r.render(scene)
        
        # Save the image.
        file_path = os.path.join(output_folder, f"{i}.jpg")
        cv2.imwrite(file_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        
        # Remove the nodes.
        scene.remove_node(cam_node)
        scene.remove_node(light_node)

    r.delete()
    print(f"Complete. Images saved to: {output_folder}")
    print(f"Automatic rendering distance: {render_distance:.2f}")


def render_snapshots_for_asset_instance(
    asset_instance_dir: str,
    overwrite: bool = True,
    distance_factor: float = 1.5,
    camera_height: float = 0.2,
    step_degrees: float = 22.5,
    fixed_camera_pose_world=None,
) -> Path | None:
    instance_dir = Path(asset_instance_dir)
    if not instance_dir.exists() or not instance_dir.is_dir():
        return None
    model_path = resolve_model_path_from_asset_instance(str(instance_dir))
    if model_path is None:
        return None
    snapshot_dir = instance_dir / "snapshot"
    if snapshot_dir.exists() and not overwrite:
        existing = [p for p in snapshot_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if existing:
            return snapshot_dir
    render_calibrated_views(
        model_path=str(model_path),
        output_folder=str(snapshot_dir),
        distance_factor=distance_factor,
        camera_height=camera_height,
        step_degrees=step_degrees,
        fixed_camera_pose_world=fixed_camera_pose_world,
    )
    return snapshot_dir

def process_all_objects(our_objects_dir, overwrite=False, distance_factor=1.5, camera_height=0.2):
    """
    Iterate over all objects under our_objects_dir and generate snapshots for
    each object.
    
    Args:
        our_objects_dir: Path to the our_objects directory.
        overwrite: Whether to overwrite existing snapshots.
        distance_factor: Rendering distance factor.
        camera_height: Camera height.
    """
    our_objects_path = Path(our_objects_dir)
    if not our_objects_path.exists():
        print(f"Error: directory does not exist: {our_objects_dir}")
        return
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # Iterate over all object directories.
    for object_dir in sorted(our_objects_path.iterdir()):
        if not object_dir.is_dir():
            continue
        
        object_name = object_dir.name
        print(f"\nProcessing object: {object_name}")
        
        # Iterate over all versions of this object (0, 1, 2, etc.).
        for version_dir in sorted(object_dir.iterdir()):
            if not version_dir.is_dir() or not version_dir.name.isdigit():
                continue
            
            version = version_dir.name
            model_path = version_dir / "mesh" / "sample.glb"
            snapshot_dir = version_dir / "snapshot"
            
            # Check whether the model file exists.
            if not model_path.exists():
                print(f"  Skipping {object_name}/{version}: model file not found ({model_path})")
                continue
            
            # Check whether the snapshot directory already contains rendered images.
            existing_snapshots = []
            if snapshot_dir.exists() and snapshot_dir.is_dir():
                existing_snapshots = [
                    f for f in snapshot_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]

            if existing_snapshots and not overwrite:
                print(f"  Skipping {object_name}/{version}: snapshots already exist (use --overwrite to replace them)")
                skipped_count += 1
                continue
            elif existing_snapshots and overwrite:
                print(f"  Overwriting {object_name}/{version}: regenerating snapshots")
            
            # Generate snapshots.
            try:
                print(f"  Generating snapshots for {object_name}/{version}...")
                render_calibrated_views(
                    model_path=str(model_path),
                    output_folder=str(snapshot_dir),
                    distance_factor=distance_factor,
                    camera_height=camera_height,
                )
                processed_count += 1
            except Exception as e:
                print(f"  Error while processing {object_name}/{version}: {e}")
                error_count += 1
    
    print("\nComplete.")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate snapshots for all objects under our_objects.")
    parser.add_argument(
        "--our-objects-dir",
        type=str,
        default=str(DIGITAL_COUSINS_OUR_OBJECTS_DIR),
        help="Path to the our_objects directory."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing snapshots (default: do not overwrite)."
    )
    parser.add_argument(
        "--distance-factor",
        type=float,
        default=1.5,
        help="Rendering distance factor (default: 1.5; 1.5 is close, 2.5 is farther, and 3.5 is distant)."
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=0.1,
        help="Camera height (default: 0.2; 0.0 is level view and 0.5 is top-down)."
    )
    
    args = parser.parse_args()
    
    process_all_objects(
        our_objects_dir=args.our_objects_dir,
        overwrite=args.overwrite,
        distance_factor=args.distance_factor,
        camera_height=args.camera_height
    )


