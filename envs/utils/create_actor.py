import sapien.core as sapien
import numpy as np
from pathlib import Path
import transforms3d as t3d
import sapien.physx as sapienp
import json
import os, re
import xml.etree.ElementTree as ET
import trimesh          


from .actor_utils import Actor, ArticulationActor


class UnStableError(Exception):

    def __init__(self, msg):
        super().__init__(msg)


def preprocess(scene, pose: sapien.Pose) -> tuple[sapien.Scene, sapien.Pose]:
    """Add entity to scene. Add bias to z axis if scene is not sapien.Scene."""
    if isinstance(scene, sapien.Scene):
        return scene, pose
    else:
        return scene.scene, sapien.Pose([pose.p[0], pose.p[1], pose.p[2] + scene.table_z_bias], pose.q)


def find_urdf_in_dir(obj_dir: Path) -> Path | None:
    """Find an URDF file inside a custom object directory."""
    urdfs = list(obj_dir.glob("*.urdf"))
    if not urdfs:
        urdfs = list(obj_dir.rglob("*.urdf"))
    if not urdfs:
        return None
    if len(urdfs) == 1:
        return urdfs[0]
    for urdf in urdfs:
        if urdf.stem == obj_dir.name:
            return urdf
    urdfs.sort(key=lambda p: len(str(p)))
    return urdfs[0]


def create_urdf_obj_from_path(
    scene,
    pose: sapien.Pose,
    urdf_path: Path,
    scale: float = 1.0,
    fix_root_link: bool = True,
    name: str | None = None,
    extra_rotation_3x3: np.ndarray | list | None = None,
) -> Actor:
    """Create an actor from a URDF path outside assets/objects."""
    scene, pose = preprocess(scene, pose)
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    model_data_path = urdf_path.parent / "model_data.json"
    model_data = None
    if model_data_path.exists():
        try:
            with open(model_data_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
        except Exception:
            model_data = None

    # Prefer model_data.json scale for custom URDF assets.
    loader_scale = scale
    if isinstance(scale, (list, tuple)):
        loader_scale = float(scale[0]) if len(scale) > 0 else 1.0
    else:
        loader_scale = float(scale)

    if isinstance(model_data, dict):
        md_scale = model_data.get("scale", None)
        if isinstance(md_scale, (list, tuple)) and len(md_scale) > 0:
            loader_scale = float(md_scale[0])
        elif isinstance(md_scale, (int, float)):
            loader_scale = float(md_scale)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = loader_scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    try:
        obj = loader.load(str(urdf_path))
    except Exception as exc:
        # Some URDFs contain multiple objects; fall back to load_multiple.
        if "load_multiple" in str(exc):
            result = loader.load_multiple(str(urdf_path))
            candidates = []
            if isinstance(result, tuple) and len(result) == 2:
                candidates.extend(list(result[0] or []))
                candidates.extend(list(result[1] or []))
            elif isinstance(result, list):
                if result and isinstance(result[0], (list, tuple)):
                    for group in result:
                        candidates.extend(list(group))
                else:
                    candidates.extend(result)
            obj = next((c for c in candidates if c is not None), None)
        else:
            raise
    if obj is None:
        return None

    # --- 使用model_data.json来修正物体位置，确保物体底部在桌面上 ---
    try:
        if isinstance(model_data, dict):
            center = model_data.get("center", [0, 0, 0])
            extents = model_data.get("extents", [0.1, 0.1, 0.1])
            scale_val = model_data.get("scale", 1.0)
            if isinstance(scale_val, (int, float)):
                scale_val = [scale_val, scale_val, scale_val]
            
            # 获取transform_matrix（如果存在）
            trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
            
            # 计算实际的中心偏移和尺寸
            real_center = [center[i] * scale_val[i] for i in range(3)]
            real_extents = [extents[i] * scale_val[i] for i in range(3)]

            # 修正pose：调整X, Y, Z使物体正确放置
            corrected_p = pose.p.copy()
            corrected_p[0] -= real_center[0]  # X轴：修正中心偏移
            corrected_p[1] -= real_center[2]  # Y轴：修正中心偏移（注意：center[2]是Y轴）

            # Final orientation = pose.q @ transform_matrix.
            # transform_matrix is a local mesh-frame correction, so it should be right-multiplied.
            pose_mat = pose.to_transformation_matrix()
            pose_rot = pose_mat[:3, :3]
            R_extra = np.array(trans_mat[:3, :3], dtype=float)
            pose_rot = pose_rot @ R_extra
            if extra_rotation_3x3 is not None:
                R_inst = np.array(extra_rotation_3x3, dtype=float).reshape(3, 3)
                pose_rot = pose_rot @ R_inst
            pose_q = t3d.quaternions.mat2quat(pose_rot)

            # Recompute bottom offset in world Z given the (possibly rotated) pose.
            # model_data axis order in this repo: (x, z, y). Convert to (x, y, z) for world math.
            ext_xzy = np.array(real_extents, dtype=float).reshape(3)
            cen_xzy = np.array(real_center, dtype=float).reshape(3)
            ext_xyz = np.array([ext_xzy[0], ext_xzy[2], ext_xzy[1]], dtype=float)
            cen_xyz = np.array([cen_xzy[0], cen_xzy[2], cen_xzy[1]], dtype=float)
            R_world = np.array(pose_rot, dtype=float).reshape(3, 3)
            z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
            center_proj = float(z_axis @ (R_world @ cen_xyz))
            extent_proj = float(np.sum(np.abs(z_axis @ R_world) * np.abs(ext_xyz)))
            bottom_offset = center_proj - 0.5 * extent_proj
            corrected_p[2] -= bottom_offset  # ensure bottom touches table

            pose = sapien.Pose(corrected_p, pose_q)
        else:
            # 如果没有model_data.json，尝试使用视觉对齐
            if hasattr(obj, "get_links"):
                mins, maxs = np.array([np.inf]*3), np.array([-np.inf]*3)
                has_visuals = False
                
                for link in obj.get_links():
                    link_pose = link.get_pose()
                    p = link_pose.p
                    mins = np.minimum(mins, p)
                    maxs = np.maximum(maxs, p)
                    has_visuals = True
                
                if has_visuals and not np.any(np.isinf(mins)):
                    visual_center_offset = (mins + maxs) / 2.0
                    z_bottom_offset = mins[2]
                    corrected_p = pose.p.copy()
                    corrected_p[0] -= visual_center_offset[0]
                    corrected_p[1] -= visual_center_offset[1]
                    corrected_p[2] -= z_bottom_offset
                    pose = sapien.Pose(corrected_p, pose.q)
    except Exception as e:
        # 如果对齐失败，使用原始pose
        print(f"Warning: Position correction failed for {name}: {e}, using original pose")

    # Apply corrected pose as-is (no forced upright).
    if hasattr(obj, "set_root_pose"):
        obj.set_root_pose(pose)
    else:
        obj.set_pose(pose)

    if name:
        try:
            obj.set_name(name)
        except Exception:
            pass

    def _estimate_model_data_from_urdf(urdf_file: Path) -> dict:
        extents = None
        center = None
        transform_matrix = np.eye(4)
        
        try:
            root = ET.parse(urdf_file).getroot()
            
            # Check for origin rotation in visual/collision elements
            origin_elem = root.find(".//visual/origin")
            if origin_elem is None:
                origin_elem = root.find(".//collision/origin")
            
            if origin_elem is not None:
                rpy_str = origin_elem.get("rpy")
                xyz_str = origin_elem.get("xyz")
                
                if rpy_str:
                    try:
                        r, p, y = [float(v) for v in rpy_str.split()]
                        # Create rotation matrix from roll-pitch-yaw
                        import transforms3d as t3d
                        rot_mat = t3d.euler.euler2mat(r, p, y, axes='sxyz')
                        transform_matrix[:3, :3] = rot_mat
                    except Exception:
                        pass
                
                if xyz_str:
                    try:
                        x, y, z = [float(v) for v in xyz_str.split()]
                        transform_matrix[:3, 3] = [x, y, z]
                    except Exception:
                        pass
            
            mesh_entries = root.findall(".//mesh")
            bounds_min = None
            bounds_max = None
            for mesh in mesh_entries:
                filename = mesh.attrib.get("filename") or mesh.attrib.get("file")
                if not filename:
                    continue
                if filename.startswith("package://"):
                    filename = filename.replace("package://", "", 1)
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    mesh_path = urdf_file.parent / mesh_path
                if not mesh_path.exists():
                    continue
                try:
                    mesh_obj = trimesh.load(mesh_path, force="mesh")
                    if isinstance(mesh_obj, trimesh.Scene):
                        mesh_obj = trimesh.util.concatenate(
                            [g for g in mesh_obj.geometry.values() if g is not None]
                        )
                    scale_attr = mesh.attrib.get("scale")
                    if scale_attr:
                        scale_vals = [float(v) for v in scale_attr.replace(",", " ").split()]
                        if len(scale_vals) == 1:
                            mesh_obj.apply_scale(scale_vals[0])
                        elif len(scale_vals) == 3:
                            mesh_obj.apply_scale(scale_vals)
                    
                    # Apply the origin transform to the mesh vertices for accurate bounds
                    if mesh_obj.vertices is not None and len(mesh_obj.vertices) > 0:
                        vertices_homogeneous = np.hstack([
                            mesh_obj.vertices,
                            np.ones((len(mesh_obj.vertices), 1))
                        ])
                        transformed_vertices = (transform_matrix @ vertices_homogeneous.T).T[:, :3]
                        
                        mesh_min = transformed_vertices.min(axis=0)
                        mesh_max = transformed_vertices.max(axis=0)
                    bounds_min = mesh_min if bounds_min is None else np.minimum(bounds_min, mesh_min)
                    bounds_max = mesh_max if bounds_max is None else np.maximum(bounds_max, mesh_max)
                except Exception:
                    continue
            if bounds_min is not None and bounds_max is not None:
                extents = (bounds_max - bounds_min).tolist()
                center = ((bounds_max + bounds_min) / 2.0).tolist()
        except Exception:
            pass

        # Fallback estimation is no longer allowed; force explicit model_data.json.
        raise FileNotFoundError(
            f"model_data.json is required for URDF asset but not found or invalid: {urdf_file}"
        )

    def _load_or_create_model_data(urdf_file: Path, is_articulation: bool, base_link: str | None) -> dict:
        model_data_path = urdf_file.parent / "model_data.json"
        if model_data_path.exists():
            try:
                with open(model_data_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                # If model_data exists but cannot be read, treat as fatal.
                raise

        # No model_data.json: hard error instead of silent fallback.
        raise FileNotFoundError(
            f"model_data.json not found for URDF asset: {urdf_file}"
        )

    if isinstance(obj, sapien.physx.PhysxArticulation):
        link_name = None
        try:
            links = obj.get_links()
            link_name = links[0].get_name() if links else None
        except Exception:
            link_name = None
        model_data = _load_or_create_model_data(urdf_path, True, link_name)
        return ArticulationActor(obj, model_data)

    model_data = _load_or_create_model_data(urdf_path, False, None)
    return Actor(obj, model_data)


def create_glb_from_path(
    scene,
    pose: sapien.Pose,
    glb_path: str,
    name: str = "glb_asset",
    scale=(1, 1, 1),
    is_static: bool = True,
    collision: bool = False,
    convex_collision: bool = False,
) -> Actor:
    """
    Create an actor from an arbitrary .glb path (not necessarily under assets/objects).

    Typical use: static room/background mesh.
    - collision=False is recommended for background-only visuals (fast + avoids physics side effects).
    """
    scene, pose = preprocess(scene, pose)
    glb_path = str(glb_path)
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"GLB file not found: {glb_path}")

    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if is_static else "dynamic")

    if collision:
        if convex_collision:
            builder.add_multiple_convex_collisions_from_file(filename=glb_path, scale=scale)
        else:
            builder.add_nonconvex_collision_from_file(filename=glb_path, scale=scale)

    builder.add_visual_from_file(filename=glb_path, scale=scale)
    entity = builder.build(name=name)
    entity.set_name(name)
    entity.set_pose(pose)
    return Actor(entity, None)


# create box
def create_entity_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    texture_repeat=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeBox(half_size=half_size, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    if texture_repeat is not None:
        render_component.attach(_create_tiled_box_shape(half_size, texture_repeat, material))
    else:
        render_component.attach(
            # add a box visual shape with given size and rendering material
            sapien.render.RenderShapeBox(half_size, material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def _create_tiled_box_shape(half_size, texture_repeat, material):
    hx, hy, hz = [float(v) for v in half_size]

    def repeat_for(face_width, face_height):
        if isinstance(texture_repeat, (int, float)):
            return face_width * float(texture_repeat), face_height * float(texture_repeat)
        if len(texture_repeat) == 2:
            return float(texture_repeat[0]), float(texture_repeat[1])
        if len(texture_repeat) == 3:
            return face_width / float(texture_repeat[0]), face_height / float(texture_repeat[2])
        raise ValueError(
            "texture_repeat must be a scalar, (u_repeat, v_repeat), or (tile_x, tile_y, tile_z)"
        )

    faces = [
        # vertices, normal, physical width, physical height
        (
            [[hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz], [hx, -hy, hz]],
            [1, 0, 0],
            2 * hy,
            2 * hz,
        ),
        (
            [[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]],
            [-1, 0, 0],
            2 * hy,
            2 * hz,
        ),
        (
            [[-hx, hy, -hz], [-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz]],
            [0, 1, 0],
            2 * hx,
            2 * hz,
        ),
        (
            [[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]],
            [0, -1, 0],
            2 * hx,
            2 * hz,
        ),
        (
            [[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]],
            [0, 0, 1],
            2 * hx,
            2 * hy,
        ),
        (
            [[-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz], [hx, -hy, -hz]],
            [0, 0, -1],
            2 * hx,
            2 * hy,
        ),
    ]
    vertices, normals, uvs, triangles = [], [], [], []
    for face_vertices, normal, face_width, face_height in faces:
        base = len(vertices)
        repeat_u, repeat_v = repeat_for(face_width, face_height)
        vertices.extend(face_vertices)
        normals.extend([normal] * 4)
        uvs.extend([[0, 0], [repeat_u, 0], [repeat_u, repeat_v], [0, repeat_v]])
        triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])

    return sapien.render.RenderShapeTriangleMesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(triangles, dtype=np.uint32),
        np.asarray(normals, dtype=np.float32),
        np.asarray(uvs, dtype=np.float32),
        material,
    )


def create_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
    boxtype="default",
    texture_repeat=None,
) -> Actor:
    entity = create_entity_box(
        scene=scene,
        pose=pose,
        half_size=half_size,
        color=color,
        is_static=is_static,
        name=name,
        texture_id=texture_id,
        texture_repeat=texture_repeat,
    )
    if boxtype == "default":
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(front)
                [
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(right)
                [
                    [-1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(left)
                [
                    [0, 0, -1, 0],
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],  # top_down(back)
                # [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # front
                # [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # right
                # [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # left
                # [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.0], [0, 0, 0, 1]], # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    else:
        data = {
            "center": [0, 0, 0],
            "extents":
            half_size,
            "scale":
            half_size,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0.7], [0, 0, 0, 1]],  # back
                [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # front
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # right
                [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # left
                [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, -0.7], [0, 0, 0, 1]],  # back
            ],
            "transform_matrix":
            np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0, 0.0],
                    [0.0, 0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],  # functional points matrix
            "contact_points_description": [],  # contact points description
            "contact_points_group": [[0, 1, 2, 3, 4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
    return Actor(entity, data)


# create spere
def create_sphere(
    scene,
    pose: sapien.Pose,
    radius: float,
    color=None,
    is_static=False,
    name="",
    texture_id=None,
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = (sapien.physx.PhysxRigidDynamicComponent()
                       if not is_static else sapien.physx.PhysxRigidStaticComponent())
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeSphere(radius=radius, material=scene.default_physical_material))

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/textures/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
    else:
        material = sapien.render.RenderMaterial(base_color=[*color[:3], 1])

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeSphere(radius=radius, material=material))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create cylinder
def create_cylinder(
    scene,
    pose: sapien.Pose,
    radius: float,
    half_length: float,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create PhysX dynamic rigid body
    rigid_component = sapien.physx.PhysxRigidDynamicComponent()
    rigid_component.attach(
        sapien.physx.PhysxCollisionShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=scene.default_physical_material,
        ))

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeCylinder(
            radius=radius,
            half_length=half_length,
            material=sapien.render.RenderMaterial(base_color=[*color[:3], 1]),
        ))

    entity.add_component(rigid_component)
    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


# create box
def create_visual_box(
    scene,
    pose: sapien.Pose,
    half_size,
    color=None,
    name="",
) -> sapien.Entity:
    scene, pose = preprocess(scene, pose)

    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    # create render body for visualization
    render_component = sapien.render.RenderBodyComponent()
    render_component.attach(
        # add a box visual shape with given size and rendering material
        sapien.render.RenderShapeBox(half_size, sapien.render.RenderMaterial(base_color=[*color[:3], 1])))

    entity.add_component(render_component)
    entity.set_pose(pose)

    # in general, entity should only be added to scene after it is fully built
    scene.add_entity(entity)
    return entity


def create_table(
        scene,
        pose: sapien.Pose,
        length: float,
        width: float,
        height: float,
        thickness=0.1,
        color=(1, 1, 1),
        name="table",
        is_static=True,
        texture_id=None,
        texture_repeat=None,
) -> sapien.Entity:
    """Create a table with specified dimensions."""
    scene, pose = preprocess(scene, pose)
    builder = scene.create_actor_builder()

    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    # Tabletop
    tabletop_pose = sapien.Pose([0.0, 0.0, -thickness / 2])  # Center the tabletop at z=0
    tabletop_half_size = [length / 2, width / 2, thickness / 2]
    builder.add_box_collision(
        pose=tabletop_pose,
        half_size=tabletop_half_size,
        material=scene.default_physical_material,
    )

    # Add texture
    if texture_id is not None:

        # test for both .png and .jpg
        texturepath = f"./assets/background_texture/{texture_id}.png"
        # create texture from file
        texture2d = sapien.render.RenderTexture2D(texturepath)
        material = sapien.render.RenderMaterial()
        material.set_base_color_texture(texture2d)
        # renderer.create_texture_from_file(texturepath)
        # material.set_diffuse_texture(texturepath)
        material.base_color = [1, 1, 1, 1]
        material.metallic = 0.1
        material.roughness = 0.3
        if texture_repeat is None:
            builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, material=material)
    else:
        builder.add_box_visual(
            pose=tabletop_pose,
            half_size=tabletop_half_size,
            material=color,
        )

    # Table legs (x4)
    leg_spacing = 0.1
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (length / 2 - leg_spacing / 2)
            y = j * (width / 2 - leg_spacing / 2)
            table_leg_pose = sapien.Pose([x, y, -height / 2 - 0.002])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2 - 0.002]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, material=color)

    builder.set_initial_pose(pose)
    table = builder.build(name=name)
    if texture_id is not None and texture_repeat is not None:
        tabletop_shape = _create_tiled_box_shape(tabletop_half_size, texture_repeat, material)
        tabletop_shape.set_local_pose(tabletop_pose)
        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(tabletop_shape)
        table.add_component(render_component)
    return table


# create obj model
def create_obj(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=None,
        convex=False,
        is_static=False,
        model_id=None,
        no_collision=False,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "textured.obj"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"textured{model_id}.obj"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        if scale is None:
            scale = model_data["scale"]
    except (OSError, KeyError, json.JSONDecodeError):
        model_data = None
    if scale is None:
        scale = (1, 1, 1)

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if not no_collision:
        if convex == True:
            builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
        else:
            builder.add_nonconvex_collision_from_file(filename=str(file_name), scale=scale)

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


# create glb model
def create_glb(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=None,
        convex=False,
        is_static=False,
        model_id=None,
) -> Actor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    if model_id is None:
        file_name = modeldir / "base.glb"
        json_file_path = modeldir / "model_data.json"
    else:
        file_name = modeldir / f"base{model_id}.glb"
        json_file_path = modeldir / f"model_data{model_id}.json"

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        if scale is None:
            scale = model_data["scale"]
    except (OSError, KeyError, json.JSONDecodeError):
        model_data = None
    if scale is None:
        scale = (1, 1, 1)

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(file_name), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(file_name),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(file_name), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_pose(pose)

    return Actor(mesh, model_data)


def get_glb_or_obj_file(modeldir, model_id):
    modeldir = Path(modeldir)
    if model_id is None:
        file = modeldir / "base.glb"
    else:
        file = modeldir / f"base{model_id}.glb"
    if not file.exists():
        if model_id is None:
            file = modeldir / "textured.obj"
        else:
            file = modeldir / f"textured{model_id}.obj"
    return file


def create_actor(
        scene,
        pose: sapien.Pose,
        modelname: str,
        scale=None,
        convex=False,
        is_static=False,
        model_id=0,
) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets/objects") / modelname

    if model_id is None:
        json_file_path = modeldir / "model_data.json"
    else:
        json_file_path = modeldir / f"model_data{model_id}.json"

    collision_file = ""
    visual_file = ""
    if (modeldir / "collision").exists():
        collision_file = get_glb_or_obj_file(modeldir / "collision", model_id)
    if collision_file == "" or not collision_file.exists():
        collision_file = get_glb_or_obj_file(modeldir, model_id)

    if (modeldir / "visual").exists():
        visual_file = get_glb_or_obj_file(modeldir / "visual", model_id)
    if visual_file == "" or not visual_file.exists():
        visual_file = get_glb_or_obj_file(modeldir, model_id)

    if not collision_file.exists() or not visual_file.exists():
        print(modelname, "is not exist model file!")
        return None

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        if scale is None:
            scale = model_data["scale"]
    except (OSError, KeyError, json.JSONDecodeError):
        model_data = None
    if scale is None:
        scale = (1, 1, 1)

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    if convex == True:
        builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
    else:
        builder.add_nonconvex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
        )

    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
    mesh = builder.build(name=modelname)
    mesh.set_name(modelname)
    mesh.set_pose(pose)
    return Actor(mesh, model_data)


# create urdf model
def create_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("./assets/objects") / modelname
    json_file_path = modeldir / "model_data.json"
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale

    try:
        with open(json_file_path, "r") as file:
            model_data = json.load(file)
        loader.scale = model_data["scale"][0]
    except:
        model_data = None

    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object: sapien.Articulation = loader.load(str(modeldir / "mobility.urdf"))

    object.set_root_pose(pose)
    object.set_name(modelname)
    return ArticulationActor(object, model_data)


def create_sapien_urdf_obj(
    scene,
    pose: sapien.Pose,
    modelname: str,
    scale=1.0,
    modelid: int = None,
    fix_root_link=False,
) -> ArticulationActor:
    scene, pose = preprocess(scene, pose)

    modeldir = Path("assets") / "objects" / modelname
    if modelid is not None:
        model_list = [model for model in modeldir.iterdir() if model.is_dir() and model.name != "visual"]

        def extract_number(filename):
            match = re.search(r"\d+", filename.name)
            return int(match.group()) if match else 0

        model_list = sorted(model_list, key=extract_number)

        if modelid >= len(model_list):
            is_find = False
            for model in model_list:
                if modelid == int(model.name):
                    modeldir = model
                    is_find = True
                    break
            if not is_find:
                raise ValueError(f"modelid {modelid} is out of range for {modelname}.")
        else:
            modeldir = model_list[modelid]
    json_file = modeldir / "model_data.json"

    if json_file.exists():
        with open(json_file, "r") as file:
            model_data = json.load(file)
        scale = model_data["scale"]
        trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
    else:
        model_data = None
        trans_mat = np.eye(4)

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = True
    object = loader.load_multiple(str(modeldir / "mobility.urdf"))[0][0]

    pose_mat = pose.to_transformation_matrix()
    pose = sapien.Pose(
        p=pose_mat[:3, 3] + trans_mat[:3, 3],
        q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
    )
    object.set_pose(pose)

    if model_data is not None:
        if "init_qpos" in model_data and len(model_data["init_qpos"]) > 0:
            object.set_qpos(np.array(model_data["init_qpos"]))
        if "mass" in model_data and len(model_data["mass"]) > 0:
            for link in object.get_links():
                link.set_mass(model_data["mass"].get(link.get_name(), 0.1))

        bounding_box_file = modeldir / "bounding_box.json"
        if bounding_box_file.exists():
            bounding_box = json.load(open(bounding_box_file, "r", encoding="utf-8"))
            model_data["extents"] = (np.array(bounding_box["max"]) - np.array(bounding_box["min"])).tolist()
    object.set_name(modelname)
    return ArticulationActor(object, model_data)
