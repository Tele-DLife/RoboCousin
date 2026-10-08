import sapien.core as sapien
import numpy as np
import transforms3d as t3d
import sapien.physx as sapienp
from .create_actor import *

from .actor_utils import ArticulationActor

import re
import json
from pathlib import Path
from .._GLOBAL_CONFIGS import DEFAULT_CUSTOM_DYNAMIC_MASS_KG, ROOT_PATH


def get_all_cluttered_objects():
    cluttered_objects_info = {}
    cluttered_objects_name = []

    # 1. 加载 objaverse (保持原样，因为它是准的)
    cluttered_objects_config = json.load(open(Path("./assets/objects/objaverse/list.json"), "r", encoding="utf-8"))
    cluttered_objects_name += cluttered_objects_config["item_names"]
    for model_name, model_ids in cluttered_objects_config["list_of_items"].items():
        cluttered_objects_info[model_name] = {
            "ids": model_ids, "type": "urdf", "root": f"objects/objaverse/{model_name}",
        }
        params = {}
        for model_id in model_ids:
            model_full_name = f"{model_name}_{model_id}"
            params[model_id] = {
                "z_max": cluttered_objects_config["z_max"][model_full_name],
                "radius": cluttered_objects_config["radius"][model_full_name],
                "z_offset": cluttered_objects_config["z_offset"][model_full_name],
            }
        cluttered_objects_info[model_name]["params"] = params

    # 2. Repo-root custom assets: our_assets/actor + our_assets/non-actor
    repo_root = Path(ROOT_PATH)

    def _scan_custom_tree(rel_prefix: str) -> None:
        """Register models under repo_root/rel_prefix (e.g. our_assets/actor)."""
        custom_objects_dir = repo_root.joinpath(*rel_prefix.split("/"))
        if not custom_objects_dir.exists():
            return
        for model_dir in custom_objects_dir.iterdir():
            if not model_dir.is_dir() or re.search(r"^(\d+)_", model_dir.name):
                continue

            model_name = model_dir.name
            model_id_list, params = [], {}
            instance_dirs = [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
            if not instance_dirs:
                instance_dirs = [model_dir]

            for instance_dir in sorted(
                instance_dirs,
                key=lambda p: int(p.name) if p.name.isdigit() else -1
            ):
                model_id = instance_dir.name if instance_dir.name.isdigit() else "0"
                urdf_files = list(instance_dir.glob("*.urdf"))
                model_data_files = list(instance_dir.glob("model_data*.json"))
                if not urdf_files or not model_data_files:
                    continue

                model_cfg = instance_dir / "model_data.json"
                if not model_cfg.exists():
                    model_cfg = model_data_files[0]

                try:
                    config = json.load(open(model_cfg, "r", encoding="utf-8"))
                    center = config.get("center", [0, 0, 0])
                    extents = config.get("extents", [0.1, 0.1, 0.1])
                    scale = config.get("scale", 1.0)
                    if isinstance(scale, (int, float)):
                        scale = [scale, scale, scale]

                    real_extents = [extents[i] * scale[i] for i in range(3)]
                    real_center = [center[i] * scale[i] for i in range(3)]

                    radius_x = (real_extents[0] / 2.0) + abs(real_center[0])
                    radius_y = (real_extents[2] / 2.0) + abs(real_center[2])

                    z_max = real_extents[1]

                    bottom_y = real_center[1] - real_extents[1] / 2.0
                    z_offset = bottom_y

                    params[model_id] = {
                        "z_max": z_max,
                        "radius": max(radius_x, radius_y) + 0.02,
                        "z_offset": z_offset,
                    }
                    model_id_list.append(model_id)
                except Exception as e:
                    print(f"Error loading {model_name} from {model_cfg}: {e}")

            if not model_id_list:
                continue

            reg_key = model_name
            if reg_key in cluttered_objects_info and rel_prefix.endswith("non-actor"):
                prev_root = str(cluttered_objects_info[reg_key].get("root", ""))
                if prev_root.startswith("our_assets/actor/"):
                    reg_key = f"{model_name}__na"

            if reg_key not in cluttered_objects_name:
                cluttered_objects_name.append(reg_key)
            cluttered_objects_info[reg_key] = {
                "ids": model_id_list,
                "type": "urdf",
                "root": f"{rel_prefix}/{model_name}",
                "params": params,
            }

    _scan_custom_tree("our_assets/actor")
    _scan_custom_tree("our_assets/non-actor")

    cluttered_objects_name.sort()
    return cluttered_objects_info, cluttered_objects_name, {}


cluttered_objects_info, cluttered_objects_list, same_obj = get_all_cluttered_objects()


def get_available_cluttered_objects(entity_on_scene: list):
    global cluttered_objects_info, cluttered_objects_list, same_obj

    model_in_use = []
    for entity_name in entity_on_scene:
        if same_obj.get(entity_name) is not None:
            model_in_use += same_obj[entity_name]
        model_in_use.append(entity_name)

    available_models = set(cluttered_objects_list) - set(model_in_use)
    available_models = list(available_models)
    available_models.sort()
    return available_models, cluttered_objects_info


def check_overlap(radius, x, y, area):
    if x <= area[0]:
        dx = area[0] - x
    elif area[0] < x and x < area[2]:
        dx = 0
    elif x >= area[2]:
        dx = x - area[2]
    if y <= area[1]:
        dy = area[1] - y
    elif area[1] < y and y < area[3]:
        dy = 0
    elif y >= area[3]:
        dy = y - area[3]

    return dx * dx + dy * dy <= radius * radius


def rand_pose_cluttered(
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=[1, 0, 0, 0],
    size_dict=None,
    obj_radius=0.1,
    z_offset=0.001,
    z_max=0,
    prohibited_area=None,
    obj_margin=0.005,
) -> sapien.Pose:
    if len(xlim) < 2 or xlim[1] < xlim[0]:
        xlim = np.array([xlim[0], xlim[0]])
    if len(ylim) < 2 or ylim[1] < ylim[0]:
        ylim = np.array([ylim[0], ylim[0]])
    if len(zlim) < 2 or zlim[1] < zlim[0]:
        zlim = np.array([zlim[0], zlim[0]])

    times = 0
    while True:
        times += 1
        if times > 100:
            return False, None
        
        new_obj_radius = obj_radius + obj_margin
        
        # Calculate valid range for object center to ensure it stays within bounds
        x_min_valid = xlim[0] + new_obj_radius
        x_max_valid = xlim[1] - new_obj_radius
        y_min_valid = ylim[0] + new_obj_radius
        y_max_valid = ylim[1] - new_obj_radius
        
        # Ensure we have valid range
        if x_min_valid >= x_max_valid or y_min_valid >= y_max_valid:
            return False, None
        
        # Sample position ensuring object center is within valid range
        x = np.random.uniform(x_min_valid, x_max_valid)
        y = np.random.uniform(y_min_valid, y_max_valid)
        
        # Verify bounds (double check)
        if x - new_obj_radius < xlim[0] or x + new_obj_radius > xlim[1]:
            continue
        if y - new_obj_radius < ylim[0] or y + new_obj_radius > ylim[1]:
            continue
        
        # Check prohibited areas
        is_overlap = False
        if prohibited_area is not None:
            for area in prohibited_area:
                if check_overlap(new_obj_radius, x, y, area):
                    is_overlap = True
                    break
        if is_overlap:
            continue
        
        # Check overlap with existing objects
        if size_dict is not None and len(size_dict) > 0:
            distances = np.sqrt((np.array([sub_list[0] for sub_list in size_dict]) - x)**2 +
                                (np.array([sub_list[1] for sub_list in size_dict]) - y)**2)
            max_distances = np.array([sub_list[3] + new_obj_radius + obj_margin for sub_list in size_dict])
            if not np.all(distances > max_distances):
                continue
        
        # Additional check for table edge (original logic)
        if y - new_obj_radius < 0:
            if z_max > 0.05:
                continue
        
        # All checks passed
        break

    z = np.random.uniform(zlim[0], zlim[1])
    z = z - z_offset

    rotate = qpos
    if rotate_rand:
        # Only allow rotation around Z-axis (vertical) to keep objects upright
        # rotate_lim[0] = X-axis rotation (roll) - keep at 0
        # rotate_lim[1] = Y-axis rotation (pitch) - keep at 0  
        # rotate_lim[2] = Z-axis rotation (yaw) - allow rotation
        angles = [0, 0, 0]  # Initialize all to 0
        # Only apply Z-axis rotation (yaw)
        if len(rotate_lim) > 2 and rotate_lim[2] > 0:
            angles[2] = np.random.uniform(-rotate_lim[2], rotate_lim[2])
        # X and Y rotations are kept at 0 to prevent objects from lying down
        # Create rotation quaternion with only Z-axis rotation
        rotate_quat = t3d.euler.euler2quat(0, 0, angles[2], axes='sxyz')
        # Multiply with base quaternion
        rotate = t3d.quaternions.qmult(rotate, rotate_quat)
        # Normalize quaternion
        rotate = np.array(rotate)
        rotate = rotate / np.linalg.norm(rotate)
        rotate = rotate.tolist()

    return True, sapien.Pose([x, y, z], rotate)


def rand_create_cluttered_actor(
    scene,
    modelname: str,
    modelid: str,
    modeltype: str,
    xlim: np.ndarray,
    ylim: np.ndarray,
    zlim: np.ndarray,
    ylim_prop=False,
    rotate_rand=False,
    rotate_lim=[0, 0, 0],
    qpos=None,
    scale=(1, 1, 1),
    convex=True,
    is_static=False,
    size_dict=None,
    obj_radius=0.1,
    z_offset=0.001,
    z_max=0,
    fix_root_link=True,
    prohibited_area=None,
    fixed_pose=None,
    extra_rotation_3x3=None,
) -> tuple[bool, Actor | None]:
    """
    fixed_pose: If not None, use this pose instead of random sampling (for cousin-coordinate mode).
    """
    if qpos is None:
        if modeltype == "glb":
            qpos = [0.707107, 0.707107, 0, 0]
            rotate_lim = [rotate_lim[0], rotate_lim[2], rotate_lim[1]]
        else:
            # For URDF objects, use identity quaternion to ensure they stand upright
            # [1, 0, 0, 0] is identity quaternion (no rotation)
            qpos = [1, 0, 0, 0]

    if fixed_pose is not None:
        obj_pose = fixed_pose
    else:
        success, obj_pose = rand_pose_cluttered(
            xlim=xlim,
            ylim=ylim,
            zlim=zlim,
            ylim_prop=ylim_prop,
            rotate_rand=rotate_rand,
            rotate_lim=rotate_lim,
            qpos=qpos,
            size_dict=size_dict,
            obj_radius=obj_radius,
            z_offset=z_offset,
            z_max=z_max,
            prohibited_area=prohibited_area,
        )
        if not success:
            return False, None

    if modeltype == "urdf":
        if modelname.startswith("our_assets/"):
            model_dir = Path(ROOT_PATH) / modelname / str(modelid)
            if not model_dir.exists():
                model_dir = Path(ROOT_PATH) / modelname
            urdf_path = find_urdf_in_dir(model_dir)
            if urdf_path and urdf_path.exists():
                obj = create_urdf_obj_from_path(
                    scene=scene,
                    pose=obj_pose,
                    urdf_path=urdf_path,
                    scale=scale if isinstance(scale, float) else scale[0],
                    fix_root_link=fix_root_link,
                    name=modelname.split("/")[-1],
                    extra_rotation_3x3=extra_rotation_3x3,
                )
            else:
                print(f"Warning: URDF not found for {modelname} in {model_dir}")
                obj = None
        elif modelname.startswith("objects/objaverse/"):
            # Objaverse URDF - modelname is full path like "objects/objaverse/apple"
            actual_modelname = modelname.replace("objects/objaverse/", "")
            obj = create_cluttered_urdf_obj(
                scene=scene,
                pose=obj_pose,
                modelname=f"objects/objaverse/{actual_modelname}/{modelid}",
                scale=scale if isinstance(scale, float) else scale[0],
                fix_root_link=fix_root_link,
            )
        else:
            # Fallback: assume it's just the model name for objaverse
            obj = create_cluttered_urdf_obj(
                scene=scene,
                pose=obj_pose,
                modelname=f"objects/objaverse/{modelname}/{modelid}",
                scale=scale if isinstance(scale, float) else scale[0],
                fix_root_link=fix_root_link,
            )
        if obj is None:
            return False, None
        _m = float(DEFAULT_CUSTOM_DYNAMIC_MASS_KG)
        if isinstance(obj, ArticulationActor):
            links = list(obj.actor.get_links())
            n = len(links) or 1
            for L in links:
                L.set_mass(_m / n)
        else:
            obj.set_mass(_m)
        return True, obj
    else:
        model_path = Path(ROOT_PATH) / modelname if modelname.startswith("our_assets/") else None

        if modelname.startswith("our_assets/"):
            if model_path.exists():
                # Try to load as regular actor from custom path
                try:
                    json_file_path = model_path / f"model_data{modelid}.json"
                    with open(json_file_path, "r") as file:
                        model_data = json.load(file)
                        scale = model_data["scale"]
                except:
                    model_data = None
                    scale = (1, 1, 1)
                
                collision_file = get_glb_or_obj_file(model_path, modelid)
                if not collision_file.exists():
                    return False, None
                
                builder = scene.create_actor_builder()
                builder.set_physx_body_type("dynamic" if not is_static else "static")
                if convex:
                    builder.add_multiple_convex_collisions_from_file(filename=str(collision_file), scale=scale)
                else:
                    builder.add_nonconvex_collision_from_file(filename=str(collision_file), scale=scale)
                
                visual_file = get_glb_or_obj_file(model_path / "visual" if (model_path / "visual").exists() else model_path, modelid)
                if visual_file.exists():
                    builder.add_visual_from_file(filename=str(visual_file), scale=scale)
                else:
                    builder.add_visual_from_file(filename=str(collision_file), scale=scale)
                
                mesh = builder.build(name=modelname.split("/")[-1])
                mesh.set_pose(obj_pose)
                from .actor_utils import Actor
                obj = Actor(mesh, model_data)
            else:
                return False, None
        else:
            # Default objects directory - extract just the model name
            # modelname is like "objects/apple" or just "apple"
            actual_modelname = modelname.split("/")[-1] if "/" in modelname else modelname
            obj = create_actor(
                scene=scene,
                pose=obj_pose,
                modelname=actual_modelname,
                model_id=modelid,
                scale=scale,
                convex=convex,
                is_static=is_static,
            )
        
        if obj is None:
            return False, None
        _m = float(DEFAULT_CUSTOM_DYNAMIC_MASS_KG)
        if isinstance(obj, ArticulationActor):
            links = list(obj.actor.get_links())
            n = len(links) or 1
            for L in links:
                L.set_mass(_m / n)
        else:
            obj.set_mass(_m)
        return True, obj


def create_cluttered_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets") / modelname
    urdf_file = str(modeldir / "model.urdf")

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = False
    
    # 加载模型
    result = loader.load_multiple(urdf_file)
    object: sapien.Articulation = result[1][0]

    # --- 核心：修正视觉偏移 (Visual Alignment) ---
    # 获取该模型所有 Link 的视觉边界
    mins, maxs = np.array([np.inf]*3), np.array([-np.inf]*3)
    has_visuals = False
    
    for link in object.get_links():
        for vb in link.get_visual_bodies():
            # 获取每个 visual body 的 AABB
            # 注意：这里我们取 link 自身的位移作为参考，
            # 解决 Mesh 相对于 Root Link 偏移的问题
            p = link.get_pose().p
            mins = np.minimum(mins, p)
            maxs = np.maximum(maxs, p)
            has_visuals = True

    if has_visuals:
        # 计算视觉中心相对于 Root Link 的偏移量
        visual_center_offset = (mins + maxs) / 2.0
        # 忽略 Z 轴偏移（因为我们希望它基于桌面高度放置），只修正 X 和 Y
        visual_center_offset[2] = 0 
        
        # 修正 Pose：减去偏移，让物体的“肚子”出现在采样点，而不是它的“原点”
        corrected_p = pose.p - visual_center_offset
        pose = sapien.Pose(corrected_p, pose.q)

    # 应用修正后的位置
    object.set_pose(pose)

    if isinstance(object, sapien.physx.PhysxArticulation):
        return ArticulationActor(object, None)
    else:
        return Actor(object, None)

def create_cluttered_urdf_obj(scene, pose: sapien.Pose, modelname: str, scale=1.0, fix_root_link=True) -> Actor:
    scene, pose = preprocess(scene, pose)
    modeldir = Path("assets") / modelname
    urdf_file = str(modeldir / "model.urdf")

    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.scale = scale
    loader.fix_root_link = fix_root_link
    loader.load_multiple_collisions_from_file = False
    
    result = loader.load_multiple(urdf_file)
    obj = result[1][0]

    # --- 视觉对齐：修复模型离心问题 ---
    mins, maxs = np.array([np.inf]*3), np.array([-np.inf]*3)
    has_visuals = False
    
    # 获取所有的渲染/组件位置
    # 在 SAPIEN 中，Entity 和 Articulation 都有 get_components
    for comp in obj.get_components():
        # 我们寻找渲染相关的组件来确定视觉中心
        if "RenderBodyComponent" in str(type(comp)) or "RenderComponent" in str(type(comp)):
            # 获取组件相对于物体的局部位移
            # 如果加载出来后物体乱飞，说明这里面的 local_pose.p 有巨大的数值
            try:
                p = comp.get_local_pose().p
                mins = np.minimum(mins, p)
                maxs = np.maximum(maxs, p)
                has_visuals = True
            except:
                continue

    if has_visuals and not np.any(np.isinf(mins)):
        # 计算偏移：只修正 X, Y 轴，Z 轴维持原样避免陷入桌面
        visual_center_offset = (mins + maxs) / 2.0
        visual_center_offset[2] = 0 
        
        # 修正 Pose
        corrected_p = pose.p - visual_center_offset
        pose = sapien.Pose(corrected_p, pose.q)

    # 应用 Pose
    if hasattr(obj, "set_root_pose"):
        obj.set_root_pose(pose)
    else:
        obj.set_pose(pose)

    from .actor_utils import ArticulationActor, Actor
    # 返回正确的包装类
    if "Articulation" in str(type(obj)):
        return ArticulationActor(obj, None)
    else:
        return Actor(obj, None)