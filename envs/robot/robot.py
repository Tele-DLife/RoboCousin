import sapien.core as sapien
import numpy as np
import pdb
from .planner import MplibPlanner
import numpy as np
import toppra as ta
import math
import yaml
import os
import transforms3d as t3d
from copy import deepcopy
import sapien.core as sapien
import envs._GLOBAL_CONFIGS as CONFIGS
from envs.utils import transforms
from .planner import CuroboPlanner
import torch.multiprocessing as mp


class Robot:

    def __init__(self, scene, need_topp=False, **kwargs):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        self._init_robot_(scene, need_topp, **kwargs)

    def _init_robot_(self, scene, need_topp=False, **kwargs):
        # self.dual_arm = dual_arm_tag
        # self.plan_success = True

        self.left_js = None
        self.right_js = None

        left_embodiment_args = kwargs["left_embodiment_config"]
        right_embodiment_args = kwargs["right_embodiment_config"]
        left_robot_file = kwargs["left_robot_file"]
        right_robot_file = kwargs["right_robot_file"]

        self.need_topp = need_topp

        self.left_urdf_path = os.path.join(left_robot_file, left_embodiment_args["urdf_path"])
        self.left_srdf_path = left_embodiment_args.get("srdf_path", None)
        self.left_curobo_yml_path = os.path.join(left_robot_file, "curobo.yml")
        if self.left_srdf_path is not None:
            self.left_srdf_path = os.path.join(left_robot_file, self.left_srdf_path)
        self.left_joint_stiffness = left_embodiment_args.get("joint_stiffness", 1000)
        self.left_joint_damping = left_embodiment_args.get("joint_damping", 200)
        self.left_gripper_stiffness = left_embodiment_args.get("gripper_stiffness", 1000)
        self.left_gripper_damping = left_embodiment_args.get("gripper_damping", 200)
        self.left_planner_type = left_embodiment_args.get("planner", "mplib_RRT")
        self.left_move_group = left_embodiment_args["move_group"][0]
        self.left_ee_name = left_embodiment_args["ee_joints"][0]
        self.left_arm_joints_name = left_embodiment_args["arm_joints_name"][0]
        self.left_gripper_name = left_embodiment_args["gripper_name"][0]
        self.left_gripper_bias = left_embodiment_args["gripper_bias"]
        self.left_gripper_scale = left_embodiment_args["gripper_scale"]
        self.left_homestate = left_embodiment_args.get("homestate", [[0] * len(self.left_arm_joints_name)])[0]
        self.left_fix_gripper_name = left_embodiment_args.get("fix_gripper_name", [])
        self.left_delta_matrix = np.array(left_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.left_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        self.left_global_trans_matrix = np.array(
            left_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _lift = left_embodiment_args.get("lift_joint_init_m", None)
        if _lift is None:
            _lift = right_embodiment_args.get("lift_joint_init_m", None)
        self._lift_joint_init_m = None if _lift is None else float(_lift)

        _entity_origion_pose = left_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])[0]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.left_entity_origion_pose = deepcopy(_entity_origion_pose)

        self.right_urdf_path = os.path.join(right_robot_file, right_embodiment_args["urdf_path"])
        self.right_srdf_path = right_embodiment_args.get("srdf_path", None)
        if self.right_srdf_path is not None:
            self.right_srdf_path = os.path.join(right_robot_file, self.right_srdf_path)
        self.right_curobo_yml_path = os.path.join(right_robot_file, "curobo.yml")
        self.right_joint_stiffness = right_embodiment_args.get("joint_stiffness", 1000)
        self.right_joint_damping = right_embodiment_args.get("joint_damping", 200)
        self.right_gripper_stiffness = right_embodiment_args.get("gripper_stiffness", 1000)
        self.right_gripper_damping = right_embodiment_args.get("gripper_damping", 200)
        self.right_planner_type = right_embodiment_args.get("planner", "mplib_RRT")
        self.right_move_group = right_embodiment_args["move_group"][1]
        self.right_ee_name = right_embodiment_args["ee_joints"][1]
        self.right_arm_joints_name = right_embodiment_args["arm_joints_name"][1]
        self.right_gripper_name = right_embodiment_args["gripper_name"][1]
        self.right_gripper_bias = right_embodiment_args["gripper_bias"]
        self.right_gripper_scale = right_embodiment_args["gripper_scale"]
        self.right_homestate = right_embodiment_args.get("homestate", [[1] * len(self.right_arm_joints_name)])[1]
        self.right_fix_gripper_name = right_embodiment_args.get("fix_gripper_name", [])
        self.right_delta_matrix = np.array(right_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.right_inv_delta_matrix = np.linalg.inv(self.right_delta_matrix)
        self.right_global_trans_matrix = np.array(
            right_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = right_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])
        _entity_origion_pose = _entity_origion_pose[0 if len(_entity_origion_pose) == 1 else 1]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.right_entity_origion_pose = deepcopy(_entity_origion_pose)
        self.is_dual_arm = kwargs["dual_arm_embodied"]

        self.left_rotate_lim = left_embodiment_args.get("rotate_lim", [0, 0])
        self.right_rotate_lim = right_embodiment_args.get("rotate_lim", [0, 0])

        self.left_perfect_direction = left_embodiment_args.get("grasp_perfect_direction",
                                                               ["front_right", "front_left"])[0]
        self.right_perfect_direction = right_embodiment_args.get("grasp_perfect_direction",
                                                                 ["front_right", "front_left"])[1]

        # Hex / mobile_ARX (lift chassis) flag + EE frame mapping (configured ee_joints vs planner move_group link)
        _lp = str(self.left_urdf_path).lower()
        _rp = str(self.right_urdf_path).lower()
        self.left_is_hex = ("hex" in _lp) or ("mobile_arx" in _lp)
        self.right_is_hex = ("hex" in _rp) or ("mobile_arx" in _rp)
        # Optional drive force limits: ONLY read/apply for these embodiments so other arms stay unchanged.
        self.left_joint_force_limit = left_embodiment_args.get("joint_force_limit", None) if self.left_is_hex else None
        self.left_gripper_force_limit = left_embodiment_args.get("gripper_force_limit", None) if self.left_is_hex else None
        self.right_joint_force_limit = right_embodiment_args.get("joint_force_limit", None) if self.right_is_hex else None
        self.right_gripper_force_limit = right_embodiment_args.get("gripper_force_limit", None) if self.right_is_hex else None
        # mobile_ARX lift should stay locked during arm motion; allow dedicated drive gains/force.
        self.left_lift_joint_stiffness = (
            left_embodiment_args.get("lift_joint_stiffness", self.left_joint_stiffness) if self.left_is_hex else None
        )
        self.left_lift_joint_damping = (
            left_embodiment_args.get("lift_joint_damping", self.left_joint_damping) if self.left_is_hex else None
        )
        self.left_lift_joint_force_limit = (
            left_embodiment_args.get("lift_joint_force_limit", self.left_joint_force_limit) if self.left_is_hex else None
        )
        self.right_lift_joint_stiffness = (
            right_embodiment_args.get("lift_joint_stiffness", self.right_joint_stiffness) if self.right_is_hex else None
        )
        self.right_lift_joint_damping = (
            right_embodiment_args.get("lift_joint_damping", self.right_joint_damping) if self.right_is_hex else None
        )
        self.right_lift_joint_force_limit = (
            right_embodiment_args.get("lift_joint_force_limit", self.right_joint_force_limit) if self.right_is_hex else None
        )
        self.hold_lift_during_control = bool(
            left_embodiment_args.get(
                "hold_lift_during_control",
                right_embodiment_args.get("hold_lift_during_control", True),
            )
        )
        self.left_ee_to_move_group_rot = np.eye(3, dtype=np.float64)
        self.right_ee_to_move_group_rot = np.eye(3, dtype=np.float64)
        if (
            self.left_is_hex
            and
            "aloha_link6" in str(self.left_move_group).lower()
            and str(self.left_ee_name).endswith("_joint_6")
        ):
            self.left_ee_to_move_group_rot = t3d.axangles.axangle2mat([0.0, 1.0, 0.0], -np.pi / 2.0)
        if (
            self.right_is_hex
            and
            "aloha_link6" in str(self.right_move_group).lower()
            and str(self.right_ee_name).endswith("_joint_6")
        ):
            self.right_ee_to_move_group_rot = t3d.axangles.axangle2mat([0.0, 1.0, 0.0], -np.pi / 2.0)

        if self.is_dual_arm:
            loader: sapien.URDFLoader = scene.create_urdf_loader()
            loader.fix_root_link = True
            self._entity = loader.load(self.left_urdf_path)
            self.left_entity = self._entity
            self.right_entity = self._entity
        else:
            arms_dis = kwargs["embodiment_dis"]
            self.left_entity_origion_pose.p += [-arms_dis / 2, 0, 0]
            self.right_entity_origion_pose.p += [arms_dis / 2, 0, 0]
            left_loader: sapien.URDFLoader = scene.create_urdf_loader()
            left_loader.fix_root_link = True
            right_loader: sapien.URDFLoader = scene.create_urdf_loader()
            right_loader.fix_root_link = True
            self.left_entity = left_loader.load(self.left_urdf_path)
            self.right_entity = right_loader.load(self.right_urdf_path)

        self.left_entity.set_root_pose(self.left_entity_origion_pose)
        self.right_entity.set_root_pose(self.right_entity_origion_pose)

    def reset(self, scene, need_topp=False, **kwargs):
        self._init_robot_(scene, need_topp, **kwargs)
        # init_joints before planner: lift qpos / FK must match before MotionGen builds its world table cuboid.
        self.init_joints()
        if self.communication_flag:
            if hasattr(self, "left_conn") and self.left_conn:
                self.left_conn.send({"cmd": "reset"})
                _ = self.left_conn.recv()
            if hasattr(self, "right_conn") and self.right_conn:
                self.right_conn.send({"cmd": "reset"})
                _ = self.right_conn.recv()
        else:
            # Check if planners exist, if not initialize them
            if not hasattr(self, "left_planner") or not hasattr(self, "right_planner"):
                self.set_planner(scene=scene)
            elif not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner):
                self.set_planner(scene=scene)

    def get_grasp_perfect_direction(self, arm_tag):
        if arm_tag == "left":
            return self.left_perfect_direction
        elif arm_tag == "right":
            return self.right_perfect_direction

    def create_target_pose_list(self, origin_pose, center_pose, arm_tag=None):
        res_lst = []
        rotate_lim = (self.left_rotate_lim if arm_tag == "left" else self.right_rotate_lim)
        rotate_step = (rotate_lim[1] - rotate_lim[0]) / CONFIGS.ROTATE_NUM
        for i in range(CONFIGS.ROTATE_NUM):
            now_pose = transforms.rotate_along_axis(
                origin_pose,
                center_pose,
                [0, 1, 0],
                rotate_step * i + rotate_lim[0],
                axis_type="target",
                towards=[0, -1, 0],
            )
            res_lst.append(now_pose)
        return res_lst

    def get_constraint_pose(self, ori_vec: list, arm_tag=None):
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        return ori_vec[:3] + (ori_vec[-3:] @ np.linalg.inv(inv_delta_matrix)).tolist()

    def init_joints(self):
        if self.left_entity is None or self.right_entity is None:
            raise ValueError("Robote entity is None")

        self.left_active_joints = self.left_entity.get_active_joints()
        self.right_active_joints = self.right_entity.get_active_joints()

        self.left_ee = self.left_entity.find_joint_by_name(self.left_ee_name)
        self.right_ee = self.right_entity.find_joint_by_name(self.right_ee_name)
        self.left_ee_link = self.left_entity.find_link_by_name(self.left_move_group)
        self.right_ee_link = self.right_entity.find_link_by_name(self.right_move_group)

        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0

        self.left_arm_joints = [self.left_entity.find_joint_by_name(i) for i in self.left_arm_joints_name]
        self.right_arm_joints = [self.right_entity.find_joint_by_name(i) for i in self.right_arm_joints_name]

        def get_gripper_joints(find, gripper_name: dict):
            gripper = [(find(gripper_name["base"]), 1.0, 0.0)]
            for g in gripper_name["mimic"]:
                gripper.append((find(g[0]), g[1], g[2]))
            return gripper

        self.left_gripper = get_gripper_joints(self.left_entity.find_joint_by_name, self.left_gripper_name)
        self.right_gripper = get_gripper_joints(self.right_entity.find_joint_by_name, self.right_gripper_name)
        left_gripper_joint_set = {g[0] for g in self.left_gripper if g and g[0] is not None}
        right_gripper_joint_set = {g[0] for g in self.right_gripper if g and g[0] is not None}
        self.gripper_name = deepcopy(self.left_fix_gripper_name) + deepcopy(self.right_fix_gripper_name)

        for g in self.left_gripper:
            self.gripper_name.append(g[0].child_link.get_name())
        for g in self.right_gripper:
            self.gripper_name.append(g[0].child_link.get_name())

        # camera link id
        self.left_camera = self.left_entity.find_link_by_name("left_camera")
        if self.left_camera is None:
            self.left_camera = self.left_entity.find_link_by_name("camera")
            if self.left_camera is None:
                print("No left camera link")
                self.left_camera = self.left_entity.get_links()[0]

        self.right_camera = self.right_entity.find_link_by_name("right_camera")
        if self.right_camera is None:
            self.right_camera = self.right_entity.find_link_by_name("camera")
            if self.right_camera is None:
                print("No right camera link")
                self.right_camera = self.right_entity.get_links()[0]

        for i, joint in enumerate(self.left_active_joints):
            if joint not in left_gripper_joint_set:
                j_name = joint.get_name()
                is_hex_lift = self.left_is_hex and j_name == "lift_joint_1"
                stiffness = self.left_lift_joint_stiffness if is_hex_lift else self.left_joint_stiffness
                damping = self.left_lift_joint_damping if is_hex_lift else self.left_joint_damping
                force_limit = None
                if self.left_is_hex:
                    if is_hex_lift and self.left_lift_joint_force_limit is not None:
                        force_limit = float(self.left_lift_joint_force_limit)
                    elif self.left_joint_force_limit is not None:
                        force_limit = float(self.left_joint_force_limit)
                try:
                    if force_limit is not None:
                        joint.set_drive_property(
                            stiffness=stiffness,
                            damping=damping,
                            force_limit=force_limit,
                        )
                    else:
                        joint.set_drive_property(stiffness=stiffness, damping=damping)
                except TypeError:
                    joint.set_drive_property(stiffness=stiffness, damping=damping)
        for i, joint in enumerate(self.right_active_joints):
            if joint not in right_gripper_joint_set:
                j_name = joint.get_name()
                is_hex_lift = self.right_is_hex and j_name == "lift_joint_1"
                stiffness = self.right_lift_joint_stiffness if is_hex_lift else self.right_joint_stiffness
                damping = self.right_lift_joint_damping if is_hex_lift else self.right_joint_damping
                force_limit = None
                if self.right_is_hex:
                    if is_hex_lift and self.right_lift_joint_force_limit is not None:
                        force_limit = float(self.right_lift_joint_force_limit)
                    elif self.right_joint_force_limit is not None:
                        force_limit = float(self.right_joint_force_limit)
                try:
                    if force_limit is not None:
                        joint.set_drive_property(
                            stiffness=stiffness,
                            damping=damping,
                            force_limit=force_limit,
                        )
                    else:
                        joint.set_drive_property(stiffness=stiffness, damping=damping)
                except TypeError:
                    joint.set_drive_property(stiffness=stiffness, damping=damping)

        for joint in self.left_gripper:
            if self.left_is_hex and self.left_gripper_force_limit is not None:
                try:
                    joint[0].set_drive_property(
                        stiffness=self.left_gripper_stiffness,
                        damping=self.left_gripper_damping,
                        force_limit=float(self.left_gripper_force_limit),
                    )
                except TypeError:
                    joint[0].set_drive_property(stiffness=self.left_gripper_stiffness, damping=self.left_gripper_damping)
            else:
                joint[0].set_drive_property(stiffness=self.left_gripper_stiffness, damping=self.left_gripper_damping)
        for joint in self.right_gripper:
            if self.right_is_hex and self.right_gripper_force_limit is not None:
                try:
                    joint[0].set_drive_property(
                        stiffness=self.right_gripper_stiffness,
                        damping=self.right_gripper_damping,
                        force_limit=float(self.right_gripper_force_limit),
                    )
                except TypeError:
                    joint[0].set_drive_property(
                        stiffness=self.right_gripper_stiffness,
                        damping=self.right_gripper_damping,
                    )
            else:
                joint[0].set_drive_property(
                    stiffness=self.right_gripper_stiffness,
                    damping=self.right_gripper_damping,
                )

        # Apply optional lift joint initial pose if configured.
        self._apply_lift_joint_init_pose(self.left_entity)
        if self.right_entity is not self.left_entity:
            self._apply_lift_joint_init_pose(self.right_entity)

        self._log_mobile_arx_lift_state("after_init_joints")

    def _embodiment_tcp_debug_label(self) -> str:
        p = (getattr(self, "left_urdf_path", "") or "").replace("\\", "/")
        low = p.lower()
        if CONFIGS.is_mobile_arx_embodiment_urdf(p):
            return "mobile_ARX"
        if "arx-x5" in low or "arx_x5" in low or "x5a.urdf" in low:
            return "ARX-X5"
        base = os.path.basename(os.path.dirname(p))
        return base or "unknown"

    def _tcp_height_debug_should_log(self) -> bool:
        v = os.environ.get("ROBOTWIN_DEBUG_TCP_HEIGHT", "0").strip().lower()
        if v not in ("1", "true", "yes"):
            return False
        lp = getattr(self, "left_urdf_path", "") or ""
        low = str(lp).lower()
        return (
            CONFIGS.is_mobile_arx_embodiment_urdf(lp)
            or "arx-x5" in low
            or "arx_x5" in low
            or "x5a.urdf" in low
        )

    def _merged_qpos_patch_plan_row(self, entity, plan_row, arm_joint_names):
        """Merge planner last row (subset DoF) into full entity qpos; lift/other joints unchanged."""
        if entity is None or plan_row is None or not arm_joint_names:
            return None
        full = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
        row = np.asarray(plan_row, dtype=np.float64).reshape(-1)
        if row.size == full.size:
            return row.copy()
        names = [j.get_name() for j in entity.get_active_joints()]
        try:
            idxs = [names.index(n) for n in arm_joint_names if n in names]
        except ValueError:
            return None
        if len(idxs) != row.size:
            return None
        full[idxs] = row
        return full

    def _log_tcp_world_heights(self, phase: str, plan_last_row=None, plan_arm_tag=None):
        """World-frame TCP after trajectory is computed (caller), not at homestate.

        Prints (1) current sim FK from drive/qpos, (2) optional FK after patching only the planned arm's
        joints from the trajectory last row into full qpos (lift etc. unchanged) — closer to rendered pose
        at path end than homestate-only timing.

        Opt-in: set ROBOTWIN_DEBUG_TCP_HEIGHT=1 (only for mobile_ARX / ARX-X5 style URDFs).
        """
        if not self._tcp_height_debug_should_log():
            return
        tag = self._embodiment_tcp_debug_label()
        try:
            lt = self.get_left_tcp_pose()
            lx, ly, lz = float(lt[0]), float(lt[1]), float(lt[2])
        except Exception as e:
            lx = ly = lz = float("nan")
            lt_err = f" err={e!r}"
        else:
            lt_err = ""
        try:
            rt = self.get_right_tcp_pose()
            rx, ry, rz = float(rt[0]), float(rt[1]), float(rt[2])
        except Exception as e:
            rx = ry = rz = float("nan")
            rt_err = f" err={e!r}"
        else:
            rt_err = ""

        def _link_z(link):
            if link is None:
                return None
            try:
                return float(link.get_pose().p[2])
            except Exception:
                return None

        z_ll = _link_z(getattr(self, "left_ee_link", None))
        z_rl = _link_z(getattr(self, "right_ee_link", None))
        mg_l = getattr(self, "left_move_group", "?")
        mg_r = getattr(self, "right_move_group", "?")
        zl_s = f" {mg_l}_world_z={z_ll:.4f}" if z_ll is not None else ""
        zr_s = f" {mg_r}_world_z={z_rl:.4f}" if z_rl is not None else ""

        fk_extra = ""
        if plan_last_row is not None and plan_arm_tag in ("left", "right"):
            ent = self.left_entity if plan_arm_tag == "left" else self.right_entity
            jnames = self.left_arm_joints_name if plan_arm_tag == "left" else self.right_arm_joints_name
            merged = self._merged_qpos_patch_plan_row(ent, plan_last_row, jnames)
            if merged is not None:
                try:
                    prev = np.asarray(ent.get_qpos(), dtype=np.float64).copy()
                    ent.set_qpos(merged)
                    l2 = self.get_left_tcp_pose()
                    r2 = self.get_right_tcp_pose()
                    ent.set_qpos(prev)
                    fk_extra = (
                        f" | fk_from_plan_last_row left_z={float(l2[2]):.4f} right_z={float(r2[2]):.4f}"
                        f" (debug: merged last row for {plan_arm_tag} arm only; other arm still sim qpos — asymmetry expected)"
                    )
                except Exception as e:
                    fk_extra = f" | fk_from_plan_last_row_err={e!r}"

        print(
            f"[{tag}][tcp][{phase}] sim_left_TCP_xyz=({lx:.4f},{ly:.4f},{lz:.4f}) z={lz:.4f}{lt_err}"
            f" | sim_right_TCP_xyz=({rx:.4f},{ry:.4f},{rz:.4f}) z={rz:.4f}{rt_err}"
            f"{zl_s}{zr_s}{fk_extra}",
            flush=True,
        )

    def _log_tcp_after_plan_dict(self, arm_tag: str, kind: str, result):
        if not isinstance(result, dict) or result.get("status") != "Success":
            return
        pos = result.get("position")
        if pos is None:
            self._log_tcp_world_heights(f"after_{arm_tag}_{kind}_success_no_position")
            return
        try:
            last = pos[-1]
        except Exception:
            self._log_tcp_world_heights(f"after_{arm_tag}_{kind}_success_bad_position")
            return
        self._log_tcp_world_heights(f"after_{arm_tag}_{kind}_success", plan_last_row=last, plan_arm_tag=arm_tag)

    def _log_tcp_after_plan_batch_dict(self, arm_tag: str, result):
        if not isinstance(result, dict):
            return
        status = result.get("status")
        pos = result.get("position")
        if pos is None or status is None:
            return
        try:
            n = len(status)
        except Exception:
            return
        pick = None
        for i in range(n):
            try:
                si = status[i]
            except Exception:
                continue
            if str(si) != "Success":
                continue
            try:
                traj = pos[i]
                row = traj[-1]
            except Exception:
                continue
            pick = (i, row)
            break
        if pick is None:
            return
        i, row = pick
        self._log_tcp_world_heights(
            f"after_{arm_tag}_plan_batch_success[i={i}]",
            plan_last_row=row,
            plan_arm_tag=arm_tag,
        )

    def _log_mobile_arx_lift_state(self, phase: str):
        """Debug: mobile_ARX lift stroke. Opt-in: ROBOTWIN_DEBUG_MOBILE_ARX_LIFT=1."""
        v1 = os.environ.get("ROBOTWIN_DEBUG_MOBILE_ARX_LIFT", "0").strip().lower()
        if v1 not in ("1", "true", "yes"):
            return
        if not CONFIGS.is_mobile_arx_embodiment_urdf(getattr(self, "left_urdf_path", "")):
            return
        ent = self.left_entity
        idx, joint, _ = self._lift_joint_index(ent)
        q_cfg = getattr(self, "_lift_joint_init_m", None)
        if idx is None or joint is None:
            print(f"[mobile_ARX][lift][{phase}] lift_joint_1: not found among active joints", flush=True)
            return
        full = np.asarray(ent.get_qpos(), dtype=np.float64)
        qpos = float(full[idx]) if full.size > idx else float("nan")
        try:
            drive_target = float(joint.get_drive_target()[0])
        except Exception:
            drive_target = float("nan")
        z_fl = None
        try:
            lk = ent.find_link_by_name("fl_base_link")
            if lk is not None:
                z_fl = float(lk.get_pose().p[2])
        except Exception:
            pass
        z_msg = f" fl_base_link_world_z={z_fl:.4f}" if z_fl is not None else ""
        print(
            f"[mobile_ARX][lift][{phase}] lift_joint_1 qpos={qpos:.4f} drive_target={drive_target:.4f} "
            f"lift_joint_init_m_cfg={q_cfg}{z_msg}",
            flush=True,
        )

    def _lift_joint_index(self, entity):
        name = "lift_joint_1"
        active = entity.get_active_joints()
        for i, joint in enumerate(active):
            if joint.get_name() == name:
                return i, joint, active
        return None, None, active

    def _apply_lift_joint_init_pose(self, entity):
        q0 = getattr(self, "_lift_joint_init_m", None)
        if q0 is None:
            return
        idx, joint, _ = self._lift_joint_index(entity)
        if idx is None or joint is None:
            return
        q = max(0.0, min(0.62, float(q0)))
        full = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
        if full.size <= idx:
            return
        full[idx] = q
        entity.set_qpos(full)
        try:
            joint.set_drive_target(q)
        except Exception:
            pass

    def _sync_lift_joint_drive_target(self, entity):
        q0 = getattr(self, "_lift_joint_init_m", None)
        if q0 is None:
            return
        idx, joint, _ = self._lift_joint_index(entity)
        if idx is None or joint is None:
            return
        q = max(0.0, min(0.62, float(q0)))
        try:
            joint.set_drive_target(q)
            joint.set_drive_velocity_target(0.0)
        except Exception:
            pass

    def _hold_lift_joint(self):
        if not self.hold_lift_during_control:
            return
        self._sync_lift_joint_drive_target(self.left_entity)
        if self.right_entity is not self.left_entity:
            self._sync_lift_joint_drive_target(self.right_entity)

    def move_to_homestate(self):
        for i, joint in enumerate(self.left_arm_joints):
            joint.set_drive_target(self.left_homestate[i])

        for i, joint in enumerate(self.right_arm_joints):
            joint.set_drive_target(self.right_homestate[i])

        self._sync_lift_joint_drive_target(self.left_entity)
        if self.right_entity is not self.left_entity:
            self._sync_lift_joint_drive_target(self.right_entity)

        self._log_mobile_arx_lift_state("after_move_to_homestate")

    def set_origin_endpose(self):
        self.left_original_pose = self.get_left_ee_pose()
        self.right_original_pose = self.get_right_ee_pose()

    def print_info(self):
        print(
            "active joints: ",
            [joint.get_name() for joint in self.left_active_joints + self.right_active_joints],
        )
        print(
            "all links: ",
            [link.get_name() for link in self.left_entity.get_links() + self.right_entity.get_links()],
        )
        print("left arm joints: ", [joint.get_name() for joint in self.left_arm_joints])
        print("right arm joints: ", [joint.get_name() for joint in self.right_arm_joints])
        print("left gripper: ", [joint[0].get_name() for joint in self.left_gripper])
        print("right gripper: ", [joint[0].get_name() for joint in self.right_gripper])
        print("left ee: ", self.left_ee.get_name())
        print("right ee: ", self.right_ee.get_name())

    def set_planner(self, scene=None):
        abs_left_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.left_curobo_yml_path)
        abs_right_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.right_curobo_yml_path)

        self.communication_flag = (abs_left_curobo_yml_path != abs_right_curobo_yml_path)

        if self.is_dual_arm:
            # For dual-arm robots, try to use left/right specific configs
            abs_left_curobo_yml_path_alt = abs_left_curobo_yml_path.replace("curobo.yml", "curobo_left.yml")
            abs_right_curobo_yml_path_alt = abs_right_curobo_yml_path.replace("curobo.yml", "curobo_right.yml")
            # Use alternative paths if they exist, otherwise use default
            if os.path.exists(abs_left_curobo_yml_path_alt):
                abs_left_curobo_yml_path = abs_left_curobo_yml_path_alt
            if os.path.exists(abs_right_curobo_yml_path_alt):
                abs_right_curobo_yml_path = abs_right_curobo_yml_path_alt
        else:
            # For single-arm robots, use the same curobo.yml for both planners
            # Prefer right config if it exists, otherwise use left
            if os.path.exists(abs_right_curobo_yml_path):
                abs_left_curobo_yml_path = abs_right_curobo_yml_path
            elif os.path.exists(abs_left_curobo_yml_path):
                abs_right_curobo_yml_path = abs_left_curobo_yml_path
            else:
                # If neither exists, try to find any curobo.yml in the directory
                left_dir = os.path.dirname(abs_left_curobo_yml_path)
                right_dir = os.path.dirname(abs_right_curobo_yml_path)
                for dir_path in [right_dir, left_dir]:
                    if dir_path and os.path.exists(dir_path):
                        default_yml = os.path.join(dir_path, "curobo.yml")
                        if os.path.exists(default_yml):
                            abs_left_curobo_yml_path = default_yml
                            abs_right_curobo_yml_path = default_yml
                            break

        # cuRobo MotionGen world cuboid uses ``0.74 - ref_z``; mobile_ARX chassis root z≈0. Use **average** of
        # ``fl_base_link`` / ``fr_base_link`` world z so left and right MotionGen instances see the same table
        # height (previously only ``fl_base_link`` was used for both planners).
        world_table_ref_z = None
        if CONFIGS.is_mobile_arx_embodiment_urdf(getattr(self, "left_urdf_path", "")):
            try:
                zl = None
                zr = None
                lk = self.left_entity.find_link_by_name("fl_base_link")
                if lk is not None:
                    zl = float(lk.get_pose().p[2])
                rk = self.left_entity.find_link_by_name("fr_base_link")
                if rk is not None:
                    zr = float(rk.get_pose().p[2])
                if zl is not None and zr is not None:
                    world_table_ref_z = 0.5 * (zl + zr)
                elif zl is not None:
                    world_table_ref_z = zl
                elif zr is not None:
                    world_table_ref_z = zr
            except Exception:
                world_table_ref_z = None

        if not self.communication_flag:
            # In non-communication mode, initialize planners in-process.
            if self.is_dual_arm:
                self.left_planner = CuroboPlanner(
                    self.left_entity_origion_pose,
                    self.left_arm_joints_name,
                    [joint.get_name() for joint in self.left_entity.get_active_joints()],
                    yml_path=abs_left_curobo_yml_path,
                    world_table_ref_z=world_table_ref_z,
                )
                self.right_planner = CuroboPlanner(
                    self.right_entity_origion_pose,
                    self.right_arm_joints_name,
                    [joint.get_name() for joint in self.right_entity.get_active_joints()],
                    yml_path=abs_right_curobo_yml_path,
                    world_table_ref_z=world_table_ref_z,
                )
            else:
                # Two-single-arm setup: each planner MUST bind to its own entity/origin,
                # otherwise left-arm planning will be mirrored/offset incorrectly.
                self.right_planner = CuroboPlanner(
                    self.right_entity_origion_pose,
                    self.right_arm_joints_name,
                    [joint.get_name() for joint in self.right_entity.get_active_joints()],
                    yml_path=abs_right_curobo_yml_path,
                    world_table_ref_z=world_table_ref_z,
                )
                self.left_planner = CuroboPlanner(
                    self.left_entity_origion_pose,
                    self.left_arm_joints_name,
                    [joint.get_name() for joint in self.left_entity.get_active_joints()],
                    yml_path=abs_left_curobo_yml_path,
                    world_table_ref_z=world_table_ref_z,
                )
        else:
            self.left_conn, left_child_conn = mp.Pipe()
            self.right_conn, right_child_conn = mp.Pipe()

            left_args = {
                "origin_pose": self.left_entity_origion_pose,
                "joints_name": self.left_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.left_entity.get_active_joints()],
                "yml_path": abs_left_curobo_yml_path,
                "world_table_ref_z": world_table_ref_z,
            }

            right_args = {
                "origin_pose": self.right_entity_origion_pose,
                "joints_name": self.right_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.right_entity.get_active_joints()],
                "yml_path": abs_right_curobo_yml_path,
                "world_table_ref_z": world_table_ref_z,
            }

            self.left_proc = mp.Process(target=planner_process_worker, args=(left_child_conn, left_args))
            self.right_proc = mp.Process(target=planner_process_worker, args=(right_child_conn, right_args))

            self.left_proc.daemon = True
            self.right_proc.daemon = True

            self.left_proc.start()
            self.right_proc.start()

        if self.need_topp:
            self.left_mplib_planner = MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    def update_world_pcd(self, world_pcd):
        try:
            self.left_planner.update_point_cloud(world_pcd, resolution=0.02)
            self.right_planner.update_point_cloud(world_pcd, resolution=0.02)
        except:
            print("Update world pointcloud wrong!")

    def _trans_from_gripper_to_endlink(self, target_pose, arm_tag=None):
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        target_pose_arr = np.array(target_pose)
        gripper_pose_pos, gripper_pose_quat = deepcopy(target_pose_arr[0:3]), deepcopy(target_pose_arr[-4:])
        gripper_pose_mat = t3d.quaternions.quat2mat(gripper_pose_quat)
        gripper_pose_pos += gripper_pose_mat @ np.array([0.12 - gripper_bias, 0, 0]).T
        gripper_pose_mat = gripper_pose_mat @ inv_delta_matrix
        gripper_pose_quat = t3d.quaternions.mat2quat(gripper_pose_mat)
        return sapien.Pose(gripper_pose_pos, gripper_pose_quat)

    @staticmethod
    def _quat_geodesic_deg(q0, q1):
        a = np.asarray(q0, dtype=np.float64).reshape(4)
        b = np.asarray(q1, dtype=np.float64).reshape(4)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-15 or nb < 1e-15:
            return float("nan")
        a = a / na
        b = b / nb
        d = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(d)))

    def debug_endlink_roundtrip_error(self, arm_tag="left"):
        is_hex = self.left_is_hex if arm_tag == "left" else self.right_is_hex
        if not is_hex:
            return None
        if arm_tag == "left":
            ee_pose = self.get_left_ee_pose()
            link_pose = self.left_ee_link.get_pose() if self.left_ee_link is not None else self.left_ee.global_pose
        else:
            ee_pose = self.get_right_ee_pose()
            link_pose = self.right_ee_link.get_pose() if self.right_ee_link is not None else self.right_ee.global_pose
        pred = self._trans_from_gripper_to_endlink(ee_pose, arm_tag=arm_tag)
        p_err = float(np.linalg.norm(np.asarray(pred.p, dtype=np.float64) - np.asarray(link_pose.p, dtype=np.float64)))
        q_err = self._quat_geodesic_deg(np.asarray(pred.q, dtype=np.float64), np.asarray(link_pose.q, dtype=np.float64))
        return {"arm": arm_tag, "pos_err_m": p_err, "quat_err_deg": q_err}

    def _world_base_pose7_for_curobo(self, arm_tag: str):
        # mobile_ARX: arm base moves with lift; pass current fl/fr_base_link world pose so world->planner-base matches sim.
        if not CONFIGS.is_mobile_arx_embodiment_urdf(self.left_urdf_path):
            return None
        entity = self.left_entity if arm_tag == "left" else self.right_entity
        name = "fl_base_link" if arm_tag == "left" else "fr_base_link"
        lk = entity.find_link_by_name(name)
        if lk is None:
            return None
        pose = lk.get_pose()
        return np.concatenate([np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)])

    def left_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.left_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_grippers(now_val, target_val)

    def right_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.right_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_grippers(now_val, target_val)

    def left_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="left")

        wb = self._world_base_pose7_for_curobo("left")
        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
                "world_base_pose_wxyz": wb,
            })
            result = self.left_conn.recv()
        else:
            result = self.left_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="left",
                world_base_pose_wxyz=wb,
            )
        self._log_tcp_after_plan_batch_dict("left", result)
        return result

    def right_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="right")

        wb = self._world_base_pose7_for_curobo("right")
        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
                "world_base_pose_wxyz": wb,
            })
            result = self.right_conn.recv()
        else:
            result = self.right_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="right",
                world_base_pose_wxyz=wb,
            )
        self._log_tcp_after_plan_batch_dict("right", result)
        return result

    def left_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="left")

        wb = self._world_base_pose7_for_curobo("left")
        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
                "world_base_pose_wxyz": wb,
            })
            result = self.left_conn.recv()
        else:
            result = self.left_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="left",
                world_base_pose_wxyz=wb,
            )
        self._log_tcp_after_plan_dict("left", "plan_path", result)
        return result

    def right_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="right")

        wb = self._world_base_pose7_for_curobo("right")
        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
                "world_base_pose_wxyz": wb,
            })
            result = self.right_conn.recv()
        else:
            result = self.right_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="right",
                world_base_pose_wxyz=wb,
            )
        self._log_tcp_after_plan_dict("right", "plan_path", result)
        return result

    # The data of gripper has been normalized
    def get_left_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.left_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.right_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_left_arm_real_jointState(self) -> list:
        jointState_list = []
        left_joints_qpos = self.left_entity.get_qpos()
        left_active_joints = self.left_entity.get_active_joints()
        for joint in self.left_arm_joints:
            jointState_list.append(left_joints_qpos[left_active_joints.index(joint)])
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_real_jointState(self) -> list:
        jointState_list = []
        right_joints_qpos = self.right_entity.get_qpos()
        right_active_joints = self.right_entity.get_active_joints()
        for joint in self.right_arm_joints:
            jointState_list.append(right_joints_qpos[right_active_joints.index(joint)])
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_left_gripper_val(self):
        if None in self.left_gripper:
            print("No gripper")
            return 0
        return self.left_gripper_val

    def get_right_gripper_val(self):
        if None in self.right_gripper:
            print("No gripper")
            return 0
        return self.right_gripper_val

    def is_left_gripper_open(self):
        return self.left_gripper_val > 0.8

    def is_right_gripper_open(self):
        return self.right_gripper_val > 0.8

    def is_left_gripper_open_half(self):
        return self.left_gripper_val > 0.45

    def is_right_gripper_open_half(self):
        return self.right_gripper_val > 0.45

    def is_left_gripper_close(self):
        return self.left_gripper_val < 0.2

    def is_right_gripper_close(self):
        return self.right_gripper_val < 0.2

    # get move group joint pose
    def get_left_ee_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=False)

    def get_right_ee_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=False)

    # get gripper centor pose
    def get_left_tcp_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=True)

    def get_right_tcp_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=True)

    def get_left_orig_endpose(self):
        pose = self.left_ee.global_pose
        global_trans_matrix = self.left_global_trans_matrix
        ee_to_move_group_rot = self.left_ee_to_move_group_rot
        pose.p = pose.p - self.left_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q) @ ee_to_move_group_rot
            @ global_trans_matrix).tolist())

    def get_right_orig_endpose(self):
        pose = self.right_ee.global_pose
        global_trans_matrix = self.right_global_trans_matrix
        ee_to_move_group_rot = self.right_ee_to_move_group_rot
        pose.p = pose.p - self.right_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q) @ ee_to_move_group_rot
            @ global_trans_matrix).tolist())

    def _trans_endpose(self, arm_tag=None, is_endpose=False):
        if arm_tag is None:
            print("No arm tag")
            return
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        global_trans_matrix = (self.left_global_trans_matrix if arm_tag == "left" else self.right_global_trans_matrix)
        delta_matrix = (self.left_delta_matrix if arm_tag == "left" else self.right_delta_matrix)
        ee_to_move_group_rot = (
            self.left_ee_to_move_group_rot if arm_tag == "left" else self.right_ee_to_move_group_rot
        )
        if arm_tag == "left":
            ee_pose = self.left_ee.global_pose
        else:
            ee_pose = self.right_ee.global_pose
        endpose_arr = np.eye(4)
        endpose_arr[:3, :3] = (t3d.quaternions.quat2mat(ee_pose.q) @ ee_to_move_group_rot @ global_trans_matrix @ delta_matrix)
        dis = gripper_bias
        if is_endpose == False:
            dis -= 0.12
        endpose_arr[:3, 3] = ee_pose.p + endpose_arr[:3, :3] @ np.array([dis, 0, 0]).T
        res = (endpose_arr[:3, 3].tolist() + t3d.quaternions.mat2quat(endpose_arr[:3, :3]).tolist())
        return res

    def _entity_qf(self, entity):
        qf = entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        entity.set_qf(qf)

    def set_arm_joints(self, target_position, target_velocity, arm_tag):
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)

        joint_lst = self.left_arm_joints if arm_tag == "left" else self.right_arm_joints
        for j in range(len(joint_lst)):
            joint = joint_lst[j]
            joint.set_drive_target(target_position[j])
            joint.set_drive_velocity_target(target_velocity[j])
        self._hold_lift_joint()

    def get_normal_real_gripper_val(self):
        normal_left_gripper_val = (self.left_gripper[0][0].get_drive_target()[0] - self.left_gripper_scale[0]) / (
            self.left_gripper_scale[1] - self.left_gripper_scale[0])
        normal_right_gripper_val = (self.right_gripper[0][0].get_drive_target()[0] - self.right_gripper_scale[0]) / (
            self.right_gripper_scale[1] - self.right_gripper_scale[0])
        normal_left_gripper_val = np.clip(normal_left_gripper_val, 0, 1)
        normal_right_gripper_val = np.clip(normal_right_gripper_val, 0, 1)
        return [normal_left_gripper_val, normal_right_gripper_val]

    def set_gripper(self, gripper_val, arm_tag, gripper_eps=0.1):  # gripper_val in [0,1]
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)
        gripper_val = np.clip(gripper_val, 0, 1)

        if arm_tag == "left":
            joints = self.left_gripper
            self.left_gripper_val = gripper_val
            gripper_scale = self.left_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[0]
        else:
            joints = self.right_gripper
            self.right_gripper_val = gripper_val
            gripper_scale = self.right_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[1]

        if not joints:
            print("No gripper")
            return

        if (gripper_val - real_gripper_val > gripper_eps
                and gripper_eps > 0) or (gripper_val - real_gripper_val < gripper_eps and gripper_eps < 0):
            gripper_val = real_gripper_val + gripper_eps  # TODO

        real_gripper_val = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])

        for joint in joints:
            real_joint: sapien.physx.PhysxArticulationJoint = joint[0]
            drive_target = real_gripper_val * joint[1] + joint[2]
            drive_velocity_target = (np.clip(drive_target - real_joint.drive_target, -1.0, 1.0) * 0.05)
            real_joint.set_drive_target(drive_target)
            real_joint.set_drive_velocity_target(drive_velocity_target)
        self._hold_lift_joint()


def planner_process_worker(conn, args):
    import os
    from .planner import CuroboPlanner  # 或者绝对路径导入

    planner = CuroboPlanner(
        args["origin_pose"],
        args["joints_name"],
        args["all_joints"],
        yml_path=args["yml_path"],
        world_table_ref_z=args.get("world_table_ref_z"),
    )

    while True:
        try:
            msg = conn.recv()
            if msg["cmd"] == "plan_path":
                result = planner.plan_path(
                    msg["qpos"],
                    msg["target_pose"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                    world_base_pose_wxyz=msg.get("world_base_pose_wxyz", None),
                )
                conn.send(result)

            elif msg["cmd"] == "plan_batch":
                result = planner.plan_batch(
                    msg["qpos"],
                    msg["target_pose_list"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                    world_base_pose_wxyz=msg.get("world_base_pose_wxyz", None),
                )
                conn.send(result)

            elif msg["cmd"] == "plan_grippers":
                result = planner.plan_grippers(
                    msg["now_val"],
                    msg["target_val"],
                )
                conn.send(result)

            elif msg["cmd"] == "update_point_cloud":
                planner.update_point_cloud(msg["pcd"], resolution=msg.get("resolution", 0.02))
                conn.send("ok")

            elif msg["cmd"] == "reset":
                planner.motion_gen.reset(reset_seed=True)
                conn.send("ok")

            elif msg["cmd"] == "exit":
                conn.close()
                break

            else:
                conn.send({"error": f"Unknown command {msg['cmd']}"})

        except EOFError:
            break
        except Exception as e:
            conn.send({"error": str(e)})
