from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import ROOT_PATH, is_mobile_arx_embodiment_urdf
from .utils import *
import sapien
import numpy as np
import os
import random
from pathlib import Path


class pick_up(Base_Task):

    def setup_demo(self, **kwargs):
        kwargs = dict(kwargs)
        if kwargs.get("use_cousin_coordinate") or kwargs.get("cousin_anchor_layout"):
            dr = dict(kwargs.get("domain_randomization") or {})
            dr["cluttered_table"] = True
            kwargs["domain_randomization"] = dr
        super()._init_task_env_(**kwargs)

    def get_active_arm(self) -> str:
        """Return the arm that should move for single-arm eval (matches play_once)."""
        if getattr(self, "bread", None):
            return "right" if float(self.bread[0].get_pose().p[0]) > 0 else "left"
        raise RuntimeError("[pick_up] get_active_arm() called before grasp target was spawned")

    def _sample_grasp_target_spawn_xy(self, rng=None):
        """Reuse base pick_up spawn strip logic; returns world (x, y) only."""
        table_cx, table_cy = getattr(self, "table_xy_bias", (0.0, 0.0))
        task_args = getattr(self, "task_args", {}) or {}
        is_mobile_arx = is_mobile_arx_embodiment_urdf(
            getattr(getattr(self, "robot", None), "left_urdf_path", "")
        )
        spawn_near_center_cfg = task_args.get("spawn_target_near_table_center", None)
        if spawn_near_center_cfg is None:
            spawn_near_center = is_mobile_arx
        else:
            spawn_near_center = bool(spawn_near_center_cfg) and is_mobile_arx
        center_scale = float(task_args.get("target_center_spawn_scale", 0.6))
        center_scale = float(np.clip(center_scale, 0.2, 1.0))
        if spawn_near_center:
            x_half = 0.2 * center_scale
            y_half = 0.1 * center_scale
            xlim0 = [table_cx - x_half, table_cx + x_half]
            ylim0 = [table_cy - y_half, table_cy + y_half]
        else:
            xlim0, ylim0 = [-0.2, 0.2], [-0.1, 0.1]
        xlim0, ylim0 = self._mobile_arx_noncousin_table_spawn_xy(xlim0, ylim0)
        xlim0, ylim0 = self._mobile_arx_grasp_target_reachable_xy(xlim0, ylim0, table_cx, table_cy)
        object_spawn_side = str(task_args.get("object_spawn_side", "auto")).strip().lower()
        side_margin = float(task_args.get("object_spawn_side_margin", 0.08))
        if object_spawn_side in ("left", "right"):
            if object_spawn_side == "left":
                xlim0 = [min(float(xlim0[0]), -side_margin), min(float(xlim0[1]), -side_margin)]
            else:
                xlim0 = [max(float(xlim0[0]), side_margin), max(float(xlim0[1]), side_margin)]
            if xlim0[0] > xlim0[1]:
                xlim0 = [xlim0[1], xlim0[0]]
        elif object_spawn_side not in ("", "auto", "none"):
            raise ValueError(
                f"object_spawn_side must be one of auto/left/right, got {object_spawn_side!r}"
            )
        if is_mobile_arx:
            dy = float(task_args.get("pick_up_mobile_arx_spawn_shift_y_m", -0.08))
            ylim0 = [float(ylim0[0]) + dy, float(ylim0[1]) + dy]

        if rng is not None:
            x = float(rng.uniform(float(xlim0[0]), float(xlim0[1])))
            y = float(rng.uniform(float(ylim0[0]), float(ylim0[1])))
            return x, y

        rand_pos = rand_pose(
            xlim=xlim0,
            ylim=ylim0,
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )
        return float(rand_pos.p[0]), float(rand_pos.p[1])

    def _prepare_cousin_anchor_layout(self):
        import cousin_coordinate

        task_args = getattr(self, "task_args", {}) or {}
        seed = task_args.get("seed", 0)
        episode_idx = int(task_args.get("now_ep_num", 0) or 0)
        rng = random.Random(int(seed) + episode_idx * 10007)

        target_labels = task_args.get("cousin_target_labels") or task_args.get("custom_objects") or [
            "earbud case"
        ]
        if isinstance(target_labels, str):
            target_labels = [target_labels]
        target_label = str(target_labels[0])

        clutter_specs = task_args.get("cousin_anchor_clutter") or []
        table_xy_bias = getattr(self, "table_xy_bias", (0.0, 0.0))
        base_xlim = task_args.get("cousin_anchor_table_xlim", [-0.59, 0.59])
        base_ylim = task_args.get("cousin_anchor_table_ylim", [-0.34, 0.34])
        table_xlim = (
            float(base_xlim[0]) + float(table_xy_bias[0]),
            float(base_xlim[1]) + float(table_xy_bias[0]),
        )
        table_ylim = (
            float(base_ylim[0]) + float(table_xy_bias[1]),
            float(base_ylim[1]) + float(table_xy_bias[1]),
        )

        layout_path, anchor_xy = cousin_coordinate.prepare_anchor_relative_layout(
            repo_root=Path(ROOT_PATH),
            sample_anchor_xy=lambda: self._sample_grasp_target_spawn_xy(rng=rng),
            target_label=target_label,
            clutter_specs=clutter_specs,
            table_xy_bias=(float(table_xy_bias[0]), float(table_xy_bias[1])),
            table_xlim=table_xlim,
            table_ylim=table_ylim,
            episode_idx=episode_idx,
            cousin_instance_indices=task_args.get("cousin_instance_indices"),
            edge_padding=float(task_args.get("cousin_anchor_edge_padding", 0.02)),
            max_tries=int(task_args.get("cousin_anchor_max_tries", 200)),
            rng=rng,
        )

        os.environ["COUSIN_RELATIVE_LAYOUT_JSON"] = layout_path
        task_args["cousin_relative_layout_json"] = layout_path
        task_args["use_cousin_coordinate"] = True
        task_args.setdefault("cousin_target_labels", target_labels)
        task_args.setdefault("cousin_exclude_labels", list(target_labels))
        task_args.setdefault("cousin_uniform_instance_sampling", True)
        task_args["cousin_anchor_xy"] = [float(anchor_xy[0]), float(anchor_xy[1])]
        self.task_args = task_args

    def _override_mixed_horizontal_height_ratio(self, actor):
        task_args = getattr(self, "task_args", {}) or {}
        ratio_cfg = task_args.get("pick_up_horizontal_height_ratio", None)
        if ratio_cfg is None:
            return
        cfg = getattr(actor, "config", {}) or {}
        if str(cfg.get("strategy", "")).lower() != "mixed":
            return
        cps = cfg.get("contact_points_pose")
        extents = cfg.get("extents")
        if not isinstance(cps, list) or not extents:
            return

        ratio = float(np.clip(float(ratio_cfg), 0.0, 1.0))
        ext_xzy = np.array(extents, dtype=np.float32).reshape(3)
        height_axis_idx = 1
        height_from_center = float(ext_xzy[height_axis_idx] * (ratio - 0.5))
        horizontal_count = 8 if len(cps) >= 12 else 6 if len(cps) >= 10 else 4 if len(cps) >= 5 else len(cps)
        for i in range(horizontal_count):
            mat = np.array(cps[i], dtype=np.float32).copy()
            mat[height_axis_idx, 3] = height_from_center
            cps[i] = mat.tolist()
        cfg["height_ratio"] = ratio

    def load_actors(self):
        self._custom_objects_injected = True  # handled below; suppress Base_Task generic injection
        custom_args = getattr(self, "task_args", {}) or {}
        use_custom = bool(custom_args.get("use_custom_objects", False))
        custom_obj_dir = custom_args.get("custom_objects_dir", DEFAULT_CUSTOM_ACTOR_OBJECTS_DIR)
        custom_obj_names = custom_args.get("custom_objects")
        custom_scale = float(custom_args.get("custom_object_scale", 1.0))
        custom_collision = custom_args.get("custom_object_collision", "mesh")
        custom_spawn_height = float(custom_args.get("custom_object_spawn_height", 0.0))

        # Cousin anchor layout: sample base spawn XY, write temp layout, then use cousin clutter path.
        task_args = getattr(self, "task_args", {}) or {}
        if task_args.get("cousin_anchor_layout"):
            self._prepare_cousin_anchor_layout()
            task_args = getattr(self, "task_args", {}) or {}

        # Cousin mode: strict precheck only. Grasp actor is spawned in Base_Task.get_cluttered_table
        # via get_clutter_placements (same ontop stacking / yaw / instance draw as preview).
        use_cousin = bool(task_args.get("use_cousin_coordinate", False))
        if use_cousin:
            try:
                import cousin_coordinate

                labels = task_args.get("cousin_target_labels") or ["clock"]
                if isinstance(labels, str):
                    labels = [labels]
                task_args.setdefault("cousin_exclude_labels", labels)

                placement = cousin_coordinate.get_placement_coordinates(
                    episode_idx=task_args.get("now_ep_num", 0),
                    seed=task_args.get("seed", 0),
                    task_name=task_args.get("task_name", ""),
                    task_config=task_args.get("task_config"),
                    table_xy_bias=getattr(self, "table_xy_bias", (0.0, 0.0)),
                    target_labels=labels,
                )
                if not isinstance(placement, dict):
                    raise RuntimeError(
                        f"[pick_up] cousin target labels {labels} not found in layout placement result"
                    )
                objs = placement.get("objects") or []
                if not objs:
                    raise RuntimeError(
                        f"[pick_up] cousin target labels {labels} produced empty placement objects"
                    )

                cousin_target_label = objs[0].get("label")
                if cousin_target_label is None:
                    raise RuntimeError("[pick_up] cousin placement object missing label")

                cousin_model_dir = cousin_coordinate._find_matching_our_object_dir(
                    Path(ROOT_PATH),
                    str(cousin_target_label),
                    strict=True,
                    actor_only=True,
                )
                if cousin_model_dir is None:
                    raise RuntimeError(
                        f"[pick_up] cousin target label '{cousin_target_label}' failed to map to our_assets/actor"
                    )

                _layout_obj = cousin_coordinate._get_layout_object_for_label(
                    str(cousin_target_label)
                )
                if _layout_obj is None:
                    raise RuntimeError(
                        f"[pick_up] cousin layout has no object for label {cousin_target_label!r}"
                    )
                cousin_coordinate._require_actor_only_match(
                    _layout_obj,
                    str(cousin_target_label),
                )
            except ImportError as e:
                raise RuntimeError(f"[pick_up] Import cousin_coordinate failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"[pick_up] strict cousin target precheck failed: {e}") from e

            try:
                self.table_height = 0.74 + self.table_z_bias
            except Exception:
                self.table_height = 0.74
            self.bread = []
            self.bread_info = []
            return

        # Base mode: sample grasp target XY using the same strip logic as before.
        task_args = getattr(self, "task_args", {}) or {}
        anchor_x, anchor_y = self._sample_grasp_target_spawn_xy()

        try:
            self.table_height = 0.74 + self.table_z_bias
        except Exception:
            self.table_height = 0.74

        z_offset = 0.005
        default_base_qpos = list(BUILTIN_OBJECT_UPRIGHT_BASE_QPOS)
        if (
            use_custom
            or pick_up_spawn_horizontal_enabled(custom_args)
            or "cousin_random_yaw" in custom_args
        ):
            spawn_qpos = pick_up_spawn_quat(custom_args, use_custom)
            rotate_rand_final = False
            rotate_lim_final = [0, 0, 0]
        else:
            spawn_qpos = default_base_qpos
            rotate_rand_final = True
            rotate_lim_final = [0, np.pi / 4, 0]
        rand_pos = rand_pose(
            xlim=[anchor_x, anchor_x],
            ylim=[anchor_y, anchor_y],
            zlim=[
                self.table_height + z_offset + custom_spawn_height,
                self.table_height + z_offset + custom_spawn_height,
            ],
            rotate_rand=rotate_rand_final,
            rotate_lim=rotate_lim_final,
            qpos=np.array(spawn_qpos, dtype=np.float32),
        )

        self.bread = []
        self.bread_info = []

        selected_obj_dir = None
        if use_custom:
            obj_dirs = get_custom_object_directories(custom_obj_dir, custom_obj_names)
            selected_obj_dir = np.random.choice(obj_dirs) if obj_dirs else None
            bread_actor = create_custom_object_actor(
                scene=self.scene,
                obj_dir=selected_obj_dir,
                pose=rand_pos,
                custom_scale=custom_scale,
                custom_collision=custom_collision,
                default_mass=0.05,
                table_height=self.table_height,
                calculate_z_offset=True,
            )
            if bread_actor is not None:
                try:
                    cfg = bread_actor.config or {}
                    ext = cfg.get("extents", None)
                    if ext is not None:
                        ext_xzy = np.array(ext, dtype=np.float32).reshape(3)
                        sc_xzy = cfg.get("scale", 1.0)
                        if isinstance(sc_xzy, (int, float, np.floating)):
                            sc_xzy = [float(sc_xzy), float(sc_xzy), float(sc_xzy)]
                        sc_xzy = np.array(sc_xzy, dtype=np.float32).reshape(3)
                        # model_data axis order (x, z, y): vertical extent uses index 1
                        self.bread_height = float(abs(ext_xzy[1] * sc_xzy[1]))
                    else:
                        self.bread_height = self._estimate_actor_max_dimension(bread_actor)
                except Exception:
                    self.bread_height = self._estimate_actor_max_dimension(bread_actor)
        else:
            bread_model_id = 7
            bread_actor = create_actor(
                scene=self,
                pose=rand_pos,
                modelname="001_bottle",
                convex=True,
                model_id=bread_model_id,
            )
            self.bread_height = self._estimate_actor_max_dimension(bread_actor)
        apply_pick_up_horizontal_spawn_physics(bread_actor, custom_args)
        self._override_mixed_horizontal_height_ratio(bread_actor)
        self.bread.append(bread_actor)
        self.bread_info.append(
            get_custom_object_label(selected_obj_dir)
            if selected_obj_dir is not None
            else "001_bottle/base7"
        )
        self.add_prohibit_area(bread_actor, padding=0.03)

    def play_once(self):
        task_args = getattr(self, "task_args", {}) or {}
        if len(self.bread) > 0:
            apply_pick_up_horizontal_spawn_physics(self.bread[0], task_args)
        if bool(task_args.get("use_cousin_coordinate")) and bool(
            task_args.get("cousin_settle_before_grasp", True)
        ):
            max_steps = int(task_args.get("cousin_settle_max_sim_steps", 8000))
            if max_steps > 0:
                self.wait_for_dynamic_settle(
                    max_steps=max_steps,
                    stable_window=int(task_args.get("cousin_settle_stable_window", 48)),
                    lin_tol=float(task_args.get("cousin_settle_lin_tol", 0.02)),
                    ang_tol=float(task_args.get("cousin_settle_ang_tol", 0.2)),
                    quiet=not bool(task_args.get("cousin_settle_verbose", False)),
                    with_render=bool(task_args.get("cousin_settle_with_render", False)),
                )

        # 成功判定基准：本段 play_once 开始、尚未做抓取/抬臂前的物体中心高度（相对桌面的绝对 z 会误伤高物体）
        self._pick_up_obj_z0 = float(self.bread[0].get_pose().p[2])

        # 1. 确定使用的手臂
        arm_tag = ArmTag("right" if self.bread[0].get_pose().p[0] > 0 else "left")
        arm_info = "left" if self.bread[0].get_pose().p[0] < 0 else "right"
        
        self.info["info"] = {
            "{B}": self.bread_info[0],
            "{a}": arm_info,
        }
        
        # 2. 物体高度 + model_data strategy：仅 mixed 策略按高度切水平/竖直接触点子集；
        # 且仅对 mixed_grasp_height_force_embodiments 中的 embodiment 生效（默认 aloha-agilex）。
        # obb / 其他策略由 choose_grasp_pose 自动选点（含 OBB 向上滤波等）。
        object_height = getattr(self, "bread_height", 0.10)
        use_height_force = self._use_mixed_grasp_height_force()
        spawn_horizontal = pick_up_spawn_horizontal_enabled(task_args)
        force_vertical = use_height_force and object_height < 0.08 and not spawn_horizontal
        force_horizontal = (use_height_force and object_height > 0.1) or spawn_horizontal

        cp_ids = [cp_id for cp_id, _ in self.bread[0].iter_contact_points()]
        obj_cfg = getattr(self.bread[0], "config", {}) or {}
        grasp_strategy = str(obj_cfg.get("strategy", "")).lower()
        use_mixed_branch = grasp_strategy == "mixed"

        grasp_action = (None, [])
        if force_vertical and use_mixed_branch:
            if len(cp_ids) >= 12:
                vertical_cp_ids = cp_ids[8:]
            elif len(cp_ids) >= 10:
                vertical_cp_ids = cp_ids[6:]
            elif len(cp_ids) >= 5:
                vertical_cp_ids = cp_ids[4:]
            else:
                vertical_cp_ids = cp_ids
            grasp_action = self.grasp_actor(
                self.bread[0], arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=vertical_cp_ids
            )
        elif force_horizontal and use_mixed_branch:
            if len(cp_ids) >= 12:
                horizontal_cp_ids = cp_ids[:8]
            elif len(cp_ids) >= 10:
                horizontal_cp_ids = cp_ids[:6]
            elif len(cp_ids) >= 5:
                horizontal_cp_ids = cp_ids[:4]
            else:
                horizontal_cp_ids = cp_ids
            grasp_action = self.grasp_actor(
                self.bread[0], arm_tag=arm_tag, pre_grasp_dis=0.1, contact_point_id=horizontal_cp_ids
            )
        else:
            grasp_action = self.grasp_actor(self.bread[0], arm_tag=arm_tag, pre_grasp_dis=0.1)

        if grasp_action[0] is None:
            grasp_action = self.grasp_actor(self.bread[0], arm_tag=arm_tag, pre_grasp_dis=0.1)

        self.move(grasp_action)

        # --- 提升：默认 3cm（``pick_up_lift_m`` 可覆盖）---
        total_lift = float(task_args.get("pick_up_lift_m", 0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=total_lift, move_axis="world"))

        return self.info

    def check_success(self):
        if len(self.bread) == 0:
            return False
        if getattr(self, "last_failure_code", None) == "no_valid_grasp_cp_after_filter":
            return False

        z0 = getattr(self, "_pick_up_obj_z0", None)
        if z0 is None:
            return False

        curr_z = float(self.bread[0].get_pose().p[2])
        task_args = getattr(self, "task_args", None) or {}
        lift_m = float(task_args.get("pick_up_lift_m", 0.03))
        # 相对 play_once 开始时物体中心至少抬高多少（默认 max(8mm, 25%×指令抬升量)，可 ``pick_up_success_min_lift_m`` 覆盖）
        min_delta = task_args.get("pick_up_success_min_lift_m", None)
        if min_delta is None:
            min_delta = max(0.008, 0.25 * lift_m)
        else:
            min_delta = float(min_delta)

        return (curr_z - float(z0)) >= min_delta