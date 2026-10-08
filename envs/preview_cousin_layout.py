from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import ROOT_PATH
from .utils import *
import numpy as np
from pathlib import Path


class preview_cousin_layout(Base_Task):
    """
    Visualize cousin_coordinate layout on the table.

    With ``use_cousin_coordinate``, all objects come from ``get_cluttered_table`` only
    (same cousin JSON → clutter pipeline as other tasks). No separate pick_up-style
    target actor — that would duplicate the first layout object.
    """

    def setup_demo(self, **kwargs):
        kwargs = dict(kwargs)
        kwargs.setdefault("save_data", False)
        kwargs.setdefault("need_plan", False)
        kwargs.setdefault("eval_mode", False)
        kwargs.setdefault("skip_stable_check", True)

        dr = dict(kwargs.get("domain_randomization") or {})
        dr["cluttered_table"] = True
        dr["clean_background_rate"] = 0
        dr.setdefault("random_background", False)
        dr.setdefault("random_light", False)
        dr.setdefault("random_table_height", 0)
        dr.setdefault("random_head_camera_dis", 0)
        kwargs["domain_randomization"] = dr

        kwargs.setdefault("task_name", "preview_cousin_layout")

        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self._custom_objects_injected = True  # handled below; suppress Base_Task generic injection
        custom_args = getattr(self, "task_args", {}) or {}
        use_custom = bool(custom_args.get("use_custom_objects", False))
        custom_obj_dir = custom_args.get("custom_objects_dir", DEFAULT_CUSTOM_ACTOR_OBJECTS_DIR)
        custom_obj_names = custom_args.get("custom_objects")
        custom_scale = float(custom_args.get("custom_object_scale", 1.0))
        custom_collision = custom_args.get("custom_object_collision", "mesh")
        custom_spawn_height = float(custom_args.get("custom_object_spawn_height", 0.0))
        custom_upright_only = bool(custom_args.get("custom_object_upright_only", True))
        custom_base_qpos = custom_args.get("custom_object_base_qpos", [0.7071, 0.7071, 0.0, 0.0])

        task_args = getattr(self, "task_args", {}) or {}
        use_cousin = bool(task_args.get("use_cousin_coordinate", False))

        try:
            self.table_height = 0.74 + self.table_z_bias
        except Exception:
            self.table_height = 0.74

        self.bread = []
        self.bread_info = []

        # Cousin preview: objects only from get_cluttered_table (after this method).
        if use_cousin:
            return

        rand_pos = rand_pose(
            xlim=[-0.2, 0.2],
            ylim=[-0.1, 0.1],
            qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True,
            rotate_lim=[0, np.pi / 4, 0],
        )

        z_offset = 0.005
        rotate_rand_final = not custom_upright_only if use_custom else True
        rotate_lim_final = [0, 0, np.pi] if use_custom else [0, np.pi / 4, 0]
        rand_pos = rand_pose(
            xlim=[rand_pos.p[0], rand_pos.p[0]],
            ylim=[rand_pos.p[1], rand_pos.p[1]],
            zlim=[
                self.table_height + z_offset + custom_spawn_height,
                self.table_height + z_offset + custom_spawn_height,
            ],
            rotate_rand=rotate_rand_final,
            rotate_lim=rotate_lim_final,
            qpos=np.array(custom_base_qpos, dtype=np.float32)
            if use_custom
            else [0.707, 0.707, 0.0, 0.0],
        )

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
        else:
            id_list = [0, 1, 3, 5, 6]
            bread_model_id = np.random.choice(id_list)
            bread_actor = create_actor(
                scene=self,
                pose=rand_pos,
                modelname="001_bottle",
                convex=True,
                model_id=bread_model_id,
            )
        self.bread.append(bread_actor)
        self.bread_info.append(
            get_custom_object_label(selected_obj_dir, Path(ROOT_PATH) / custom_obj_dir)
            if selected_obj_dir is not None
            else f"object_{bread_actor.actor.get_name()}"
        )
        self.add_prohibit_area(bread_actor, padding=0.03)

    def play_once(self):
        return {}

    def check_success(self):
        # No task criterion; episode is successful if init + play_once finished (for collect scripts).
        return True
