"""Shared eval-time success checks for shake_bottle tasks."""

from __future__ import annotations

import numpy as np
import transforms3d as t3d

# Policy eval: require one pos↔neg swing (expert play_once does 3; collection unchanged).
REQUIRED_SHAKE_CYCLES = 1
MIN_LIFT_M = 0.02
MAX_HOLD_DIST_M = 0.15
# Expert peaks near ±157°; ignore small wiggles during eval.
SHAKE_PEAK_RAD = np.deg2rad(45.0)


def reset_eval_shake_state(task) -> None:
    task._eval_shake_cycles = 0
    task._eval_shake_phase = None
    task._eval_shake_ref_q = None


def _bottle_shake_angle_y(bottle_q: np.ndarray, ref_q: np.ndarray) -> float:
    ref_mat = t3d.quaternions.quat2mat(ref_q)
    curr_mat = t3d.quaternions.quat2mat(bottle_q)
    rel_mat = ref_mat.T @ curr_mat
    return float(t3d.euler.mat2euler(rel_mat)[1])


def _update_eval_shake_cycles(task, bottle_q: np.ndarray) -> int:
    if task._eval_shake_ref_q is None:
        task._eval_shake_ref_q = np.array(bottle_q, dtype=np.float64)

    angle_y = _bottle_shake_angle_y(bottle_q, task._eval_shake_ref_q)
    phase = task._eval_shake_phase

    if angle_y >= SHAKE_PEAK_RAD:
        if phase == "neg":
            task._eval_shake_cycles += 1
        task._eval_shake_phase = "pos"
    elif angle_y <= -SHAKE_PEAK_RAD:
        if phase == "pos":
            task._eval_shake_cycles += 1
        task._eval_shake_phase = "neg"

    return int(task._eval_shake_cycles)


def bottle_is_held(task, bottle_pos: np.ndarray) -> bool:
    init_z = float(getattr(task, "bottle_init_z", bottle_pos[2]))
    if bottle_pos[2] < init_z + MIN_LIFT_M:
        return False

    arm_tag = "right" if bottle_pos[0] > 0 else "left"
    if arm_tag == "right":
        if not task.is_right_gripper_close():
            return False
        ee_pos = np.array(task.robot.get_right_ee_pose()[:3], dtype=np.float64)
    else:
        if not task.is_left_gripper_close():
            return False
        ee_pos = np.array(task.robot.get_left_ee_pose()[:3], dtype=np.float64)

    return float(np.linalg.norm(bottle_pos - ee_pos)) < MAX_HOLD_DIST_M


def check_shake_bottle_success(task) -> bool:
    bottle_pos = np.array(task.bottle.get_pose().p, dtype=np.float64)
    if not bottle_is_held(task, bottle_pos):
        if getattr(task, "eval_mode", False):
            reset_eval_shake_state(task)
        return False

    if not getattr(task, "eval_mode", False):
        return True

    # Expert seed check calls check_success once after play_once (move(), not take_action).
    # Shake cycles only accumulate during policy take_action; skip here so expert filter still works.
    if int(getattr(task, "take_action_cnt", 0) or 0) <= 0:
        return True

    bottle_q = np.array(task.bottle.get_pose().q, dtype=np.float64)
    cycles = _update_eval_shake_cycles(task, bottle_q)
    return cycles >= REQUIRED_SHAKE_CYCLES
