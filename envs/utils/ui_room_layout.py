"""
Load UI-style room furniture layouts (JSON) into an existing Sapien scene.

Layout format matches ``envs/room_config/*/*.json`` and ``ui_generated_layout.json``.

**Avoiding the manipulation table (default):** after resolving the full JSON layout,
we take the **union XY AABB** of all pieces; if it intersects the **table + margin**
rectangle, we apply **one** translation ``(dx, dy)`` to **every** object. The slide
direction never favors **+Y** (RoboTwin's default ``wall`` sits at ``y≈1`` with half
thickness ``0.6``; the desk is on the **y < 0.4** side). We then clamp the union so
``y_max <= room_side_max_y`` (default ``≈0.32``), keeping the block on the **same side
as the desk** vs that wall. Override with YAML ``ui_room_room_side_max_y`` if your wall
differs.

Disable table keepout shift with ``ui_room_layout_disable_keepout: true`` (clamp to
wall side still applies unless you fork this module).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import sapien.core as sapien
import trimesh
from transforms3d.quaternions import quat2mat

from envs._GLOBAL_CONFIGS import ROOT_PATH

_DEFAULT_TABLE_LENGTH = 1.2
_DEFAULT_TABLE_WIDTH = 0.7
_DEFAULT_ARM_CLEARANCE_XY = 0.72

# Matches ``create_table_and_wall`` default wall: pose y=1, half_y=0.6 → wall occupies y∈[0.4, 1.6].
# Table sits at y≈0 on the room side; furniture must stay y < inner_face - margin (same side as desk).
_DEFAULT_WALL_Y_CENTER = 1.0
_DEFAULT_WALL_Y_HALF_THICK = 0.6
_DEFAULT_ROOM_SIDE_Y_MARGIN = 0.08  # stay clearly in front of wall inner face


def resolve_glb_path(path_str: str) -> str:
    """Resolve room-layout mesh paths against repo root when needed."""
    p = Path(path_str)
    if p.is_file():
        return str(p)
    if not p.is_absolute():
        candidate_rel = Path(ROOT_PATH) / p
        if candidate_rel.is_file():
            return str(candidate_rel)
    s = str(path_str).replace("\\", "/")
    marker = "our_assets/"
    if marker in s:
        rel = s[s.index(marker) :]
        candidate = Path(ROOT_PATH) / rel
        if candidate.is_file():
            return str(candidate)
    return str(path_str)


def get_aligned_pose(x, y, z, angle_deg, fix_axis=None):
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    if fix_axis == "x_to_z":
        half_angle = -np.pi / 4.0
        qw = np.cos(half_angle)
        qy = np.sin(half_angle)
    elif fix_axis == "y_to_z":
        half_angle = np.pi / 4.0
        qw = np.cos(half_angle)
        qx = np.sin(half_angle)
    elif fix_axis == "neg_x_to_z":
        half_angle = np.pi / 4.0
        qw = np.cos(half_angle)
        qy = np.sin(half_angle)
    elif fix_axis == "neg_y_to_z":
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
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ]

    final_quat = quat_multiply(quat_rotate_z, quat_fix)
    return sapien.Pose(p=[x, y, z], q=final_quat)


def get_model_metadata(model_path: str, scale=1.0):
    mesh = trimesh.load(str(model_path), force="mesh")
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    if np.isscalar(scale):
        extents = (bounds[1] - bounds[0]) * float(scale)
    else:
        sc = np.asarray(scale, dtype=np.float64)
        extents = (bounds[1] - bounds[0]) * sc
    return center, extents


def get_world_extents(raw_extents, quat):
    mat = quat2mat(quat)
    half = np.array(raw_extents) / 2.0
    corners = np.array(
        [[i, j, k] for i in [-half[0], half[0]] for j in [-half[1], half[1]] for k in [-half[2], half[2]]]
    )
    world_corners = corners @ mat.T
    world_min = np.min(world_corners, axis=0)
    world_max = np.max(world_corners, axis=0)
    return world_max - world_min


def calculate_semantic_pos(item, asset_metadata, placed_objects, align_offset: float):
    self_ext = asset_metadata[item["name"]]["world_extents"]
    floor_h = 0.1

    if item.get("anchor") is None:
        pos = item["pos_abs"]
        x, y = float(pos[0]), float(pos[1])
        z = self_ext[2] / 2.0 + floor_h
        return x, y, z

    anchor_name = item["anchor"]
    if anchor_name not in placed_objects:
        raise KeyError(f"Missing anchor object: {anchor_name}")

    ax, ay = placed_objects[anchor_name]["pos_2d"]
    az = placed_objects[anchor_name]["pos_z"]
    a_ext = asset_metadata[anchor_name]["world_extents"]

    side = item.get("rel_side", "left")
    gap = float(item.get("gap", 0.0))

    anchor_bottom_z = az - (a_ext[2] / 2.0)
    anchor_top_z = az + (a_ext[2] / 2.0)

    if side == "top":
        x = ax + float(item.get("x_extra", 0.0))
        y = ay + float(item.get("y_extra", 0.0))
        z = anchor_top_z + (self_ext[2] / 2.0) + gap
    else:
        z = anchor_bottom_z + (self_ext[2] / 2.0)
        if align_offset == -90:
            if side == "left":
                x = ax - (a_ext[0] / 2 + self_ext[0] / 2 + gap)
                y = ay
            elif side == "right":
                x = ax + (a_ext[0] / 2 + self_ext[0] / 2 + gap)
                y = ay
            elif side == "back":
                x = ax
                y = ay + (a_ext[1] / 2 + self_ext[1] / 2 + gap)
            elif side == "front":
                x = ax
                y = ay - (a_ext[1] / 2 + self_ext[1] / 2 + gap)
            else:
                x, y = ax, ay
        elif align_offset == 90:
            if side == "left":
                x = ax + (a_ext[0] / 2 + self_ext[0] / 2 + gap)
                y = ay
            elif side == "right":
                x = ax - (a_ext[0] / 2 + self_ext[0] / 2 + gap)
                y = ay
            elif side == "back":
                x = ax
                y = ay - (a_ext[1] / 2 + self_ext[1] / 2 + gap)
            elif side == "front":
                x = ax
                y = ay + (a_ext[1] / 2 + self_ext[1] / 2 + gap)
            else:
                x, y = ax, ay
        else:
            x, y = ax, ay

    return x, y, z


def _aabb_overlap_xy(ax0: float, ax1: float, ay0: float, ay1: float, bx0: float, bx1: float, by0: float, by1: float) -> bool:
    if ax1 < bx0 or ax0 > bx1:
        return False
    if ay1 < by0 or ay0 > by1:
        return False
    return True


def _default_room_side_max_y() -> float:
    return float(_DEFAULT_WALL_Y_CENTER - _DEFAULT_WALL_Y_HALF_THICK - _DEFAULT_ROOM_SIDE_Y_MARGIN)


def _group_shift_xy_clear_table(
    gx0: float,
    gx1: float,
    gy0: float,
    gy1: float,
    bx: float,
    by: float,
    khx: float,
    khy: float,
    *,
    step: float,
    max_shift: float,
) -> tuple[float, float]:
    """
    Move union AABB until it clears the table keepout. Primary slide never pushes toward +Y
    (RoboTwin default back wall is at y≈+1); bias is into the room (-Y), then extra -Y / ±X.
    """
    kx0, kx1 = bx - khx, bx + khx
    ky0, ky1 = by - khy, by + khy

    def _ov(dxx: float, dyy: float) -> bool:
        return _aabb_overlap_xy(gx0 + dxx, gx1 + dxx, gy0 + dyy, gy1 + dyy, kx0, kx1, ky0, ky1)

    if not _ov(0.0, 0.0):
        return 0.0, 0.0
    gcx = (gx0 + gx1) * 0.5
    gcy = (gy0 + gy1) * 0.5
    room_inward = np.array([0.0, -1.0], dtype=np.float64)
    v = np.array([gcx - bx, gcy - by], dtype=np.float64)
    if v[1] >= -1e-6:
        v = np.array([gcx - bx, -1.0], dtype=np.float64)
    nv = float(np.linalg.norm(v))
    if nv < 1e-5:
        v = room_inward.copy()
    else:
        v = v / nv
        if float(np.dot(v, room_inward)) < 0.05:
            v = room_inward.copy()
    dx, dy = 0.0, 0.0
    nmax = max(1, int(max_shift / max(step, 1e-6)) + 2)
    for _ in range(nmax):
        if not _ov(dx, dy):
            return dx, dy
        dx += float(v[0]) * step
        dy += float(v[1]) * step
    if _ov(dx, dy):
        for _ in range(nmax):
            if not _ov(dx, dy):
                return dx, dy
            dy -= step
    if _ov(dx, dy):
        for sx in (step, -step):
            for _ in range(nmax):
                if not _ov(dx, dy):
                    return dx, dy
                dx += sx
    return dx, dy


def _scale_vec(item: dict[str, Any]) -> list[float]:
    s = item.get("scale", 1.0)
    if isinstance(s, (int, float)):
        return [float(s)] * 3
    if isinstance(s, (list, tuple)) and len(s) >= 3:
        return [float(s[0]), float(s[1]), float(s[2])]
    return [1.0, 1.0, 1.0]


def _spawn_actor(
    scene: sapien.Scene,
    pose: sapien.Pose,
    model_path: str,
    name: str,
    *,
    scale: list[float],
    is_static: bool = True,
    with_collision: bool = False,
) -> sapien.Entity:
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static" if is_static else "dynamic")
    sc = np.array(scale, dtype=np.float32)
    builder.add_visual_from_file(filename=model_path, pose=sapien.Pose(), scale=sc)
    if with_collision:
        builder.add_convex_collision_from_file(
            filename=model_path,
            pose=sapien.Pose(),
            scale=sc,
            material=scene.default_physical_material,
        )
    builder.set_initial_pose(pose)
    return builder.build(name=name)


def spawn_ui_room_furniture(
    scene: sapien.Scene,
    layout_json_path: str | os.PathLike[str],
    *,
    align_offset_deg: float = 90.0,
    with_collision: bool = False,
    name_prefix: str = "ui_bg_",
    verbose: bool = True,
    avoid_table_robot: bool = True,
    table_xy_bias: tuple[float, float] = (0.0, 0.0),
    table_length: float = _DEFAULT_TABLE_LENGTH,
    table_width: float = _DEFAULT_TABLE_WIDTH,
    arm_clearance_xy: float = _DEFAULT_ARM_CLEARANCE_XY,
    keepout_margin_m: float = 0.12,
    group_shift_step_m: float = 0.06,
    group_shift_max_m: float = 8.0,
    # Max world Y for any furniture (union top); keeps block on same side as desk vs default wall.
    room_side_max_y: float | None = None,
) -> list[str]:
    path = Path(layout_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"ui room layout not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        room_config: list[dict[str, Any]] = json.load(f)

    bx, by = float(table_xy_bias[0]), float(table_xy_bias[1])
    khx = max(0.05, float(table_length) * 0.5 + float(arm_clearance_xy) + float(keepout_margin_m))
    khy = max(0.05, float(table_width) * 0.5 + float(arm_clearance_xy) + float(keepout_margin_m))

    asset_metadata: dict[str, Any] = {}
    for item in room_config:
        raw_path = item["path"]
        model_path = resolve_glb_path(raw_path)
        scale_vec = _scale_vec(item)
        _, raw_extents = get_model_metadata(model_path, scale=scale_vec)
        rot = float(align_offset_deg) + float(item.get("rotation_deg", 0))
        temp_pose = get_aligned_pose(0, 0, 0, rot, fix_axis=item.get("fix_axis"))
        world_ext = get_world_extents(raw_extents, temp_pose.q)
        asset_metadata[item["name"]] = {"world_extents": world_ext, "final_quat": temp_pose.q}

    # Pass 1: semantic layout only (internal relative positions unchanged).
    sem_placed: dict[str, Any] = {}
    resolved: list[tuple[str, dict[str, Any], float, float, float, str]] = []

    for item in room_config:
        name = str(item["name"])
        model_path = resolve_glb_path(item["path"])
        if not Path(model_path).is_file():
            if verbose:
                print(f"[ui_room_layout] skip missing mesh: {model_path} ({name})")
            continue
        try:
            x, y, z = calculate_semantic_pos(item, asset_metadata, sem_placed, align_offset_deg)
        except Exception as e:
            if verbose:
                print(f"[ui_room_layout] pose error for {name}: {e}")
            continue
        sem_placed[name] = {"pos_2d": (x, y), "pos_z": z}
        resolved.append((name, item, x, y, z, model_path))

    rmsy = float(room_side_max_y) if room_side_max_y is not None else _default_room_side_max_y()

    gx0, gy0 = 1e9, 1e9
    gx1, gy1 = -1e9, -1e9
    if resolved:
        for name, _it, x, y, _z, _mp in resolved:
            ext = asset_metadata[name]["world_extents"]
            fhx = max(float(ext[0]) * 0.5, 0.01)
            fhy = max(float(ext[1]) * 0.5, 0.01)
            gx0 = min(gx0, x - fhx)
            gx1 = max(gx1, x + fhx)
            gy0 = min(gy0, y - fhy)
            gy1 = max(gy1, y + fhy)

    dx, dy = 0.0, 0.0
    if avoid_table_robot and resolved:
        dx, dy = _group_shift_xy_clear_table(
            gx0, gx1, gy0, gy1, bx, by, khx, khy, step=group_shift_step_m, max_shift=group_shift_max_m
        )
        if verbose and (abs(dx) > 1e-6 or abs(dy) > 1e-6):
            print(f"[ui_room_layout] group shift (whole room) dx={dx:.3f} dy={dy:.3f} (clear table keepout)")

    if resolved and gy1 + dy > rmsy:
        dy += rmsy - (gy1 + dy)
        if verbose:
            print(f"[ui_room_layout] clamp room to y<= {rmsy:.3f} (desk side of default wall)")
    if avoid_table_robot and resolved and verbose:
        kx0, kx1 = bx - khx, bx + khx
        ky0, ky1 = by - khy, by + khy
        gx0s, gy0s, gx1s, gy1s = 1e9, 1e9, -1e9, -1e9
        for name, _it, x, y, _z, _mp in resolved:
            ext = asset_metadata[name]["world_extents"]
            fhx = max(float(ext[0]) * 0.5, 0.01)
            fhy = max(float(ext[1]) * 0.5, 0.01)
            gx0s = min(gx0s, x + dx - fhx)
            gx1s = max(gx1s, x + dx + fhx)
            gy0s = min(gy0s, y + dy - fhy)
            gy1s = max(gy1s, y + dy + fhy)
        if _aabb_overlap_xy(gx0s, gx1s, gy0s, gy1s, kx0, kx1, ky0, ky1):
            print(
                "[ui_room_layout] warn: room AABB still intersects table keepout after shift/clamp; "
                "increase ui_room_group_shift_max_m or shrink presets."
            )
        if gy1s > rmsy + 0.02:
            print(f"[ui_room_layout] warn: room top y={gy1s:.3f} above room_side_max_y={rmsy:.3f}")

    built_names: list[str] = []
    for name, item, x, y, z, model_path in resolved:
        x += dx
        y += dy
        pose = sapien.Pose(p=[x, y, z], q=asset_metadata[name]["final_quat"])
        actor_name = f"{name_prefix}{name}"
        _spawn_actor(
            scene,
            pose,
            model_path,
            actor_name,
            scale=_scale_vec(item),
            is_static=bool(item.get("is_static", True)),
            with_collision=with_collision,
        )
        built_names.append(actor_name)
        if verbose:
            print(f"[ui_room_layout] {actor_name} @ ({x:.2f}, {y:.2f}, {z:.2f}) collision={with_collision}")

    return built_names
