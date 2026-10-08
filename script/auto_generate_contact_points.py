#!/usr/bin/env python3
"""
Automatically generate contact points for object assets.

Grasps are generated directly in the mesh coordinate frame, without inferring
"main axis" or object type from extents. If `model_data.json` does not exist,
it is created from mesh files automatically.

Usage:
    # Process all objects (actor + non-actor)
    python script/auto_generate_contact_points.py --all --strategy mixed --height_axis Y

    # Process one branch under our_assets (for example actor only)
    python script/auto_generate_contact_points.py --assets_branch_dir our_assets/actor --strategy mixed --height_axis Y

    # Process a single object instance
    python script/auto_generate_contact_points.py --object_dir our_assets/actor/coke/0

    # Specify strategy and parameters
    python script/auto_generate_contact_points.py --object_dir our_assets/actor/coke/0 --strategy mixed --height_axis Y --height_ratio 0.6 --vertical_ratio 0.3
"""

import json
import argparse
import numpy as np
import transforms3d as t3d
from pathlib import Path
import trimesh
import xml.etree.ElementTree as ET
from typing import List


def _collect_object_instance_dirs(our_objects_dir: Path) -> List[Path]:
    object_dirs: List[Path] = []
    for obj_dir in our_objects_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        instance_subdirs = [p for p in obj_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if instance_subdirs:
            object_dirs.extend(sorted(instance_subdirs, key=lambda p: int(p.name)))
        elif (obj_dir / "mesh").exists():
            # Backward compatibility for legacy single-level layout.
            object_dirs.append(obj_dir)
    return object_dirs


def find_urdf_in_dir(directory):
    """Return the first URDF file found in the directory."""
    directory = Path(directory)
    for urdf_file in directory.glob("*.urdf"):
        return urdf_file
    return None


def generate_rotation_matrix(axis, angle):
    """Generate a rotation matrix around the specified axis."""
    if axis == 'X':
        return t3d.euler.euler2mat(angle, 0, 0)
    elif axis == 'Y':
        return t3d.euler.euler2mat(0, angle, 0)
    elif axis == 'Z':
        return t3d.euler.euler2mat(0, 0, angle)


def axis_to_index(axis: str) -> int:
    axis = str(axis).upper().strip()
    if axis not in ("X", "Y", "Z"):
        raise ValueError(f"axis must be one of X/Y/Z, got {axis!r}")
    return {"X": 0, "Y": 1, "Z": 2}[axis]


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute rotation R such that R @ a ~= b."""
    va = np.asarray(a, dtype=float).reshape(3)
    vb = np.asarray(b, dtype=float).reshape(3)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3, dtype=float)
    va = va / na
    vb = vb / nb
    c = float(np.clip(np.dot(va, vb), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=float)
    if c < -1.0 + 1e-10:
        # 180-degree case: choose any axis orthogonal to va.
        helper = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(helper, va))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = np.cross(va, helper)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return np.eye(3, dtype=float)
        axis = axis / axis_norm
        K = _skew(axis)
        # Rodrigues for pi rad: R = I + 2 K^2
        return np.eye(3, dtype=float) + 2.0 * (K @ K)

    v = np.cross(va, vb)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3, dtype=float)
    vx = _skew(v)
    # Stable form of Rodrigues using cross-product matrix of v (not unit axis).
    return np.eye(3, dtype=float) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def maybe_align_thin_object_short_axis_to_height(
    raw_extents: np.ndarray,
    threshold: float = 0.03,
    mesh_height_axis: str = "Y",
) -> np.ndarray:
    """
    If the shortest raw AABB axis is <= threshold, rotate model so that this axis
    aligns with the specified mesh height axis.
    NOTE:
    - This uses ONLY raw extents axis semantics (X/Y/Z), without OBB orientation.
    - Return value is transform_matrix only.
    """
    raw_extents = np.asarray(raw_extents, dtype=float).reshape(3)
    short_idx = int(np.argmin(raw_extents))
    short_len = float(raw_extents[short_idx])

    transform_matrix = np.eye(4, dtype=float)
    if short_len > float(threshold):
        return transform_matrix

    axis_names = {0: "X", 1: "Y", 2: "Z"}
    target_axis_idx = axis_to_index(mesh_height_axis)
    current_short_axis = np.zeros(3, dtype=float)
    current_short_axis[short_idx] = 1.0
    target_axis = np.zeros(3, dtype=float)
    target_axis[target_axis_idx] = 1.0

    R_align = _rotation_matrix_from_a_to_b(current_short_axis, target_axis)
    transform_matrix[:3, :3] = R_align

    print(
        f"[Thin-object alignment] shortest axis={short_len:.4f}m "
        f"({axis_names.get(short_idx, str(short_idx))}) <= {float(threshold):.4f}m, "
        f"aligned to mesh height axis {mesh_height_axis.upper()}"
    )
    return transform_matrix


def generate_mixed_grasps(
    extents,
    height_axis="Y",
    num_horizontal=8,
    height_ratio=0.5,
    vertical_ratio=0.15,
    num_vertical=4,
):
    """
    Generate a mixed grasp set: horizontal ring grasps + vertical grasps.

    Args:
        extents: Object dimensions [x, y, z].
        height_axis: Height axis in mesh coordinates (default: Y).
        num_horizontal: Number of horizontal grasps (default: 8, every 45 degrees).
        height_ratio: Height ratio for horizontal grasps along `height_axis` (0.0-1.0).
        vertical_ratio: Height ratio for vertical grasps along `height_axis` (0.0-1.0).
        num_vertical: Number of vertical grasps (default: 4, every 90 degrees).
    """
    print(f"[Generate] Mixed grasp mode: {num_horizontal} horizontal + {num_vertical} vertical (lower-mid region)")
    
    contact_points = []
    height_axis = str(height_axis).upper().strip()
    height_axis_idx = axis_to_index(height_axis)
    
    # Compute horizontal grasp translation using the configured height ratio.
    height_from_bottom = extents[height_axis_idx] * height_ratio
    height_from_center = height_from_bottom - extents[height_axis_idx] / 2
    horizontal_position = np.array([0.0, 0.0, 0.0])
    horizontal_position[height_axis_idx] = height_from_center
    
    print(f"  [Horizontal] {num_horizontal} points, height_ratio={height_ratio}")
    print(f"    Position: {height_from_bottom*100:.1f}cm from bottom, {height_from_center*100:.1f}cm from center")
    
    # Build horizontal grasps. We keep the historical rotation convention for
    # RoboTwin compatibility, but use explicit `height_axis` instead of inferred axes.
    if height_axis_idx == 1:  # Height axis is Y.
        base_rotation = np.array([
            [0, -1, 0],  # X points to -Y (gripper approach).
            [1, 0, 0],   # Y points to +X (radial / finger axis).
            [0, 0, 1]    # Z points to +Z (tangential).
        ])
        rotation_axis = 'Y'
    elif height_axis_idx == 2:  # Height axis is Z.
        base_rotation = np.array([
            [0, 0, -1],  # X points to -Z (gripper approach).
            [1, 0, 0],   # Y points to +X (radial / finger axis).
            [0, -1, 0]   # Z is tangential.
        ])
        rotation_axis = 'Z'
    else:  # Height axis is X.
        base_rotation = np.array([
            [-1, 0, 0],  # X points to -X (gripper approach).
            [0, 1, 0],   # Y points to +Y (radial / finger axis).
            [0, 0, -1]   # Z is tangential.
        ])
        rotation_axis = 'X'
    
    for i in range(num_horizontal):
        angle = 2 * np.pi * i / num_horizontal
        rotation = generate_rotation_matrix(rotation_axis, angle)
        final_rotation = rotation @ base_rotation
        
        transform = np.eye(4)
        transform[:3, :3] = final_rotation
        transform[:3, 3] = horizontal_position
        contact_points.append(transform.tolist())
        
        print(f"    Point {i}: rotation={int(np.degrees(angle))} degrees")
    
    # Build vertical grasps by approaching along `height_axis`.
    print(f"  [Vertical] {num_vertical} points: lower-mid region, approach along -{height_axis} (top-down), every 90 degrees")

    # IMPORTANT (RoboTwin convention):
    # In Base_Task.get_grasp_pose(), the pre-grasp translation offset is effectively along the
    # contact frame's +Y axis (because of the fixed post-multiply matrix inside get_grasp_pose()).
    # Therefore for a true "vertical" pre-grasp (above/below), we must align contact +Y with height_axis,
    # instead of aligning contact +X.
    def _unit(axis: str) -> np.ndarray:
        if axis == "X":
            return np.array([1.0, 0.0, 0.0])
        if axis == "Y":
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])

    def _make_R_from_y(y_axis: np.ndarray) -> np.ndarray:
        """Build a right-handed rotation matrix with given +Y axis (columns are X,Y,Z)."""
        y = np.array(y_axis, dtype=float).reshape(3)
        y = y / (np.linalg.norm(y) + 1e-12)
        # pick a reference not parallel to y
        ref = np.array([1.0, 0.0, 0.0]) if abs(float(y[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = ref - y * float(np.dot(ref, y))
        x = x / (np.linalg.norm(x) + 1e-12)
        z = np.cross(x, y)
        z = z / (np.linalg.norm(z) + 1e-12)
        R = np.stack([x, y, z], axis=1)
        # ensure proper rotation
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        return R

    def _vertical_base_rotation() -> np.ndarray:
        """Vertical grasp base rotation: align contact frame +Y with +height_axis so pre-grasp is above."""
        return _make_R_from_y(_unit(height_axis))

    def _vertical_position(ratio: float) -> np.ndarray:
        ratio = float(ratio)
        ratio = max(0.0, min(1.0, ratio))
        pos = np.array([0.0, 0.0, 0.0], dtype=float)
        h_from_bottom = extents[height_axis_idx] * ratio
        h_from_center = h_from_bottom - extents[height_axis_idx] / 2
        pos[height_axis_idx] = h_from_center
        return pos

    # Generate vertical grasps by rotating around `height_axis`.
    pos_v = _vertical_position(vertical_ratio)
    rot_v_base = _vertical_base_rotation()
    
    for i in range(num_vertical):
        # Rotate around `height_axis`.
        angle = 2 * np.pi * i / num_vertical
        rotation_around_height = generate_rotation_matrix(rotation_axis, angle)
        rot_v = rotation_around_height @ rot_v_base
        
        T_v = np.eye(4)
        T_v[:3, :3] = rot_v
        T_v[:3, 3] = pos_v
        contact_points.append(T_v.tolist())
        print(
            f"    Point {num_horizontal + i}: vertical[{i}] ratio={float(vertical_ratio):.2f}, "
            f"rotation={int(np.degrees(angle))} degrees"
        )
    
    return contact_points


def generate_vertical_grasps(extents, height_axis="Y", num_points=2):
    """
    Generate vertical grasps, typically for flat objects.

    Args:
        extents: Object dimensions.
        height_axis: Thickness axis in mesh coordinates.
        num_points: 1 for top-only grasp, 2 for both top and bottom.
    """
    print(f"[Generate] Vertical grasp mode, {num_points} points")
    
    contact_points = []
    height_axis = str(height_axis).upper().strip()
    main_axis = axis_to_index(height_axis)
    thickness = extents[main_axis]
    offset = thickness * 0.3  # Slightly above the surface.
    
    # Top approach grasp.
    position_top = [0.0, 0.0, 0.0]
    position_top[main_axis] = offset
    
    if main_axis == 2:  # Thickness axis is Z.
        rotation_top = np.array([
            [1, 0, 0],   # X is horizontal.
            [0, 1, 0],   # Y is horizontal.
            [0, 0, 1]    # Z points upward.
        ])
    elif main_axis == 1:  # Thickness axis is Y.
        rotation_top = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
    else:  # Thickness axis is X.
        rotation_top = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    
    transform_top = np.eye(4)
    transform_top[:3, :3] = rotation_top
    transform_top[:3, 3] = position_top
    contact_points.append(transform_top.tolist())
    
    print(f"  Point 0: top approach, position={position_top}")
    
    # Optional bottom approach grasp.
    if num_points > 1:
        position_bottom = [0.0, 0.0, 0.0]
        position_bottom[main_axis] = -offset
        
        rotation_bottom = rotation_top @ np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        transform_bottom = np.eye(4)
        transform_bottom[:3, :3] = rotation_bottom
        transform_bottom[:3, 3] = position_bottom
        contact_points.append(transform_bottom.tolist())
        
        print(f"  Point 1: bottom approach, position={position_bottom}")
    
    return contact_points

def generate_obb_grasps(model_data, inset: float = 0.0, inset_ratio: float = 0.0):
    """
    Generate grasps using the oriented bounding box (OBB).

    Contact points are generated on opposite OBB faces and follow the RoboTwin
    convention where pre-grasp retreat is along contact frame +Y.
    """
    print("[Generate] OBB grasp mode")
    
    # Fall back to AABB extents when OBB metadata is unavailable.
    if 'aligned_obb_extents' not in model_data or 'obb_transform' not in model_data:
        print("[Warning] Missing OBB data, falling back to AABB data")
        aligned_obb_extents = np.array(model_data['extents'])
        obb_transform = np.eye(4)
    else:
        aligned_obb_extents = np.array(model_data['aligned_obb_extents'])
        obb_transform = np.array(model_data['obb_transform'])

    contact_points = []

    # Sort OBB axes by length: [short, mid, long].
    sorted_indices = np.argsort(aligned_obb_extents)
    axis_short = sorted_indices[0]
    axis_mid = sorted_indices[1]
    axis_long = sorted_indices[2]
    
    print(
        f"  [OBB analysis] local extents: short={aligned_obb_extents[axis_short]:.3f}m, "
        f"mid={aligned_obb_extents[axis_mid]:.3f}m, long={aligned_obb_extents[axis_long]:.3f}m"
    )

    def _safe_normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return v * 0.0
        return v / n

    def add_grasp(approach_axis, approach_dir, finger_axis, point_idx):
        """
        Build a grasp in OBB-local coordinates, then map it back to mesh coordinates.

        Args:
            approach_axis: Axis index (0/1/2) used as face normal.
            approach_dir: Side of the face (+1 or -1).
            finger_axis: Axis index for gripper finger/opening preference.
        """
        # Local translation along face normal with configurable inward inset.
        # Default policy:
        #   inset_abs = min(0.4 * T_axis, 0.06)
        # Explicit inset policy (when provided):
        #   inset_abs = max(inset, inset_ratio * T_axis)
        pos_local = np.array([0.0, 0.0, 0.0])
        half = float(aligned_obb_extents[approach_axis] / 2.0)
        T_axis = float(aligned_obb_extents[approach_axis])

        inset_val = float(inset)
        inset_ratio_val = float(inset_ratio)
        if inset_val <= 0.0 and inset_ratio_val <= 0.0:
            raw_inset = 0.4 * T_axis
            inset_abs = min(raw_inset, 0.06)  # Cap default inset to 6 cm.
        else:
            inset_abs = max(inset_val, inset_ratio_val * T_axis)

        inset_abs = max(0.0, min(inset_abs, half * 0.95))
        pos_local[approach_axis] = approach_dir * (half - inset_abs)
        
        # Build local rotation (columns are contact-frame X/Y/Z axes).
        # RoboTwin convention:
        # - Base_Task.get_grasp_pose retreats along contact +Y.
        # - Set contact +Y to face outward so retreat moves away from the surface.
        #
        # Alignment choice:
        # - grasp +X (approach) = -contact +Y
        # - grasp +Y (finger/open axis) = -contact +Z
        # Therefore:
        # - contact_y is the outward face normal.
        # - contact_z is an in-plane axis aligned with finger preference when possible.
        contact_y = np.zeros(3)
        contact_y[approach_axis] = float(approach_dir)  # outward
        contact_y = _safe_normalize(contact_y)

        finger_dir = np.zeros(3)
        finger_dir[finger_axis] = 1.0
        # If finger_axis equals approach_axis, use another in-plane axis.
        if finger_axis == approach_axis:
            for alt in (0, 1, 2):
                if alt != approach_axis:
                    finger_dir = np.zeros(3)
                    finger_dir[alt] = 1.0
                    break

        # Enforce grasp +Y = -contact +Z alignment with finger_dir.
        contact_z = -finger_dir
        contact_z = contact_z - contact_y * float(np.dot(contact_z, contact_y))
        contact_z = _safe_normalize(contact_z)

        contact_x = np.cross(contact_y, contact_z)
        contact_x = _safe_normalize(contact_x)

        # Re-orthogonalize to reduce numerical drift.
        contact_z = np.cross(contact_x, contact_y)
        contact_z = _safe_normalize(contact_z)

        R_local = np.column_stack((contact_x, contact_y, contact_z))
        # Ensure right-handed rotation matrix.
        if np.linalg.det(R_local) < 0:
            R_local[:, 2] *= -1.0
        
        T_local = np.eye(4)
        T_local[:3, :3] = R_local
        T_local[:3, 3] = pos_local
        
        # Map from OBB-local coordinates back to mesh coordinates.
        T_world = obb_transform @ T_local
        
        axis_names = {axis_short: "short", axis_mid: "mid", axis_long: "long"}
        contact_points.append(T_world.tolist())
        print(
            f"    Point {point_idx}: face normal={axis_names.get(approach_axis, approach_axis)}, "
            f"side={'+' if approach_dir>0 else '-'}; in-plane align={axis_names.get(finger_axis, finger_axis)}"
        )
        return True

    # Iterate over three face-normal axes (X/Y/Z). A face is considered graspable
    # if its shorter in-plane edge is within gripper width threshold.
    # Gripper width threshold used to decide whether a face is graspable.
    # NOTE: 0.08m is too tight for many thin objects (e.g. smartphone width ~0.081m),
    # and would incorrectly skip the natural "top-face" grasps. Use a slightly larger
    # default and leave a small epsilon for float noise.
    grasp_width_threshold_m = float(model_data.get("grasp_width_threshold_m", 0.085))
    eps = 1e-6
    candidate_face_axes = []
    for approach_axis in (0, 1, 2):
        in_plane_axes = [a for a in (0, 1, 2) if a != approach_axis]
        a0, a1 = int(in_plane_axes[0]), int(in_plane_axes[1])
        in_plane_lengths = np.array([float(aligned_obb_extents[a0]), float(aligned_obb_extents[a1])], dtype=float)
        min_edge_scaled = float(np.min(in_plane_lengths))
        finger_axis = int(in_plane_axes[int(np.argmin(in_plane_lengths))])
        ok = bool(min_edge_scaled <= grasp_width_threshold_m + eps)
        print(
            f"  [OBB filter] face-normal axis={approach_axis}, in-plane axes={in_plane_axes}, "
            f"in-plane extents={in_plane_lengths.tolist()}, min={min_edge_scaled:.4f}m -> {'keep' if ok else 'skip'}; "
            f"finger_axis={finger_axis}"
        )
        if ok:
            candidate_face_axes.append((int(approach_axis), finger_axis))

    # Fallback when no face passes threshold: use min-area and mid-area faces.
    if not candidate_face_axes:
        face_areas = np.array(
            [
                aligned_obb_extents[1] * aligned_obb_extents[2],  # Face normal X.
                aligned_obb_extents[0] * aligned_obb_extents[2],  # Face normal Y.
                aligned_obb_extents[0] * aligned_obb_extents[1],  # Face normal Z.
            ],
            dtype=float,
        )
        axis_order = np.argsort(face_areas)  # [min_area_axis, mid_area_axis, max_area_axis]
        min_axis = int(axis_order[0])
        mid_axis = int(axis_order[1])
        print(
            f"  [OBB analysis] face areas (m^2): X={float(face_areas[0]):.6f}, "
            f"Y={float(face_areas[1]):.6f}, Z={float(face_areas[2]):.6f}"
        )
        print(
            f"  [OBB fallback] no face meets threshold {grasp_width_threshold_m:.3f}m; "
            f"use min-area + mid-area faces: {min_axis}, {mid_axis}"
        )
        candidate_face_axes = [(min_axis, axis_short), (mid_axis, axis_short)]

    if float(inset) > 0.0 or float(inset_ratio) > 0.0:
        print(
            f"  [OBB params] inset={float(inset):.4f}m, inset_ratio={float(inset_ratio):.4f} "
            f"(inward offset along face normal)"
        )

    # Generate two grasps per selected face axis (positive/negative side).
    point_idx = 0
    for approach_axis, finger_axis in candidate_face_axes:
        for approach_dir in (1, -1):
            kept = add_grasp(int(approach_axis), int(approach_dir), int(finger_axis), point_idx)
            if kept:
                point_idx += 1
    
    return contact_points

def check_symmetry_by_rotation(mesh, angles = [0, 15, 40, 70]):
    """Estimate rotational symmetry from AABB area variation under rotation."""
    extents_list = []
    
    for angle in angles:
        # Rotate around the height axis.
        rot_mat = t3d.axangles.axangle2mat([0, 1, 0], angle)
        # Temporary transformed copy for extent estimation.
        temp_mesh = mesh.copy()
        temp_mesh.apply_transform(t3d.affines.compose([0,0,0], rot_mat, [1,1,1]))
        
        # Record projected X/Z extents on the horizontal plane.
        e = temp_mesh.extents
        extents_list.append([e[0], e[2]])
    
    extents_list = np.array(extents_list)
    # Variation of projected XZ area (std / mean).
    areas = extents_list[:, 0] * extents_list[:, 1]
    variation = np.std(areas) / np.mean(areas)
    
    return variation

def simplified_strategy_selector(mesh):
    e = mesh.extents

    # Small area variation (<5%) suggests strong rotational symmetry.
    variation = check_symmetry_by_rotation(mesh)
    
    if variation < 0.05:
        # Symmetric objects favor ring-style grasps.
        print(f"[Strategy] Use 'mixed' strategy")
        return 'mixed'
    else:
        # Asymmetric objects favor OBB face-based grasps.
        print(f"[Strategy] Use 'obb' strategy")
        return 'obb'    


def auto_generate_contact_points(
    model_data,
    strategy='mixed',
    num_points=None,
    height_ratio=0.5,
    height_axis="Y",
    vertical_ratio=0.15,
    num_vertical=4,
    obb_inset=0.0,
    obb_inset_ratio=0.0,
):
    """
    Generate contact points and write them back into `model_data`.

    Args:
        model_data: Dictionary containing object geometry metadata.
        strategy: One of `mixed`, `vertical`, or `obb`.
        num_points: Optional number of points (for mixed: horizontal point count).
        height_ratio: Horizontal grasp height ratio for `mixed` strategy.
        height_axis: Height axis in mesh coordinates (default: Y).
        vertical_ratio: Vertical grasp position ratio for `mixed` strategy.
        num_vertical: Number of vertical grasps in `mixed` strategy.
    """
    # IMPORTANT:
    # When transform_matrix is identity, existing grasps are known-good. So any fix must be gated on
    # transform_matrix != I and must not change default behavior.
    #
    # For thin objects we often set a non-identity model_data.transform_matrix to "redefine" the local
    # axes (e.g., thickness-as-up in mesh frame) and runtime applies this transform on the actor pose.
    # In that case, OBB-derived grasps MUST use an OBB transform consistent with the transformed local
    # frame; otherwise the contact frame axes (notably the in-plane finger/open axis) will be rotated
    # incorrectly even for top-down grasps.
    extents_print = np.array(model_data["extents"], dtype=float).reshape(3)
    print(f"[Analysis] Object extents: X={extents_print[0]:.3f}, Y={extents_print[1]:.3f}, Z={extents_print[2]:.3f}")

    # Generate grasps directly in mesh coordinates.
    print(f"\n[Strategy] Use '{strategy}' strategy")
    
    # If transform_matrix is non-identity, keep mixed/vertical as-is (they only use extents and a chosen
    # height_axis), but for OBB we must rotate the OBB transform into the transformed local frame.
    model_data_for_grasp = model_data
    try:
        T_tf = np.array(model_data.get("transform_matrix", np.eye(4)), dtype=float).reshape(4, 4)
        R_tf = T_tf[:3, :3]
        if np.max(np.abs(R_tf - np.eye(3))) > 1e-8:
            model_data_for_grasp = dict(model_data)

            # --- (1) Rotate OBB transform into the transformed local frame ---
            if isinstance(model_data.get("obb_transform", None), (list, tuple, np.ndarray)):
                T_obb = np.array(model_data["obb_transform"], dtype=float).reshape(4, 4)
                # OBB local -> mesh local. After redefining mesh local by T_tf, mapping becomes T_tf @ T_obb.
                model_data_for_grasp["obb_transform"] = (T_tf @ T_obb).tolist()

            # --- (2) Rotate OBB extents into the same axis convention ---
            # generate_obb_grasps uses aligned_obb_extents to decide:
            # - which axis is short/mid/long
            # - which faces are graspable
            # If we rotate obb_transform but keep aligned_obb_extents in the old axis order,
            # approach_axis can effectively "point to the wrong physical direction", producing
            # unexpected top-down vs side grasps and wrong finger/open axis.
            if isinstance(model_data.get("aligned_obb_extents", None), (list, tuple, np.ndarray)):
                a = np.array(model_data["aligned_obb_extents"], dtype=float).reshape(3)
                a2 = (np.abs(R_tf) @ a.reshape(3, 1)).reshape(3)
                model_data_for_grasp["aligned_obb_extents"] = a2.tolist()
            if isinstance(model_data.get("obb_extents", None), (list, tuple, np.ndarray)):
                e = np.array(model_data["obb_extents"], dtype=float).reshape(3)
                e2 = (np.abs(R_tf) @ e.reshape(3, 1)).reshape(3)
                model_data_for_grasp["obb_extents"] = e2.tolist()
    except Exception:
        model_data_for_grasp = model_data

    # Generate contact points.
    if strategy == 'mixed':
        n = num_points if num_points else 8  # Horizontal points, default 8.
        contact_points = generate_mixed_grasps(
            model_data["extents"],
            height_axis=height_axis,
            num_horizontal=n,
            height_ratio=height_ratio,
            vertical_ratio=vertical_ratio,
            num_vertical=num_vertical,
        )
    elif strategy == 'vertical':
        n = num_points if num_points else 2
        contact_points = generate_vertical_grasps(model_data["extents"], height_axis=height_axis, num_points=n)
    elif strategy == 'obb':                                 
        contact_points = generate_obb_grasps(
            model_data_for_grasp,
            inset=float(obb_inset),
            inset_ratio=float(obb_inset_ratio),
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Supported strategies: mixed, vertical, obb")
    
    # Update model_data.
    model_data['contact_points_pose'] = contact_points
    
    # Put all generated points into a single group by default.
    if len(contact_points) > 1:
        model_data['contact_points_group'] = [list(range(len(contact_points)))]
        model_data['contact_points_mask'] = [True]
    
    print(f"\n[Done] Generated {len(contact_points)} contact points")
    
    return model_data


def apply_dynamic_scale(
    extents: np.ndarray,
    urdf_scale: np.ndarray,
    enable: bool = True,
    min_axis_target: float = 0.075,
) -> np.ndarray:
    """
    Apply dynamic uniform scaling based on the shortest X/Z axis.

    If the shortest axis in X/Z is larger than `min_axis_target`, scale the
    model down so the shortest X/Z axis equals `min_axis_target`.

    Args:
        extents: Effective AABB dimensions [x, y, z] used for X/Z thresholding.
        urdf_scale: Scale vector defined in URDF.
        enable: Whether dynamic scaling is enabled.
        min_axis_target: Target shortest X/Z axis length in meters.

    Returns:
        Adjusted scale vector.
    """
    scale = np.array([1.0, 1.0, 1.0])
    
    if not enable:
        return scale
    
    # Dimensions after applying URDF scale.
    actual_dimensions_with_urdf = extents * urdf_scale
    # Use only X/Z for thresholding (height axis is excluded).
    axis_lengths_with_urdf = np.array(actual_dimensions_with_urdf, dtype=float).reshape(3)
    xz_lengths_with_urdf = axis_lengths_with_urdf[[0, 2]]
    min_xz_with_urdf = float(np.min(xz_lengths_with_urdf))
    min_xz_local_idx = int(np.argmin(xz_lengths_with_urdf))  # 0 -> X, 1 -> Z
    min_axis_idx = 0 if min_xz_local_idx == 0 else 2
    axis_names = {0: "X", 1: "Y", 2: "Z"}
    
    if min_xz_with_urdf > min_axis_target:
        # Uniform scale so the shortest X/Z axis equals target.
        scale_factor = min_axis_target / min_xz_with_urdf
        scale = np.array([scale_factor, scale_factor, scale_factor])
        print(
            f"[Dynamic scale] shortest XZ axis: {min_xz_with_urdf:.4f}m "
            f"({axis_names.get(min_axis_idx, str(min_axis_idx))}) > {min_axis_target:.4f}m"
        )
        print(
            f"[Dynamic scale] dimensions (with URDF scale): "
            f"X={actual_dimensions_with_urdf[0]:.4f}m, Y={actual_dimensions_with_urdf[1]:.4f}m, Z={actual_dimensions_with_urdf[2]:.4f}m"
        )
        print(f"[Dynamic scale] apply scale factor: {scale_factor:.4f}")
        print(f"[Dynamic scale] updated model_data scale: {scale.tolist()}")
        final_dimensions = extents * scale * urdf_scale
        print(f"[Dynamic scale] final dimensions (with URDF scale): {final_dimensions.tolist()}")
        final_xz = np.asarray(final_dimensions, dtype=float).reshape(3)[[0, 2]]
        print(f"[Dynamic scale] shortest XZ axis after scaling: {float(np.min(final_xz)):.4f}m")
    
    return scale


def create_model_data_from_mesh(object_dir, enable_dynamic_scale=True, min_axis_target=0.08):
    """
    Create `model_data.json` fields from mesh geometry.

    Args:
        object_dir: Object directory path.
        enable_dynamic_scale: Whether dynamic scaling is enabled.
        min_axis_target: Target shortest X/Z axis length in meters.
    """
    
    object_dir = Path(object_dir)
    
    # Search for mesh files.
    mesh_dir = object_dir / "mesh"
    mesh_file = None
    
    if mesh_dir.exists():
        for name in ["sample.obj", "sample.glb", "sample_collision.obj", "base.glb", "base.obj"]:
            candidate = mesh_dir / name
            if candidate.exists():
                mesh_file = candidate
                break
    
    if not mesh_file:
        # Fallback: search in object root.
        for name in ["sample.obj", "sample.glb", "base.glb", "base.obj"]:
            candidate = object_dir / name
            if candidate.exists():
                mesh_file = candidate
                break
    
    if not mesh_file:
        print(f"[Error] Mesh file not found (tried sample.obj, sample.glb, etc.)")
        return None
    
    print(f"[Analysis] Load mesh: {mesh_file}")
    
    try:
        # Load mesh.
        mesh = trimesh.load(str(mesh_file), force='mesh')
        
        # 1) Compute AABB for baseline metadata compatibility.
        bounds = mesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        extents = bounds[1] - bounds[0]
        
        # 2) Compute OBB (minimum-volume oriented bounding box).
        try:
            hull = mesh.convex_hull
            obb = hull.bounding_box_oriented
            # `obb.extents` are side lengths in OBB-local coordinates.
            obb_extents = obb.extents
            # `obb.primitive.transform` maps OBB-local -> mesh coordinates.
            obb_transform = obb.primitive.transform
            print(
                f"[Analysis] OBB computed successfully, local extents: "
                f"[{obb_extents[0]:.4f}, {obb_extents[1]:.4f}, {obb_extents[2]:.4f}]"
            )
        except Exception as e:
            print(f"[Warning] OBB computation failed, use AABB instead: {e}")
            obb_extents = extents
            obb_transform = np.eye(4)
            obb_transform[:3, 3] = center

        # 3) Compute aligned extents for scaling decisions.
        # aligned_extents are AABB projections on OBB axes:
        #   aligned_extents = |R_obb|^T @ extents
        R_obb = obb_transform[:3, :3]
        aligned_extents = np.abs(R_obb).T @ np.asarray(extents, dtype=float).reshape(3)
        print(f"[Analysis] Aligned extents: {aligned_extents.tolist()}")

        # Read URDF scale (used for final-size estimation only).
        urdf_scale = [1.0, 1.0, 1.0]
        urdf_path = find_urdf_in_dir(object_dir)
        if urdf_path:
            try:
                tree = ET.parse(urdf_path)
                root = tree.getroot()
                mesh_node = root.find(".//link/visual/geometry/mesh")
                if mesh_node is not None and mesh_node.get("scale"):
                    urdf_scale = [float(v) for v in mesh_node.get("scale").split()]
                    print(f"[Read] URDF scale: {urdf_scale}")
            except Exception as e:
                print(f"[Warning] Failed to read URDF: {e}")
        
        # Decide whether thin objects should be axis-aligned first, then compute
        # effective extents for dynamic X/Z scaling decisions.
        transform_matrix = maybe_align_thin_object_short_axis_to_height(
            raw_extents=extents,
            threshold=0.04,
            mesh_height_axis="Y",
        )
        R_thin = np.asarray(transform_matrix, dtype=float).reshape(4, 4)[:3, :3]
        extents_vec = np.asarray(extents, dtype=float).reshape(3)
        # After rotation R, mesh-frame AABB extents become:
        #   extents' = |R| @ extents
        extents_for_dynamic_scale = (np.abs(R_thin) @ extents_vec).reshape(3)
        if not np.allclose(R_thin, np.eye(3)):
            print(
                f"[Dynamic scale] Effective AABB after thin-object alignment (for XZ thresholding): "
                f"{extents_for_dynamic_scale.tolist()}"
            )

        urdf_scale_array = np.array(urdf_scale)
        scale_array = apply_dynamic_scale(
            extents_for_dynamic_scale,
            urdf_scale_array,
            enable=enable_dynamic_scale,
            min_axis_target=min_axis_target,
        )
        scale = scale_array.tolist()

        aligned_obb_extents = aligned_extents * scale

        # Build model_data.
        model_data = {
            "center": center.tolist(),
            "extents": extents.tolist(),
            "scale": scale,
            "transform_matrix": transform_matrix.tolist(),
            "target_pose": [],
            "contact_points_pose": [],
            "functional_matrix": [],
            "orientation_point": [],
            "contact_points_group": [],
            "contact_points_mask": [],
            "target_point_description": [],
            "contact_points_description": [],
            "functional_point_description": [],
            "orientation_point_description": [],
            "obb_extents": obb_extents.tolist(), 
            "aligned_obb_extents": aligned_obb_extents.tolist(),
            "obb_transform": obb_transform.tolist(),        
        }
        
        # Return (model_data, mesh) for optional auto strategy selection.
        return (model_data, mesh)
        
    except Exception as e:
        print(f"[Error] Failed to load mesh: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Automatically generate contact points')
    parser.add_argument('--object_dir', type=str, default=None,
                        help='Object instance directory path, e.g. our_assets/actor/coke/0')
    parser.add_argument('--object_class_dir', type=str, default=None,
                        help='Object class directory path, e.g. our_assets/actor/coke')
    parser.add_argument('--assets_branch_dir', type=str, default=None,
                        help='Branch directory under our_assets, e.g. our_assets/actor or our_assets/non-actor; process all classes and instances in that branch')
    parser.add_argument('--all', action='store_true',
                        help='Process all objects under our_assets/actor and our_assets/non-actor')
    parser.add_argument('--strategy', type=str, default= None,
                        choices=['mixed', 'vertical', 'obb', 'auto'],
                        help='Grasp strategy: mixed (8 horizontal + 4 vertical), vertical, or obb. auto is kept for backward compatibility (equivalent to mixed)')
    parser.add_argument('--num_points', type=int, default=None,
                        help='Number of points to generate (use default when omitted)')
    parser.add_argument('--height_ratio', type=float, default=0.6,
                        help='Horizontal grasp height ratio for mixed strategy (0.0-1.0; default 0.6)')
    parser.add_argument('--height_axis', type=str, default='Y',
                        choices=['X', 'Y', 'Z', 'x', 'y', 'z'],
                        help='Object height axis in mesh coordinates, default Y. Mixed strategy rotates around this axis to generate horizontal and vertical grasps.')
    parser.add_argument('--vertical_ratio', type=float, default=0.3,
                        help='Vertical grasp position ratio for mixed strategy along height_axis (from bottom to top, 0.0-1.0; default 0.3)')
    parser.add_argument('--num_vertical', type=int, default=4,
                        help='Number of vertical grasp points in mixed strategy (default 4, every 90 degrees)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (overwrite original file when omitted)')
    parser.add_argument('--disable_dynamic_scale', action='store_true',
                        help='Disable dynamic scaling (enabled by default). When enabled, auto-scales if the shorter axis in X/Z is larger than min_axis_target')
    parser.add_argument('--min_axis_target', type=float, default=0.07,
                        help='Target shortest axis length for dynamic scaling in meters (default 0.075). Only X and Z are considered')
    parser.add_argument(
        '--obb_inset',
        type=float,
        default=0.03,
        help='OBB strategy: inward offset distance along selected face normal (meters). Moves the point from face center toward the interior.',
    )
    parser.add_argument(
        '--obb_inset_ratio',
        type=float,
        default=0.5,
        help='OBB strategy: inward offset ratio relative to axis length (0-0.5). Final inset is max(obb_inset, obb_inset_ratio * extent).',
    )
    parser.add_argument(
        '--skip_existing_model_data',
        dest='skip_existing_model_data',
        action='store_true',
        default=True,
        help='Skip object if model_data.json already exists in the object directory (default enabled)',
    )
    parser.add_argument(
        '--no_skip_existing_model_data',
        dest='skip_existing_model_data',
        action='store_false',
        help='Do not skip objects with existing model_data.json; regenerate anyway',
    )
    
    args = parser.parse_args()
    
    # Resolve object directories to process.
    if args.all:
        roots = (
            Path("our_assets/actor"),
            Path("our_assets/non-actor"),
        )
        object_dirs: List[Path] = []
        for our_objects_dir in roots:
            if our_objects_dir.exists():
                object_dirs.extend(_collect_object_instance_dirs(our_objects_dir))
        if not object_dirs:
            print("[Error] No objects found in either our_assets/actor or our_assets/non-actor")
            return

    elif args.assets_branch_dir:
        branch_path = Path(args.assets_branch_dir)
        if not branch_path.exists():
            print(f"[Error] Branch directory does not exist: {args.assets_branch_dir}")
            return
        if not branch_path.is_dir():
            print(f"[Error] Not a directory: {args.assets_branch_dir}")
            return
        object_dirs = _collect_object_instance_dirs(branch_path)
        print(f"[Branch] Found {len(object_dirs)} object instance directories under {branch_path}")
        if not object_dirs:
            print(f"[Error] No objects found under {args.assets_branch_dir} (expected class/index instances or class directories containing meshes)")
            return

    elif args.object_class_dir:
        # Process all numbered instance subdirectories under the class directory.
        class_path = Path(args.object_class_dir)
        if not class_path.exists():
            print(f"[Error] Class directory does not exist: {args.object_class_dir}")
            return
        
        # Collect numeric instance directories.
        instance_subdirs = [p for p in class_path.iterdir() if p.is_dir() and p.name.isdigit()]
        if instance_subdirs:
            object_dirs = sorted(instance_subdirs, key=lambda p: int(p.name))
            print(f"[Class] Found {len(object_dirs)} instance directories under {class_path.name}")
        else:
            # If no numeric subdirectories exist, allow class directory itself if it contains meshes.
            if (class_path / "mesh").exists() or any(class_path.glob("*.obj")) or any(class_path.glob("*.glb")):
                object_dirs = [class_path]
            else:
                print(f"[Warning] No numbered instance subdirectories or mesh files found under {args.object_class_dir}")
                return

    elif args.object_dir:
        # Process one object directory.
        object_dirs = [Path(args.object_dir)]
    
    else:
        print("[Error] Please specify one of: --all, --assets_branch_dir, --object_class_dir, or --object_dir")
        return

    if not object_dirs:
        print(f"[Warning] No object directories to process")
        return
    
    # Process each object.
    success_count = 0
    skipped_count = 0
    failed_objects = []
    
    for object_dir in object_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {object_dir.name}")
        print(f"{'='*60}")
        
        try:
            # Load or auto-create model_data.json.
            model_data_path = object_dir / "model_data.json"
            if args.skip_existing_model_data and model_data_path.exists():
                print(f"[Skip] Already exists: {model_data_path}, skip regeneration")
                skipped_count += 1
                continue
            
            result = create_model_data_from_mesh(
                object_dir,
                enable_dynamic_scale=not args.disable_dynamic_scale,  # Enabled by default unless explicitly disabled.
                min_axis_target=args.min_axis_target,
            )
            if result is None:
                print(f"[Error] Failed to create model_data.json from mesh")
                failed_objects.append((object_dir.name, "Failed to create model_data.json"))
                continue
            model_data, mesh_from_creation = result
            print(f"[Success] Base model_data.json created automatically")
            # Validate required fields.
            if 'extents' not in model_data:
                print(f"[Error] 'extents' field is missing in model_data.json")
                failed_objects.append((object_dir.name, "Missing 'extents' field"))
                continue
            
            # Generate contact points.
            strategy = args.strategy 
            if not strategy or strategy == 'auto':
                # Use returned mesh when auto-created, otherwise use mesh loaded in model_data.
                mesh_for_strategy = model_data.get('mesh', mesh_from_creation)
                strategy = simplified_strategy_selector(mesh_for_strategy)
            model_data["strategy"] = strategy
            model_data = auto_generate_contact_points(
                model_data,
                strategy=strategy,
                num_points=args.num_points,
                height_ratio=args.height_ratio,
                height_axis=args.height_axis,
                vertical_ratio=args.vertical_ratio,
                num_vertical=args.num_vertical,
                obb_inset=args.obb_inset,
                obb_inset_ratio=args.obb_inset_ratio,
            )
            # Keep the chosen vertical ratio recorded in output for traceability
            model_data["vertical_ratio"] = float(args.vertical_ratio)
            # Keep the chosen height ratio recorded in output for traceability
            model_data["height_ratio"] = float(args.height_ratio)
            
            # Save output.
            output_path = Path(args.output) if args.output else model_data_path
            print(f"\n[Save] {output_path}")
            with open(output_path, 'w') as f:
                json.dump(model_data, f, indent=4)
            
            print(f"✅ {object_dir.name} completed!")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {object_dir.name} failed: {e}")
            failed_objects.append((object_dir.name, str(e)))
    
    # Print summary.
    print(f"\n{'='*60}")
    print(f"Batch processing completed")
    print(f"{'='*60}")
    print(f"✅ Success: {success_count}/{len(object_dirs)}")
    print(f"⏭️ Skipped: {skipped_count}/{len(object_dirs)}")
    
    if failed_objects:
        print(f"\n❌ Failed objects:")
        for name, reason in failed_objects:
            print(f"  - {name}: {reason}")
    
    if success_count > 0:
        print(f"\n📦 Next step:")
        print(
            "  Run collection: "
            "python script/collect_data_complete.py pick_up demo_randomized"
        )
        print(f"\n🔧 If any object result is not ideal, tune it individually:")
        print(f"  python script/auto_generate_contact_points.py --object_dir our_assets/actor/OBJECT_NAME/INSTANCE_ID --height_ratio 0.6")
        print(f"  python script/auto_generate_contact_points.py --object_dir our_assets/actor/OBJECT_NAME/INSTANCE_ID --strategy vertical")


if __name__ == "__main__":
    main()


