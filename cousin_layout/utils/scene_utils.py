# Portions of this file are adapted from Digital Cousins:
# https://github.com/cremebrule/digital-cousins
# Modified by RoboCousin contributors.
# Licensed under Apache-2.0; see licenses/Digital-Cousins-Apache-2.0.txt.
import numpy as np
import torch as th
import trimesh
from shapely.geometry import LineString, MultiPoint

from . import transform_utils as T
from .processing_utils import distance_to_plane, create_polygon_from_vertices
try:
    import omnigibson as og  # type: ignore
except Exception:
    og = None


def _xy_footprint_convex_hull_polygon(
    pts_xyz: np.ndarray,
    *,
    max_points: int = 8000,
):
    """
    Build a Shapely Polygon for the XY footprint as the convex hull of (filtered) 3D points.

    Uses 2–98% XY percentiles to limit outlier stretch, subsamples large clouds for speed.
    Returns None if too degenerate; caller should fall back to axis-aligned rectangle / bbox.
    """
    pts = np.asarray(pts_xyz, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], size=max_points, replace=False)]
    x1, x2 = float(np.percentile(pts[:, 0], 2.0)), float(np.percentile(pts[:, 0], 98.0))
    y1, y2 = float(np.percentile(pts[:, 1], 2.0)), float(np.percentile(pts[:, 1], 98.0))
    if not np.isfinite([x1, x2, y1, y2]).all() or (x2 <= x1) or (y2 <= y1):
        xy = pts[:, :2]
    else:
        m = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
        xy = pts[m, :2] if int(m.sum()) >= 3 else pts[:, :2]
    if xy.shape[0] < 3:
        return None
    hull = MultiPoint([tuple(p) for p in xy]).convex_hull
    if hull.geom_type == "Polygon" and hull.area > 0:
        return hull
    span = max(x2 - x1, y2 - y1, 1e-6)
    buf = max(span * 1e-5, 2e-4)
    if hull.geom_type == "LineString":
        g = hull.buffer(buf)
        return g if g.geom_type == "Polygon" and g.area > 0 else None
    if hull.geom_type == "Point":
        g = hull.buffer(buf)
        return g if g.geom_type == "Polygon" and g.area > 0 else None
    return None


def _ensure_nonzero_xy_extent(
    xmin: float, xmax: float, ymin: float, ymax: float, eps: float = 1e-4
) -> tuple[float, float, float, float]:
    """Avoid axis-aligned rects collapsing to a line/point in XY (breaks create_polygon_from_vertices)."""
    if not np.isfinite([xmin, xmax, ymin, ymax]).all():
        return xmin, xmax, ymin, ymax
    if xmax <= xmin:
        mid = 0.5 * (float(xmin) + float(xmax))
        xmin, xmax = mid - eps, mid + eps
    if ymax <= ymin:
        mid = 0.5 * (float(ymin) + float(ymax))
        ymin, ymax = mid - eps, mid + eps
    return xmin, xmax, ymin, ymax


def _widen_support_surface_footprint_xy(poly, buffer_m: float = 0.06):
    """
    Slightly enlarge desk/table/floor support footprints in XY (meters).

    Used only for support_surface candidates so edge-of-desk objects can overlap the support polygon
    without changing behavior for generic objects (books, laptops as supports, etc.).
    """
    if poly is None:
        return poly
    try:
        if getattr(poly, "is_empty", True):
            return poly
        g = poly.buffer(float(buffer_m))
        if g.geom_type == "Polygon":
            return g
        if g.geom_type == "MultiPolygon":
            return max(g.geoms, key=lambda p: float(p.area))
    except Exception:
        return poly
    return poly


def slice_points_world_xyz_for_support_footprint(
    pts_world: np.ndarray,
    footprint_slice_frac: float | None,
) -> np.ndarray:
    """
    Bottom vertical band of object points in world z (target lower contact patch in compute_object_z_offset).

    Pair with slice_points_world_xyz_for_top_footprint for support upper surfaces.

    - frac in [0, 1): bottom fraction of robust vertical range (2–98% z), with the same
      sparse-slice fallback as the polygon path.
    - frac >= 1 (including None -> 1.0): all points within z in [p2, p98].

    Returns finite 3D points; falls back to all finite input points if a slice is empty.
    """
    pts = np.asarray(pts_world, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts

    frac = float(footprint_slice_frac) if footprint_slice_frac is not None else 1.0
    if not np.isfinite(frac):
        frac = 1.0
    frac = float(np.clip(frac, 0.0, 1.0))

    if frac < 1.0:
        z_lo = float(np.percentile(pts[:, 2], 2.0))
        z_hi_full = float(np.percentile(pts[:, 2], 98.0))
        if np.isfinite(z_lo) and np.isfinite(z_hi_full) and z_hi_full > z_lo + 1e-9:
            height = max(z_hi_full - z_lo, 1e-9)
        else:
            z_lo = float(np.min(pts[:, 2]))
            z_hi_full = float(np.max(pts[:, 2]))
            height = max(z_hi_full - z_lo, 1e-9)
        z_hi = z_lo + frac * height
        pts_slice = pts[pts[:, 2] <= z_hi]
        if pts_slice.shape[0] < 3:
            order = np.argsort(pts[:, 2])
            k = int(np.ceil(frac * float(pts.shape[0])))
            k = int(np.clip(k, 3, pts.shape[0]))
            pts_slice = pts[order[:k]]
        return pts_slice if pts_slice.shape[0] > 0 else pts

    z_lo = float(np.percentile(pts[:, 2], 2.0))
    z_hi_full = float(np.percentile(pts[:, 2], 98.0))
    if np.isfinite(z_lo) and np.isfinite(z_hi_full) and z_hi_full > z_lo + 1e-9:
        pts_band = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi_full)]
    else:
        pts_band = pts
    return pts_band if pts_band.shape[0] > 0 else pts


def slice_points_world_xyz_for_top_footprint(
    pts_world: np.ndarray,
    top_slice_frac: float | None,
) -> np.ndarray:
    """
    Top portion of each object's point cloud in world z (mirror of slice_points_world_xyz_for_support_footprint).

    Used to approximate the **upper contact surface** of a would-be support object (e.g. desktop, not the whole
    desk volume). Pair with bottom-slice footprints when testing physical on-top relations.

    - frac in (0, 1): keep points in the top `frac` of the robust vertical range (2–98% z).
    - frac >= 1: all points within z in [p2, p98] (same band as the bottom-slicer full mode).
    """
    pts = np.asarray(pts_world, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return pts

    frac = float(top_slice_frac) if top_slice_frac is not None else 1.0
    if not np.isfinite(frac):
        frac = 1.0
    frac = float(np.clip(frac, 0.0, 1.0))

    if frac < 1.0:
        z_lo = float(np.percentile(pts[:, 2], 2.0))
        z_hi_full = float(np.percentile(pts[:, 2], 98.0))
        if np.isfinite(z_lo) and np.isfinite(z_hi_full) and z_hi_full > z_lo + 1e-9:
            height = max(z_hi_full - z_lo, 1e-9)
        else:
            z_lo = float(np.min(pts[:, 2]))
            z_hi_full = float(np.max(pts[:, 2]))
            height = max(z_hi_full - z_lo, 1e-9)
        z_lo_slice = z_hi_full - frac * height
        pts_slice = pts[pts[:, 2] >= z_lo_slice]
        if pts_slice.shape[0] < 3:
            order = np.argsort(pts[:, 2])
            k = int(np.ceil(frac * float(pts.shape[0])))
            k = int(np.clip(k, 3, pts.shape[0]))
            pts_slice = pts[order[-k:]]
        return pts_slice if pts_slice.shape[0] > 0 else pts

    z_lo = float(np.percentile(pts[:, 2], 2.0))
    z_hi_full = float(np.percentile(pts[:, 2], 98.0))
    if np.isfinite(z_lo) and np.isfinite(z_hi_full) and z_hi_full > z_lo + 1e-9:
        pts_band = pts[(pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi_full)]
    else:
        pts_band = pts
    return pts_band if pts_band.shape[0] > 0 else pts


def compute_relative_cam_pose_from(z_dir, origin_pos, init_quat=None):
    """
    Computes from z_dir origin_pos -> equivalent camera pose in OG world coordinates, assuming @z_dir, @origin_pos
    are specified in the camera frame where z points into the camera frame and the camera is located at the origin

    Args:
        z_dir (np.ndarray): (x,y,z) direction of the OG z-axis, expressed in the camera frame
        origin_pos (np.ndarray): (x,y,z) origin specified in the camera frame
        init_quat (None or np.ndarray): If specified, (x,y,z,w) initial quaternion of the OmniGibson camera.
            If None, will assume it is at (0, 0, 0, 1)

    Returns:
        2-tuple:
            - np.ndarray: (x,y,z) relative camera position in the OG world frame
            - np.ndarray: (x,y,z,w) relative camera quaternion orientation in the OG world frame
    """
    cam_quat = np.array([0.0, 0.0, 0.0, 1.0]) if init_quat is None else init_quat

    # Invert z_dir because OG camera points downwards wrt to a zero orientation
    z_dir = -z_dir

    # Get relative orientation -- this is simply the shortest rotation from z_dir to the vector defined by cam_rot @ [0, 0, 1].T
    cam_dir = T.quat2mat(cam_quat) @ np.array([0, 0, 1.0])

    # Get rotation direction via cross product
    rot_dir = np.cross(z_dir, cam_dir)
    rot_dir /= np.linalg.norm(rot_dir)

    # Get rotation magnitude via dot product
    rot_mag = np.arccos(np.dot(z_dir, cam_dir))
    rot_quat = T.axisangle2quat(rot_mag * rot_dir)

    rel_pos = T.axisangle2mat(rot_mag * rot_dir) @ origin_pos

    return rel_pos, rot_quat


def align_model_pose(
        obj,
        pc_obj,
        obj_z_angle,
        obj_ori_offset,
        z_dir,
        cam_pos,
        cam_quat,
        is_articulated,
        verbose=False,
):
    """
    Computes an object model's pose expressed in the OG world frame, given camera information, the object's model,
    and its corresponding point cloud

    Args:
        obj (DatasetObject): OmniGibson object to fit to the object point cloud
        pc_obj (np.ndarray): Point cloud to which @obj should be fit
        obj_z_angle (float): Object z orientation, applied after the object is adjusted via @obj_ori_offset to be
            upright
        obj_ori_offset (None or np.ndarray): If specified, the xyz-euler orientation offset that should be applied
            to @obj's canonical orientation so that it is semantically considered upright with its front face facing
            the OG world frame's +x axis
        z_dir (np.ndarray): (x,y,z) direction of the OG z-axis, expressed in the camera frame
        cam_pose (np.ndarray): (x,y,z) position of the camera in the OG world frame
        cam_quat (np.ndarray): (x,y,z,w) quaternion orientation of the camera in the OG world frame
        is_articulatd (bool): Whether the object is articulated or not
        verbose (bool): Whether to use verbose print out or not

    Returns:
        3-tuple:
            - np.ndarray: (x,y,z) scale of the object, fit to the corresponding point cloud
            - np.ndarray: (x,y,z) AABB extent of the object, fit to the corresponding point cloud
            - np.ndarray: (4,4) homogeneous pose matrix representating the relative pose from @cam_pose, @cam_quat to
                @obj such that it is fit to the corresponding point cloud
    """
    # Make sure sim is playing
    assert og.sim.is_playing()

    # Get tilt angle -- atan2(y / z)
    tilt_angle = np.arctan2(z_dir[1], z_dir[2])

    # Rotate pc_obj, first by tilt angle, then by nn_z_pose angle
    # This is a rotation first about the image frame X axis, then the image frame Z axis
    tilt_mat = T.euler2mat([tilt_angle, 0, 0])
    z_rot_mat = T.euler2mat([0, 0, -obj_z_angle])
    pc_obj_rot = pc_obj @ tilt_mat.T @ z_rot_mat.T

    # Refine pose
    oriented_2d_rot = np.eye(3)
    oriented_2d_rot[:2, :2] = trimesh.bounds.oriented_bounds_2D(pc_obj_rot[:, :2])[0][:2, :2]
    z_offset = T.mat2euler(oriented_2d_rot)[2]
    # Rotate to min range of (-45, 45) degrees, since bounding box can always be rotated 90 deg and still be at the min
    # See https://stackoverflow.com/a/2323034
    lim = np.pi / 2
    z_offset = (((z_offset % lim) + lim) % lim)
    z_offset = z_offset - lim if z_offset > lim / 2 else z_offset

    # Define threshold for refining orientation -- we'll "snap" to a specific orientation given the threshold
    # Depends on whether the object is articulated or not
    # In general, non-articulated object point clouds are much more noisy, so we use a smaller threshold for them
    refine_angle_threshold = 20 if is_articulated else 15
    if abs(z_offset * 180 / np.pi) <= refine_angle_threshold:
        z_refine_rot_mat = T.euler2mat([0, 0, z_offset])
        pc_obj_rot_refined = pc_obj_rot @ z_refine_rot_mat.T

        if verbose:
            print(f"Finetuned {obj.name}'s z-rotation by {z_offset * 180 / np.pi} degrees")
        pc_obj_rot = pc_obj_rot_refined

        # Update z-rot angle
        obj_z_angle -= z_offset
        z_rot_mat = T.euler2mat( [0, 0, -obj_z_angle])

    # Calculate bbox extents of rotated pc
    input_obj_aabb = pc_obj_rot.min(axis=0), pc_obj_rot.max(axis=0)
    # X and Y in image frame are flipped compared to the OG simulator's
    input_obj_aabb_extent = input_obj_aabb[1] - input_obj_aabb[0]
    input_obj_aabb_center_rot = (input_obj_aabb[1] + input_obj_aabb[0]) / 2

    # Compute input obj bbox center -- first in image frame, then convert to OG global frame
    input_obj_aabb_center = tilt_mat.T @ z_rot_mat.T @ input_obj_aabb_center_rot
    og_cam_local_tf = T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
    og_cam_global_tf = T.pose2mat((cam_pos, cam_quat))
    obj_bbox_pos = (og_cam_global_tf @ og_cam_local_tf @ np.array([*input_obj_aabb_center, 1.0]))[:3]

    # Get nn cab bbox and re-scale so matches the computed aabb_extents
    obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float), th.tensor([0, 0, 0, 1], dtype=th.float))
    obj.keep_still()
    og.sim.step_physics()
    obj_aabb_extent = obj.aabb_extent
    scale_factor = input_obj_aabb_extent / obj_aabb_extent
    og.sim.stop()
    obj_scale = obj.scale * scale_factor
    obj_bbox_quat = T.mat2quat(og_cam_global_tf[:3, :3] @ og_cam_local_tf[:3, :3] @ tilt_mat.T @ z_rot_mat.T)

    # Update scale, play, then set pose
    obj.scale = obj_scale
    og.sim.play()
    obj.set_position_orientation(th.tensor([0, 0, 0], dtype=th.float), th.tensor([0, 0, 0, 1], dtype=th.float))
    og.sim.step_physics()
    obj_bbox_extent = obj.aabb_extent.cpu().detach().numpy()
    obj.set_bbox_center_position_orientation(th.tensor(obj_bbox_pos, dtype=th.float), th.tensor(obj_bbox_quat, dtype=th.float))
    og.sim.step_physics()

    # Make sure all object's lower face is parallel to the floor
    _, bbox_orn_in_world, _, _ = obj.get_base_aligned_bbox()
    current_orientation = bbox_orn_in_world.cpu().detach().numpy()  # Current orientation as a quaternion
    desired_z_axis = np.array([0, 0, 1.0])
    current_z_axis = T.quat2mat(current_orientation)[:, 2]

    # Angle between the current z-axis and the desired z-axis
    angle = np.arccos(np.clip(
        np.dot(current_z_axis, desired_z_axis) / (np.linalg.norm(current_z_axis) * np.linalg.norm(desired_z_axis)),
        -1.0, 1.0))
    # Axis for the rotation (cross product of the two vectors)
    axis = np.cross(current_z_axis, desired_z_axis)
    if np.linalg.norm(axis) < 1e-6:
        # If the axis length is close to zero, the axes are aligned or directly opposite.
        # Check if they are aligned (angle close to zero) -- if nontrivial angle, then fix
        if angle > 1e-6:
            # 180 degrees flip, choose any perpendicular axis
            axis = np.array([1, 0, 0])  # Arbitrary axis perpendicular to z
            # Rotation quaternion
            alignment_quaternion = T.axisangle2quat(axis * angle)

            # New orientation is the product of alignment quaternion and current orientation
            new_orientation = T.quat_multiply(alignment_quaternion, current_orientation)

            # Convert quaternion to tuple format (x, y, z, w) and set as new orientation
            obj.set_position_orientation(orientation=th.tensor(new_orientation, dtype=th.float))

    # Apply orientation offset as well
    if obj_ori_offset is not None:
        ori = T.quat2euler(obj.get_position_orientation()[1].cpu().detach().numpy())
        ori[0] += obj_ori_offset[0]
        ori[1] += obj_ori_offset[1]
        obj.set_position_orientation(orientation=T.euler2quat(th.tensor(ori, dtype=th.float)))

    # Take one step to apply updates
    og.sim.step_physics()

    # Calculate relative transformation
    obj_pos, obj_quat = obj.get_position_orientation()
    tf_from_cam = T.pose2mat(T.relative_pose_transform(
        obj_pos.cpu().detach().numpy(),
        obj_quat.cpu().detach().numpy(),
        cam_pos,
        cam_quat,
    ))

    return obj_scale.cpu().detach().numpy(), obj_bbox_extent, tf_from_cam


def align_obj_with_wall(
        obj,
        cam_pos,
        cam_quat,
        wall_normal,
        wall_point,
        wall_is_vertical=True,
        resize_only=False,
):
    '''
    Adjust the orientation of the object and resize it to align with the wall specified by @wall_plane

    Parameters:
        obj (DatasetObject): The object to reorient and resize according to the wall
        cam_pose (np.ndarray): (x,y,z) position of the camera in the OG world frame
        cam_quat (np.ndarray): (x,y,z,w) quaternion orientation of the camera in the OG world frame
        wall_normal (np.ndarray): (x,y,z) normal direction of the wall plane expressed in the camera frame
        wall_point (np.ndarray): (x,y,z) mean point of the wall plane point cloud expressed in the camera frame
        wall_is_vertical (bool): If True, will assume the wall is vertical and "snap" the wall plane to the nearest
            upright orientation (i.e.: no slanted walls will occur)
        resize_only (bool): If True, only resize the object without reorienting it. This is useful if an object is
            touching multiple walls, and therefore should only have one wall to reorient it

    Returns:
        3-tuple:
            - np.ndarray: (x,y,z) scale of the object, fit to the corresponding wall
            - np.ndarray: (x,y,z) AABB extent of the object, fit to the corresponding wall
            - np.ndarray: (4,4) homogeneous pose matrix representating the relative pose from @cam_pose, @cam_quat to
                @obj such that it is fit to the corresponding wall
    '''
    # Make sure sim is playing
    assert og.sim.is_playing()

    # OmniGibson camera always is rotated 180 deg wrt the x-axis (-z into camera frame),
    # compared to the image camera convention (+z into camera frame)
    og_cam_ori_offset = T.euler2mat([np.pi, 0, 0])

    # Transform the normal vector
    wall_z_dir_og_frame = T.quat2mat(cam_quat) @ og_cam_ori_offset @ wall_normal

    # Calculate the point on the plane in input camera's frame and transform it
    og_cam_local_tf = T.pose2mat(([0, 0, 0], T.mat2quat(og_cam_ori_offset)))
    og_cam_global_tf = T.pose2mat((cam_pos, cam_quat))
    p_og_frame = (og_cam_global_tf @ og_cam_local_tf @ np.array([*wall_point, 1.0]))[:3]

    # Select the wall norm that points from the wall toward viewer cam (all objects)
    p2cam_vec = cam_pos - p_og_frame

    # Flip z direction if it's facing away fromt the camera
    if np.dot(p2cam_vec, wall_z_dir_og_frame) <= 0:
        wall_z_dir_og_frame = -wall_z_dir_og_frame

    if wall_is_vertical:
        # We snap the wall direction to the nearest horizontal direction (i.e.: zero out z-values)
        # This means that we assume all walls are perfectly vertical, with no slant
        wall_z_dir_og_frame[-1] = 0
        wall_z_dir_og_frame = wall_z_dir_og_frame / np.linalg.norm(wall_z_dir_og_frame)

    # Compute new d' for the plane in OG viewer camera's frame
    d_og_frame = -np.dot(wall_z_dir_og_frame, p_og_frame)

    # Infer robot x and y axis -- this is merely the 3 columns of the orientation matrix
    obj_quat = obj.get_position_orientation()[1]
    obj_ori_mat = T.quat2mat(obj_quat.cpu().detach().numpy())
    obj_x_dir = obj_ori_mat[:, 0]
    obj_y_dir = obj_ori_mat[:, 1]

    # Determine whether to align x-axis or y-axis of object with wall normal -- this will be done based on the min
    # between the two dot products
    if abs(np.dot(obj_x_dir, wall_z_dir_og_frame)) > abs(np.dot(obj_y_dir, wall_z_dir_og_frame)):
        # align x axis with wall normal
        chosen_axis, other_axis, align_axis_idx = obj_x_dir, obj_y_dir, 0
    else:
        # align y axis with wall normal
        chosen_axis, other_axis, align_axis_idx = obj_y_dir, obj_x_dir, 1

    # Re-orient if requested
    if not resize_only:
        # Compute rotation axis and angle
        rotation_angle = np.arccos(np.dot(chosen_axis, wall_z_dir_og_frame))
        # Create rotation matrix from axis-angle
        new_obj_ori_euler = T.mat2euler(obj_ori_mat)
        # Infer offset sign for modifying orientation
        offset_sign = (-1.0) ** (align_axis_idx)        # + for x, - for y
        if rotation_angle > np.pi / 2:  # x axis points to opposite dir with norm
            rotation_angle = np.pi - rotation_angle
            if np.dot(other_axis, wall_z_dir_og_frame) > 0:
                offset_sign *= -1
        else:
            if np.dot(other_axis, wall_z_dir_og_frame) < 0:
                offset_sign *= -1
        new_obj_ori_euler[-1] += rotation_angle * offset_sign

        # Update object orientation and chosen axis
        obj_quat = T.euler2quat(new_obj_ori_euler)
        obj.set_position_orientation(orientation=th.tensor(obj_quat, dtype=th.float))
        og.sim.step_physics()
        chosen_axis = T.euler2mat(new_obj_ori_euler)[:, align_axis_idx]

    # Update object scale and update object's pose based on wall info
    center = obj.aabb_center.cpu().detach().numpy()
    dist2wall = distance_to_plane(center, [*wall_z_dir_og_frame, d_og_frame])
    obj_scale = obj.scale
    chosen_axis_scale = obj_scale[align_axis_idx]
    bbox_center_in_world, bbox_orn_in_world, bbox_extent_in_desired_frame, bbox_center_in_desired_frame = obj.get_base_aligned_bbox()
    chosen_axis_extent = bbox_extent_in_desired_frame[align_axis_idx]
    extent_needed = chosen_axis_extent / 2.0 + dist2wall
    scale_per_extent = chosen_axis_scale / chosen_axis_extent
    scale_needed = extent_needed * scale_per_extent
    obj_scale[align_axis_idx] = scale_needed

    center_step_size = (extent_needed / 2.0 - chosen_axis_extent / 2.0).item()
    center2wallpoint = p_og_frame - center
    # Chosen_axis point to the wall if dot product is positive, otherwise points away from wall
    center_step_dir = chosen_axis if np.dot(chosen_axis, center2wallpoint) > 0 else -chosen_axis

    new_center = center + center_step_dir * center_step_size

    # Update scale
    with og.sim.stopped():
        obj.scale = obj_scale

    # Update object pose
    obj.set_position_orientation(th.tensor([0.0, 0.0, 0.0], dtype=th.float), th.tensor([0, 0, 0, 1], dtype=th.float))
    og.sim.step_physics()
    obj_bbox_extent = obj.aabb_extent.cpu().detach().numpy()
    obj.set_bbox_center_position_orientation(th.tensor(new_center, dtype=th.float), th.tensor(obj_quat, dtype=th.float))

    # Calculate relative transformation
    obj_pos, obj_quat = obj.get_position_orientation()
    tf_from_cam = T.pose2mat(T.relative_pose_transform(
        obj_pos.cpu().detach().numpy(),
        obj_quat.cpu().detach().numpy(),
        cam_pos,
        cam_quat,
    ))
    og.sim.step_physics()
    return obj_scale.cpu().detach().numpy(), obj_bbox_extent, tf_from_cam


def compute_object_z_offset(
    target_obj_name,
    sorted_obj_bbox_info,
    verbose=False,
    footprint_slice_frac: float = 1.0,
    contact_z_tolerance: float = 0.05,
):
    '''
    Computes object below @target_obj_name and the necessary z-offset required to place the target object on top of
    below object's bounding box.

    On-top uses **support top surface × target bottom surface** in XY, not the overlap of whole-object footprints:
      - Target: bottom vertical band of points (lower contact patch).
      - Candidate support: top vertical band (e.g. desktop, floor top), not the full desk volume.
    The median z of the two bands must agree within @contact_z_tolerance (meters), otherwise the pair is rejected
    (e.g. PC tower under desk can still overlap the desktop in XY projection, but z separates the surfaces).

    If ``sorted_obj_bbox_info[name]["support_surface"]`` is True (desk/table/floor from Step-1), the support
    footprint is widened (larger top band, XY buffer, wider z window for the close-slab heuristic, and a
    slightly lower overlap fraction vs. the target) so edge-of-desk objects still match the desktop.

    Args:
        target_obj_name (str): Name of the object that we want to move to avoid "sinking into" another object or
            "floating" in the air during scene matching
        sorted_obj_bbox_info (dict): a dictionary storing all objects' lower_corner, bbox center, and upper_corner.
            Should be sorted (from lowest to highest z-value) before passing in
        verbose (bool): Whether to use verbose print outs or not
        footprint_slice_frac (float): Band thickness as a fraction of robust height (2–98% z). If >= 1, uses 0.2.
        contact_z_tolerance (float): Max |z_target_bottom_median - z_support_top_median| for a valid contact (meters).

    Returns:
        2-tuple:
            - str: Name of the object beneath the target object
            - float: Z-offset value to apply to the target object such that it lies on the object below
    '''
    all_obj_names = list(sorted_obj_bbox_info.keys())
    target_obj_idx = all_obj_names.index(target_obj_name)
    obj_names_to_check = all_obj_names[:target_obj_idx]

    f = float(footprint_slice_frac) if footprint_slice_frac is not None else 1.0
    if not np.isfinite(f):
        f = 1.0
    f = float(np.clip(f, 0.0, 1.0))
    f_contact = 0.2 if f >= 1.0 - 1e-9 else f

    def _polygon_xy_from_contact_band(info: dict, *, use_top: bool, widen_support_top_band: bool = False):
        """XY polygon for bottom contact (target) or top contact (support)."""
        pts_world = info.get("points_world_xyz", None)
        if pts_world is not None:
            pts = np.asarray(pts_world, dtype=float)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                pts = pts[np.isfinite(pts).all(axis=1)]
                if pts.shape[0] > 0:
                    if use_top and widen_support_top_band:
                        f_top = min(1.0, max(float(f_contact), 0.42))
                    else:
                        f_top = f_contact
                    subset = (
                        slice_points_world_xyz_for_top_footprint(pts, f_top)
                        if use_top
                        else slice_points_world_xyz_for_support_footprint(pts, f_contact)
                    )
                    if subset.shape[0] < 3:
                        subset = pts
                    z_ref = float(np.median(subset[:, 2]))
                    hull_poly = _xy_footprint_convex_hull_polygon(subset)
                    if hull_poly is not None and hull_poly.area > 1e-12:
                        return hull_poly, z_ref
                    xmin = float(np.percentile(subset[:, 0], 2.0))
                    xmax = float(np.percentile(subset[:, 0], 98.0))
                    ymin = float(np.percentile(subset[:, 1], 2.0))
                    ymax = float(np.percentile(subset[:, 1], 98.0))
                    if (not np.isfinite([xmin, xmax, ymin, ymax]).all()) or (xmax <= xmin) or (ymax <= ymin):
                        xmin = float(np.min(subset[:, 0]))
                        xmax = float(np.max(subset[:, 0]))
                        ymin = float(np.min(subset[:, 1]))
                        ymax = float(np.max(subset[:, 1]))
                    xmin, xmax, ymin, ymax = _ensure_nonzero_xy_extent(xmin, xmax, ymin, ymax)
                    rect_xy = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
                    return create_polygon_from_vertices(rect_xy), z_ref

        bbox_bottom_desired_frame = info["bbox_bottom_in_desired_frame"]
        bbox_top_desired_frame = info["bbox_top_in_desired_frame"]
        frame_to_world = info["desired_frame_to_world"]
        face = np.asarray(bbox_top_desired_frame if use_top else bbox_bottom_desired_frame, dtype=float)
        vw = trimesh.transformations.transform_points(face, frame_to_world)
        z_ref = float(np.median(vw[:, 2]))
        verts_xy = [tuple(pt[:2]) for pt in vw]
        v2 = np.asarray(verts_xy, dtype=float).reshape(-1, 2)
        v2 = v2[np.isfinite(v2).all(axis=1)]
        if v2.shape[0] == 0:
            return create_polygon_from_vertices(verts_xy), z_ref
        u = np.unique(v2, axis=0)
        if u.shape[0] < 4:
            xmin = float(np.min(u[:, 0]))
            xmax = float(np.max(u[:, 0]))
            ymin = float(np.min(u[:, 1]))
            ymax = float(np.max(u[:, 1]))
            xmin, xmax, ymin, ymax = _ensure_nonzero_xy_extent(xmin, xmax, ymin, ymax)
            rect_xy = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
            return create_polygon_from_vertices(rect_xy), z_ref
        return create_polygon_from_vertices(verts_xy), z_ref

    polygon_tar, z_tar_ref = _polygon_xy_from_contact_band(sorted_obj_bbox_info[target_obj_name], use_top=False)
    _, _, target_bbox_low_z = sorted_obj_bbox_info[target_obj_name]["lower"]

    def _polygon_xy_support_best_near_target_z(info: dict, target_z: float):
        """
        Support contact patch selection for candidate supports.

        Problem: for objects like laptops / monitors, the 'top band' can be dominated by tall vertical
        structures (e.g. a screen). That makes z_ref too high and rejects real contacts (phone on laptop base).

        Strategy:
        - First, try points within a z-window around target_z (if enough points).
        - Otherwise, compute both top-band and bottom-band contact patches and pick whichever z_ref is closer
          to target_z.

        For ``support_surface`` (desk/table/floor from Step-1 graph): widen the top band, use a slightly
        larger z window, buffer the XY footprint, and relax overlap thresholds in the outer loop — so
        objects near the desk edge still match the desktop.
        """
        is_support_surface = bool(info.get("support_surface"))
        z_win_scale = 3.0 if is_support_surface else 2.0

        pts_world = info.get("points_world_xyz", None)
        if pts_world is not None:
            pts = np.asarray(pts_world, dtype=float)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                pts = pts[np.isfinite(pts).all(axis=1)]
                if pts.shape[0] > 0:
                    tol = float(contact_z_tolerance) * z_win_scale
                    close = pts[np.abs(pts[:, 2] - float(target_z)) <= tol]
                    if close.shape[0] >= 10:
                        z_ref = float(np.median(close[:, 2]))
                        hull_poly = _xy_footprint_convex_hull_polygon(close)
                        if hull_poly is not None and hull_poly.area > 1e-12:
                            if is_support_surface:
                                # For desk/table/floor, use AABB instead of convex hull so
                                # objects at corners are not missed due to hull diagonal cutoff.
                                # Use min/max (not percentile) to cover the full observed extent.
                                xmin_c = float(np.min(close[:, 0]))
                                xmax_c = float(np.max(close[:, 0]))
                                ymin_c = float(np.min(close[:, 1]))
                                ymax_c = float(np.max(close[:, 1]))
                                xmin_c, xmax_c, ymin_c, ymax_c = _ensure_nonzero_xy_extent(xmin_c, xmax_c, ymin_c, ymax_c)
                                rect_poly = create_polygon_from_vertices([(xmin_c, ymin_c), (xmin_c, ymax_c), (xmax_c, ymax_c), (xmax_c, ymin_c)])
                                poly = _widen_support_surface_footprint_xy(rect_poly)
                            else:
                                poly = hull_poly
                            return poly, z_ref
                        xmin = float(np.percentile(close[:, 0], 2.0))
                        xmax = float(np.percentile(close[:, 0], 98.0))
                        ymin = float(np.percentile(close[:, 1], 2.0))
                        ymax = float(np.percentile(close[:, 1], 98.0))
                        if (not np.isfinite([xmin, xmax, ymin, ymax]).all()) or (xmax <= xmin) or (ymax <= ymin):
                            xmin = float(np.min(close[:, 0]))
                            xmax = float(np.max(close[:, 0]))
                            ymin = float(np.min(close[:, 1]))
                            ymax = float(np.max(close[:, 1]))
                        xmin, xmax, ymin, ymax = _ensure_nonzero_xy_extent(xmin, xmax, ymin, ymax)
                        rect_xy = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
                        poly = create_polygon_from_vertices(rect_xy)
                        if is_support_surface:
                            poly = _widen_support_surface_footprint_xy(poly)
                        return poly, z_ref

        poly_top, z_top = _polygon_xy_from_contact_band(
            info, use_top=True, widen_support_top_band=is_support_surface
        )
        poly_bot, z_bot = _polygon_xy_from_contact_band(info, use_top=False, widen_support_top_band=False)
        if abs(float(z_bot) - float(target_z)) <= abs(float(z_top) - float(target_z)):
            poly, z_ref = poly_bot, z_bot
        else:
            poly, z_ref = poly_top, z_top
        if is_support_surface:
            poly = _widen_support_surface_footprint_xy(poly)
        return poly, z_ref

    obj_name_beneath = "floor"
    for check_obj_name in reversed(obj_names_to_check):  # Start from larger z value
        polygon_sup, z_sup_ref = _polygon_xy_support_best_near_target_z(
            sorted_obj_bbox_info[check_obj_name],
            target_z=z_tar_ref,
        )
        dz = abs(z_tar_ref - z_sup_ref)
        if dz > float(contact_z_tolerance):
            if verbose:
                print(
                    f"{target_obj_name} vs {check_obj_name}: contact z gap |dz|={dz:.3f}m "
                    f"(target bottom-band z~{z_tar_ref:.3f}, support top-band z~{z_sup_ref:.3f}) "
                    f"> tol={contact_z_tolerance:.3f}; skip."
                )
            continue

        intersection_area_threshold = 0.3 if sorted_obj_bbox_info[check_obj_name]["articulated"] and not sorted_obj_bbox_info[target_obj_name]["articulated"] else 0.5
        if sorted_obj_bbox_info[check_obj_name].get("support_surface"):
            intersection_area_threshold = min(float(intersection_area_threshold), 0.28)
        if polygon_tar.intersects(polygon_sup):
            intersect_area = polygon_tar.intersection(polygon_sup).area
            if verbose:
                print(f"{target_obj_name} intersects with {check_obj_name}")
                print(f"target area: {polygon_tar.area}, candidate area: {polygon_sup.area}, intersect_area: {intersect_area}")
            if intersect_area / max(polygon_tar.area, 1e-12) >= intersection_area_threshold or intersect_area / max(polygon_sup.area, 1e-12) >= intersection_area_threshold:
                if verbose:
                    print(f"Detected that {target_obj_name} is on top of {check_obj_name}!")
                obj_name_beneath = check_obj_name
                z_offset = z_sup_ref - z_tar_ref
                break

    # If none of the objects are beneath the target object,
    # set z_offset to -target_obj_low_z to put the object on the ground
    if obj_name_beneath == "floor":
        z_offset = -float(target_bbox_low_z)

    return obj_name_beneath, z_offset


def compute_obj_bbox_info(obj):
    """
    Computes object bbox information

    Args:
        obj (DatasetObject): Object whose bbox info should be computed

    Returns:
        dict: Keyword-mapped information about the object's bbox
    """
    obj_pos, obj_quat = obj.get_position_orientation()
    desired_frame_to_world = T.pose2mat((obj_pos.cpu().detach().numpy(), obj_quat.cpu().detach().numpy()))
    lower_corner, upper_corner = obj.aabb
    center = obj.aabb_center
    bbox_center_in_world, bbox_orn_in_world, bbox_extent_in_desired_frame, bbox_center_in_desired_frame = obj.get_base_aligned_bbox()

    bottom_corner_0 = (bbox_center_in_desired_frame[0] - bbox_extent_in_desired_frame[0] / 2.0,
                       bbox_center_in_desired_frame[1] - bbox_extent_in_desired_frame[1] / 2.0,
                       bbox_center_in_desired_frame[2] - bbox_extent_in_desired_frame[2] / 2.0)  # bottom left
    bottom_corner_1 = (bbox_center_in_desired_frame[0] + bbox_extent_in_desired_frame[0] / 2.0,
                       bbox_center_in_desired_frame[1] - bbox_extent_in_desired_frame[1] / 2.0,
                       bbox_center_in_desired_frame[2] - bbox_extent_in_desired_frame[2] / 2.0)  # bottom right
    bottom_corner_2 = (bbox_center_in_desired_frame[0] + bbox_extent_in_desired_frame[0] / 2.0,
                       bbox_center_in_desired_frame[1] + bbox_extent_in_desired_frame[1] / 2.0,
                       bbox_center_in_desired_frame[2] - bbox_extent_in_desired_frame[2] / 2.0)  # top right
    bottom_corner_3 = (bbox_center_in_desired_frame[0] - bbox_extent_in_desired_frame[0] / 2.0,
                       bbox_center_in_desired_frame[1] + bbox_extent_in_desired_frame[1] / 2.0,
                       bbox_center_in_desired_frame[2] - bbox_extent_in_desired_frame[2] / 2.0)  # top left
    bbox_bottom_in_desired_frame = np.array([bottom_corner_0, bottom_corner_1, bottom_corner_2, bottom_corner_3])

    top_corner_0 = (bbox_center_in_desired_frame[0] - bbox_extent_in_desired_frame[0] / 2.0,
                    bbox_center_in_desired_frame[1] - bbox_extent_in_desired_frame[1] / 2.0,
                    bbox_center_in_desired_frame[2] + bbox_extent_in_desired_frame[2] / 2.0)  # bottom left
    top_corner_1 = (bbox_center_in_desired_frame[0] + bbox_extent_in_desired_frame[0] / 2.0,
                    bbox_center_in_desired_frame[1] - bbox_extent_in_desired_frame[1] / 2.0,
                    bbox_center_in_desired_frame[2] + bbox_extent_in_desired_frame[2] / 2.0)  # bottom right
    top_corner_2 = (bbox_center_in_desired_frame[0] + bbox_extent_in_desired_frame[0] / 2.0,
                    bbox_center_in_desired_frame[1] + bbox_extent_in_desired_frame[1] / 2.0,
                    bbox_center_in_desired_frame[2] + bbox_extent_in_desired_frame[2] / 2.0)  # top right
    top_corner_3 = (bbox_center_in_desired_frame[0] - bbox_extent_in_desired_frame[0] / 2.0,
                    bbox_center_in_desired_frame[1] + bbox_extent_in_desired_frame[1] / 2.0,
                    bbox_center_in_desired_frame[2] + bbox_extent_in_desired_frame[2] / 2.0)  # top left
    bbox_top_in_desired_frame = np.array([top_corner_0, top_corner_1, top_corner_2, top_corner_3])
    obj_bbox_info = {
        "lower": lower_corner.cpu().detach().numpy(),
        "center": center.cpu().detach().numpy(),
        "upper": upper_corner.cpu().detach().numpy(),
        "bbox_bottom_in_desired_frame": bbox_bottom_in_desired_frame,
        "bbox_top_in_desired_frame": bbox_top_in_desired_frame,
        "desired_frame_to_world": desired_frame_to_world,
    }

    return obj_bbox_info

def get_vis_cam_trajectory(center_pos, cam_pos, cam_quat, d_tilt, radius, n_steps):
    '''
    Generate camera trajectory to visualize a scene.

    Args:
        center_pos (List or np.ndarray): Center position where the camera rotates around to visualize the scene.
        cam_pos (List or np.ndarray): Starting camera postion.
        cam_quat (List or np.ndarray): Starting camera orientation.
        d_tilt (float): Camera tilt angle (in degree) when visualizing the scene.
        radius (float): The radius of the circle that the camera rotates around.
        n_steps (int): The number of steps of the trajectory.

    Returns:
        List: A list of [position, quaternion] representing camera trajectory.

    '''
    center_pos = np.array(center_pos, dtype=float)
    cam_pos = np.array(cam_pos, dtype=float)
    # Keep orbit in an XY plane at the camera's height (avoid strict equality / float issues).
    center_pos[-1] = cam_pos[-1]
    # Generates trajectory commands to rotate camera 360 degrees about a specific point
    initial_roll_angle, initial_tilt_angle, initial_pan_angle = T.quat2euler(cam_quat)
    resulting_roll = initial_roll_angle - np.radians(d_tilt)

    d_z_total = radius * np.tan(np.radians(d_tilt))
    
    cmds = []
    center2cam = cam_pos - center_pos
    center2cam_norm = float(np.linalg.norm(center2cam))
    # Degenerate case: center == camera. Pick an arbitrary XY direction to define the orbit.
    if (not np.isfinite(center2cam_norm)) or center2cam_norm < 1e-9:
        center2cam_unit = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        center2cam_unit = center2cam / center2cam_norm
        # Project onto XY plane for stability.
        center2cam_unit[-1] = 0.0
        xy = float(np.linalg.norm(center2cam_unit[:2]))
        center2cam_unit = center2cam_unit if xy < 1e-9 else center2cam_unit / xy
    rotate_starting_pos = np.array(center_pos) + radius * center2cam_unit + np.array([0, 0, d_z_total]) # starting position of the circle
    new_center = np.array(center_pos) + np.array([0, 0, d_z_total])
    cam2start = rotate_starting_pos - np.array(cam_pos)
    center2start = rotate_starting_pos - np.array(new_center)
    center2start_norm = float(np.linalg.norm(center2start))
    if (not np.isfinite(center2start_norm)) or center2start_norm < 1e-9:
        center2start_unit = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        center2start_unit = center2start / center2start_norm
        center2start_unit[-1] = 0.0
        xy_norm = float(np.linalg.norm(center2start_unit[:2]))
        center2start_unit = center2start_unit if xy_norm < 1e-9 else center2start_unit / xy_norm
    ortho_center2start_unit = np.array([-center2start_unit[1], center2start_unit[0], 0])
    step_vec = cam2start / 5.0
    for i in range(5):
        add_tilt = i * np.radians(d_tilt) / 5
        pos = np.array(cam_pos) + i * step_vec
        quat = T.euler2quat((initial_roll_angle - add_tilt, initial_tilt_angle, initial_pan_angle))
        cmds.append([pos, quat])

    for i in range(n_steps):
        # Compute offset
        pan_angle = i * 2 * np.pi / n_steps
        # Get orientation and pos
        dx_value = radius * np.cos(pan_angle)
        dy_value = radius * np.sin(pan_angle)
        dx = dx_value * center2start_unit   # 3D vec
        dy = dy_value * ortho_center2start_unit # 3D vec
        offset = dx + dy
        pos = np.array(new_center) + offset
        quat = T.euler2quat((resulting_roll, initial_tilt_angle, initial_pan_angle + pan_angle)) 
        cmds.append([pos, quat])

    end_pos = pos
    end_quat = quat
    end_roll_angle, _, _ = T.quat2euler(end_quat)
    end2cam = np.array(cam_pos) - end_pos
    step_vec = end2cam / 5.0
    roll_diff = initial_roll_angle - end_roll_angle
    for i in range(5):
        add_roll = (i + 1) * roll_diff / 5.0
        pos = end_pos + (i + 1) * step_vec
        quat = T.euler2quat((end_roll_angle + add_roll, initial_tilt_angle, initial_pan_angle))
        cmds.append([pos, quat])

    return cmds