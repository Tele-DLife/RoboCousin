"""
Specified (cousin) coordinates for data collection.

When --use-cousin-coordinate is passed to collect_data_complete.py:
- Task objects (e.g. basket): use get_placement_coordinates() in task load_actors.
- Cluttered table objects: use get_clutter_placements() in Base_Task.get_cluttered_table.
- pick_up grasp target: also emitted by get_clutter_placements (same stacking as preview), with
  ``spawn_as_actor_only=True``; load_actors only runs strict precheck.

Coordinate logic lives here.

We can also map a "relative layout" JSON (from digital-cousins) into RoboTwin table XY.
"""

from __future__ import annotations

import json
import os
import random
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Keep empty by default. Prefer passing layout path via:
# - env: COUSIN_RELATIVE_LAYOUT_JSON
# - task config: cousin_relative_layout_json (handled by entry scripts)
DEFAULT_RELATIVE_LAYOUT_PATH = ""

# RoboTwin default table size (meters). Keep consistent with envs/_base_task.py:create_table_and_wall
ROBOTWIN_TABLE_LENGTH_M = 1.2
ROBOTWIN_TABLE_WIDTH_M = 0.7

# How many leading entries from each layout object's `instance_score_rank` to sample
# randomly from (when that field is non-empty). Config key: cousin_instance_score_pool_top_k
DEFAULT_COUSIN_INSTANCE_SCORE_POOL_TOP_K = 3


def _norm_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def resolve_custom_asset_class_dir(
    repo_root: Path,
    class_name: str,
    *,
    actor_only: bool = False,
) -> Optional[Path]:
    """Resolve class folder: ``our_assets/actor`` then (unless *actor_only*) ``our_assets/non-actor``."""
    if not isinstance(class_name, str) or not class_name.strip():
        return None
    cls = class_name.strip()
    actor = repo_root / "our_assets" / "actor" / cls
    if actor.is_dir():
        return actor
    if actor_only:
        return None
    non = repo_root / "our_assets" / "non-actor" / cls
    if non.is_dir():
        return non
    return None


def _layout_rotation_snapshot_step_degrees(layout: Dict[str, Any]) -> float:
    """Snapshot yaw step from layout root (required; no default)."""
    v = layout.get("rotation_snapshot_step_degrees")
    if v is None:
        raise RuntimeError(
            "relative layout json missing top-level 'rotation_snapshot_step_degrees'"
        )
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"rotation_snapshot_step_degrees must be a number, got {v!r}"
        ) from e


def _model_dir_from_match(repo_root: Path, match: Any, *, context: str) -> Path:
    """Resolve ``class_path`` from a cousin match dict; must be under ``repo/our_assets``."""
    if not isinstance(match, dict):
        raise RuntimeError(f"{context}: match must be an object, not {type(match).__name__}")
    cp = match.get("class_path")
    if not isinstance(cp, str) or not cp.strip():
        raise RuntimeError(f"{context}: missing or empty class_path")
    p = Path(cp).expanduser().resolve()
    rr = repo_root.resolve()
    assets_root = rr / "our_assets"
    try:
        p.relative_to(assets_root)
    except ValueError as e:
        raise RuntimeError(
            f"{context}: class_path {p} must be under {assets_root}"
        ) from e
    if not p.is_dir():
        raise RuntimeError(f"{context}: class_path is not a directory: {p}")
    return p


def _require_combined_match(o: Dict[str, Any], name: str) -> Dict[str, Any]:
    m = o.get("our_objects_combined_match")
    if m is None:
        raise RuntimeError(
            f"cousin layout object {name!r}: our_objects_combined_match is null "
            "(required for clutter)"
        )
    if not isinstance(m, dict):
        raise RuntimeError(
            f"cousin layout object {name!r}: our_objects_combined_match must be an object"
        )
    return m


def _require_actor_only_match(o: Dict[str, Any], label: str) -> Dict[str, Any]:
    m = o.get("our_objects_actor_only_match")
    if m is None:
        raise RuntimeError(
            f"cousin layout label {label!r}: our_objects_actor_only_match is null "
            "(required for actor / grasp target)"
        )
    if not isinstance(m, dict):
        raise RuntimeError(
            f"cousin layout label {label!r}: our_objects_actor_only_match must be an object"
        )
    return m


def _find_matching_our_object_dir(
    repo_root: Path,
    label: str,
    strict: bool = False,
    *,
    actor_only: bool = False,
) -> Optional[Path]:
    """
    Resolve class directory from relative-layout JSON by label.

    - ``actor_only=True``: ``our_objects_actor_only_match.class_path`` (raises if null when strict).
    - ``actor_only=False``: ``our_objects_combined_match.class_path``.
    """
    layout_path = os.getenv("COUSIN_RELATIVE_LAYOUT_JSON", DEFAULT_RELATIVE_LAYOUT_PATH)
    if not layout_path or not os.path.exists(layout_path):
        if strict:
            raise RuntimeError(f"relative layout json not found: {layout_path}")
        return None

    try:
        layout = _load_relative_layout_json(layout_path)
    except Exception as e:
        if strict:
            raise RuntimeError(f"failed to load relative layout json: {e}") from e
        return None

    objs = layout.get("objects", []) or []
    selected = set((layout.get("filter", {}) or {}).get("selected_names", []) or [])
    if selected:
        objs = [o for o in objs if o.get("name") in selected]

    target_norm = _norm_label(label)
    if not target_norm:
        if strict:
            raise RuntimeError("empty label for layout lookup")
        return None

    for o in objs:
        if _norm_label(str(o.get("label", ""))) != target_norm:
            continue
        if not isinstance(o, dict):
            continue
        try:
            if actor_only:
                m = _require_actor_only_match(o, label)
                ctx = f"our_objects_actor_only_match (label={label!r})"
            else:
                m = _require_combined_match(o, str(o.get("name", "")))
                ctx = f"our_objects_combined_match (name={o.get('name')!r})"
            return _model_dir_from_match(repo_root, m, context=ctx)
        except RuntimeError:
            if strict:
                raise
            return None

    if strict:
        raise RuntimeError(f"label '{label}' not found in relative layout json objects")
    return None


def _get_layout_object_for_label(label: str) -> Optional[Dict[str, Any]]:
    """Return the first layout object dict whose label matches (same filter semantics as other readers)."""
    layout_path = os.getenv("COUSIN_RELATIVE_LAYOUT_JSON", DEFAULT_RELATIVE_LAYOUT_PATH)
    if not layout_path or not os.path.exists(layout_path):
        return None
    try:
        layout = _load_relative_layout_json(layout_path)
    except Exception:
        return None
    objs = layout.get("objects", []) or []
    selected = set((layout.get("filter", {}) or {}).get("selected_names", []) or [])
    if selected:
        objs = [o for o in objs if o.get("name") in selected]
    target_norm = _norm_label(label)
    if not target_norm:
        return None
    for o in objs:
        if _norm_label(str(o.get("label", ""))) == target_norm:
            return o if isinstance(o, dict) else None
    return None


def _list_instance_dir_names(model_dir: Path) -> List[str]:
    instance_dirs = [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not instance_dirs:
        return ["0"]
    return [p.name for p in sorted(instance_dirs, key=lambda p: int(p.name))]


def _filter_allowed_instance_ids(
    all_names: List[str],
    allowed_instance_ids: Any,
) -> List[str]:
    if allowed_instance_ids is None:
        return list(all_names)
    if isinstance(allowed_instance_ids, (list, tuple, set)):
        allowed_set = {str(x).strip() for x in allowed_instance_ids if str(x).strip()}
        return [n for n in all_names if n in allowed_set]
    return list(all_names)


def _resolve_allowed_instance_ids_for_label(
    allowed_map: Any,
    label: str,
) -> Optional[Any]:
    if not isinstance(allowed_map, dict):
        return None
    label_norm = _norm_label(label)
    if label in allowed_map:
        return allowed_map[label]
    for key, value in allowed_map.items():
        if _norm_label(str(key)) == label_norm:
            return value
    return allowed_map.get("default")


def _parse_instance_score_rank_entry(entry: Any) -> Optional[str]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        v = entry.get("instance_id", entry.get("id"))
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None
    s = str(entry).strip()
    return s if s else None


def _pick_random_instance_id(
    model_dir: Path,
    rng: random.Random,
    layout_obj: Optional[Dict[str, Any]] = None,
    instance_score_pool_top_k: int = DEFAULT_COUSIN_INSTANCE_SCORE_POOL_TOP_K,
    *,
    uniform_sampling: bool = False,
    allowed_instance_ids: Any = None,
) -> str:
    """
    Custom asset layout (under ``our_assets/actor`` or ``our_assets/non-actor``):
      ``<repo>/our_assets/<bucket>/<name>/<id>/`` with URDF + model_data.

    When ``uniform_sampling`` is True, ignore similarity rank and sample uniformly from
  ``allowed_instance_ids`` (or all instance folders when unset).

    Otherwise, when `layout_obj` provides a non-empty `instance_score_rank` list, randomly
    pick among the first min(pool_top_k, len(rank)) entries that exist as instance
    directories under `model_dir`. When the field is missing or empty, pick uniformly
    among all numeric instance directories (previous behavior).
    """
    all_names = _list_instance_dir_names(model_dir)
    pool = _filter_allowed_instance_ids(all_names, allowed_instance_ids)
    if not pool:
        pool = all_names

    if uniform_sampling:
        return rng.choice(pool)

    # digital-cousins exports instance_score_rank under rotation_estimation by default:
    #   rotation_estimation: { ..., instance_score_rank: [...] }
    # Some legacy / hand-authored layouts may also place it at the object top-level.
    rank_raw = None
    if layout_obj:
        rank_raw = layout_obj.get("instance_score_rank")
        if not rank_raw:
            rot_est = layout_obj.get("rotation_estimation") or {}
            if isinstance(rot_est, dict):
                rank_raw = rot_est.get("instance_score_rank")
    if not rank_raw or not isinstance(rank_raw, (list, tuple)) or len(rank_raw) == 0:
        return rng.choice(pool)

    top_k = max(1, int(instance_score_pool_top_k))
    slice_end = min(top_k, len(rank_raw))
    ranked_pool: List[str] = []
    seen: set = set()
    for i in range(slice_end):
        eid = _parse_instance_score_rank_entry(rank_raw[i])
        if not eid or eid in seen:
            continue
        if eid not in pool:
            continue
        if (model_dir / eid).is_dir():
            ranked_pool.append(eid)
            seen.add(eid)

    if ranked_pool:
        return rng.choice(ranked_pool)
    return rng.choice(pool)


def _load_model_data_extent_scale_y(model_dir: Path, instance_id: str) -> Optional[Tuple[float, float]]:
    """
    Load model_data.json for the given object instance and return (extents[1], scale[1]).
    extents[1] is the model's thickness (height axis); scale[1] is scale_y.
    Returns None if file missing or invalid.
    """
    instance_dir = model_dir / instance_id if (model_dir / instance_id).is_dir() else model_dir
    model_cfg = instance_dir / "model_data.json"
    if not model_cfg.exists():
        for f in instance_dir.glob("model_data*.json"):
            model_cfg = f
            break
    if not model_cfg.exists():
        return None
    try:
        with open(model_cfg, "r", encoding="utf-8") as f:
            config = json.load(f)
        extents = config.get("extents", [0.1, 0.1, 0.1])
        scale = config.get("scale", 1.0)
        if isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        if len(extents) < 2 or len(scale) < 2:
            return None
        return float(extents[1]), float(scale[1])
    except Exception:
        return None


def _rel_uv_to_robotwin_xy(
    u: float,
    v: float,
    table_xy_bias: Tuple[float, float] = (0.0, 0.0),
    table_length_m: float = ROBOTWIN_TABLE_LENGTH_M,
    table_width_m: float = ROBOTWIN_TABLE_WIDTH_M,
    region_axes: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    Convert region_position_rel = (u,v) in [0,1] into RoboTwin world XY on table.

    Default matches the provided JSON:
      region.axes = {x: right, y: up, origin: bottom_left}
    """
    ax = region_axes if isinstance(region_axes, dict) else {}
    x_axis = str(ax.get("x", "right")).lower()
    y_axis = str(ax.get("y", "up")).lower()
    origin = str(ax.get("origin", "bottom_left")).lower()

    uu, vv = float(u), float(v)

    # origin handling
    if origin in ("bottom_right", "br"):
        uu = 1.0 - uu
    elif origin in ("top_left", "tl"):
        vv = 1.0 - vv
    elif origin in ("top_right", "tr"):
        uu = 1.0 - uu
        vv = 1.0 - vv

    # axis direction handling
    if x_axis == "left":
        uu = 1.0 - uu
    if y_axis == "down":
        vv = 1.0 - vv

    x_local = (uu - 0.5) * float(table_length_m)
    y_local = (vv - 0.5) * float(table_width_m)
    return float(table_xy_bias[0] + x_local), float(table_xy_bias[1] + y_local)


    x_local = (uu - 0.5) * float(table_length_m)
    y_local = (vv - 0.5) * float(table_width_m)
    return float(table_xy_bias[0] + x_local), float(table_xy_bias[1] + y_local)


def _robotwin_xy_to_rel_uv(
    x: float,
    y: float,
    table_xy_bias: Tuple[float, float] = (0.0, 0.0),
    table_length_m: float = ROBOTWIN_TABLE_LENGTH_M,
    table_width_m: float = ROBOTWIN_TABLE_WIDTH_M,
    region_axes: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Inverse of ``_rel_uv_to_robotwin_xy`` for default region axes."""
    ax = region_axes if isinstance(region_axes, dict) else {}
    x_axis = str(ax.get("x", "right")).lower()
    y_axis = str(ax.get("y", "up")).lower()
    origin = str(ax.get("origin", "bottom_left")).lower()

    x_local = float(x) - float(table_xy_bias[0])
    y_local = float(y) - float(table_xy_bias[1])
    uu = x_local / float(table_length_m) + 0.5
    vv = y_local / float(table_width_m) + 0.5

    if origin in ("bottom_right", "br"):
        uu = 1.0 - uu
    elif origin in ("top_left", "tl"):
        vv = 1.0 - vv
    elif origin in ("top_right", "tr"):
        uu = 1.0 - uu
        vv = 1.0 - vv

    if x_axis == "left":
        uu = 1.0 - uu
    if y_axis == "down":
        vv = 1.0 - vv

    return float(uu), float(vv)


def _horizontal_footprint_radius_m(instance_dir: Path) -> float:
    """Conservative horizontal footprint radius from model_data extents (x,z,y order)."""
    model_cfg = instance_dir / "model_data.json"
    if not model_cfg.exists():
        for f in instance_dir.glob("model_data*.json"):
            model_cfg = f
            break
    if not model_cfg.exists():
        return 0.05
    try:
        with open(model_cfg, "r", encoding="utf-8") as f:
            config = json.load(f)
        extents = config.get("extents", [0.1, 0.1, 0.1])
        scale = config.get("scale", 1.0)
        if isinstance(scale, (int, float)):
            scale = [scale, scale, scale]
        if len(extents) < 3 or len(scale) < 3:
            return 0.05
        half_x = float(extents[0]) * float(scale[0]) * 0.5
        half_y = float(extents[2]) * float(scale[2]) * 0.5
        return float(math.hypot(half_x, half_y))
    except Exception:
        return 0.05


def _max_horizontal_footprint_radius_for_class(
    repo_root: Path,
    class_name: str,
    *,
    allowed_instance_ids: Any = None,
    actor_only: bool = True,
) -> float:
    class_dir = resolve_custom_asset_class_dir(repo_root, class_name, actor_only=actor_only)
    if class_dir is None and actor_only:
        class_dir = resolve_custom_asset_class_dir(repo_root, class_name, actor_only=False)
    if class_dir is None:
        return 0.05
    radii = []
    for iid in _filter_allowed_instance_ids(
        _list_instance_dir_names(class_dir),
        allowed_instance_ids,
    ):
        inst_dir = class_dir / iid
        if inst_dir.is_dir():
            radii.append(_horizontal_footprint_radius_m(inst_dir))
    return max(radii) if radii else 0.05


def _build_our_objects_match_block(
    repo_root: Path,
    class_name: str,
    *,
    allowed_instance_ids: Any = None,
    actor_only: bool = True,
) -> Dict[str, Any]:
    class_dir = resolve_custom_asset_class_dir(repo_root, class_name, actor_only=actor_only)
    if class_dir is None and actor_only:
        class_dir = resolve_custom_asset_class_dir(repo_root, class_name, actor_only=False)
    if class_dir is None:
        raise RuntimeError(f"our_assets class not found for label {class_name!r}")
    instance_ids = _filter_allowed_instance_ids(
        _list_instance_dir_names(class_dir),
        allowed_instance_ids,
    )
    instances = [
        {"instance_id": iid, "best_snapshot_index": 0}
        for iid in instance_ids
    ]
    return {
        "class_name": class_name,
        "class_path": str(class_dir.resolve()),
        "instances": instances,
    }


def _label_to_layout_instance_name(label: str, idx: int = 0) -> str:
    return f"{_norm_label(label).replace(' ', '_')}_{int(idx)}"


def _xy_fits_table_bounds(
    x: float,
    y: float,
    radius: float,
    table_xlim: Tuple[float, float],
    table_ylim: Tuple[float, float],
    edge_padding: float = 0.0,
) -> bool:
    pad = float(edge_padding) + float(radius)
    return (
        table_xlim[0] + pad <= float(x) <= table_xlim[1] - pad
        and table_ylim[0] + pad <= float(y) <= table_ylim[1] - pad
    )


def prepare_anchor_relative_layout(
    *,
    repo_root: Path,
    sample_anchor_xy,
    target_label: str,
    clutter_specs: List[Dict[str, Any]],
    table_xy_bias: Tuple[float, float] = (0.0, 0.0),
    table_xlim: Tuple[float, float] = (-0.59, 0.59),
    table_ylim: Tuple[float, float] = (-0.34, 0.34),
    episode_idx: int = 0,
    cousin_instance_indices: Any = None,
    edge_padding: float = 0.02,
    max_tries: int = 200,
    rng: Optional[random.Random] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[str, Tuple[float, float]]:
    """
    Sample an anchor XY (via ``sample_anchor_xy``), place clutter labels at fixed offsets,
    and write a temporary cousin relative layout JSON.

    Resamples until every object fits inside ``table_xlim`` / ``table_ylim`` (with footprint).
    """
    if rng is None:
        rng = random.Random(0)

    target_label = str(target_label).strip()
    if not target_label:
        raise ValueError("target_label must be non-empty")

    radii: Dict[str, float] = {
        target_label: _max_horizontal_footprint_radius_for_class(
            repo_root,
            target_label,
            allowed_instance_ids=_resolve_allowed_instance_ids_for_label(
                cousin_instance_indices, target_label
            ),
            actor_only=True,
        )
    }
    normalized_clutter: List[Tuple[str, float, float]] = []
    for spec in clutter_specs or []:
        if not isinstance(spec, dict):
            continue
        label = str(spec.get("label", "")).strip()
        offset = spec.get("offset_xy", spec.get("offset", None))
        if not label or not isinstance(offset, (list, tuple)) or len(offset) < 2:
            continue
        dx, dy = float(offset[0]), float(offset[1])
        radii[label] = _max_horizontal_footprint_radius_for_class(
            repo_root,
            label,
            allowed_instance_ids=_resolve_allowed_instance_ids_for_label(
                cousin_instance_indices, label
            ),
            actor_only=False,
        )
        normalized_clutter.append((label, dx, dy))

    anchor_xy: Optional[Tuple[float, float]] = None
    placements_xy: List[Tuple[str, float, float]] = []
    for _ in range(max(1, int(max_tries))):
        ax, ay = sample_anchor_xy()
        candidate = [(target_label, float(ax), float(ay))]
        ok = _xy_fits_table_bounds(
            ax, ay, radii[target_label], table_xlim, table_ylim, edge_padding
        )
        for label, dx, dy in normalized_clutter:
            cx = float(ax) + dx
            cy = float(ay) + dy
            if not _xy_fits_table_bounds(
                cx, cy, radii[label], table_xlim, table_ylim, edge_padding
            ):
                ok = False
                break
            candidate.append((label, cx, cy))
        if ok:
            anchor_xy = (float(ax), float(ay))
            placements_xy = candidate
            break

    if anchor_xy is None:
        raise RuntimeError(
            f"[cousin_anchor_layout] failed to sample in-bounds layout after {max_tries} tries "
            f"(target={target_label!r}, clutter={[x[0] for x in normalized_clutter]})"
        )

    region_axes = {"x": "right", "y": "up", "origin": "bottom_left"}
    objects: List[Dict[str, Any]] = []
    for idx, (label, px, py) in enumerate(placements_xy):
        u, v = _robotwin_xy_to_rel_uv(
            px,
            py,
            table_xy_bias=table_xy_bias,
            region_axes=region_axes,
        )
        allowed_ids = _resolve_allowed_instance_ids_for_label(cousin_instance_indices, label)
        match = _build_our_objects_match_block(
            repo_root,
            label,
            allowed_instance_ids=allowed_ids,
            actor_only=(label == target_label),
        )
        obj_entry: Dict[str, Any] = {
            "name": _label_to_layout_instance_name(label, idx),
            "label": label,
            "region_position_rel": [float(u), float(v)],
            "our_objects_combined_match": match,
        }
        if label == target_label:
            obj_entry["our_objects_actor_only_match"] = dict(match)
        objects.append(obj_entry)

    layout = {
        "source": {"mode": "cousin_anchor_layout"},
        "region": {
            "a": ROBOTWIN_TABLE_LENGTH_M,
            "b": ROBOTWIN_TABLE_WIDTH_M,
            "axes": region_axes,
        },
        "objects": objects,
        "ontop": {"base_names": [], "levels": []},
        "rotation_snapshot_step_degrees": 22.5,
    }

    if output_dir is None:
        output_dir = Path("/tmp/robocousin_anchor_layouts")
    output_dir.mkdir(parents=True, exist_ok=True)
    layout_path = output_dir / f"anchor_layout_ep{int(episode_idx)}.json"
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    print(
        "[CousinAnchor] wrote temporary layout "
        f"{layout_path} anchor_xy=({anchor_xy[0]:.3f}, {anchor_xy[1]:.3f})"
    )
    return str(layout_path), anchor_xy


def _load_relative_layout_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _layout_cam_horizontal_azimuth_rad(layout: Optional[Dict[str, Any]]) -> float:
    """
    Horizontal azimuth (radians) of the snapshot camera direction, consistent with
    digital-cousins ``snapshot_generation._compute_eye_from_cam_pose_world``:
    world (z-up) -> render (y-up): p = [x_w, z_w, -y_w], horizontal = [p[0], 0, p[2]].

    This is **per layout file** (each scene / COUSIN_RELATIVE_LAYOUT_JSON has its own camera).
    A single constant in YAML cannot replace this across scenes.
    """
    if not isinstance(layout, dict):
        return 0.0
    try:
        pos = (layout.get("source", {}) or {}).get("cam_pose_world", {}).get("position")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            return 0.0
        x_w, y_w, z_w = float(pos[0]), float(pos[1]), float(pos[2])
        p0, p2 = x_w, -y_w
        return math.atan2(p0, p2)
    except Exception:
        return 0.0


def _apply_cousin_yaw_mirror(yaw: float, mirror_mode: str) -> float:
    """Apply optional correction after spin/cam/pi."""
    m = (mirror_mode or "none").strip().lower()
    if m in ("none", "", "0", "off", "identity"):
        return yaw
    # -yaw: reflection that often reads as front/back from table front
    if m in ("negate", "minus", "flip_front_back", "front_back", "fb"):
        return -yaw
    # π - yaw: horizontal-plane reflection (often fixes left/right inversion)
    if m in ("pi_minus", "pi_minus_yaw", "left_right", "lr", "flip_left_right"):
        return math.pi - yaw
    # +π: opposite heading (if left/right issue is actually a 180-degree heading error)
    if m in ("add_pi", "pi_plus", "flip_180", "180"):
        return yaw + math.pi
    return yaw


def _cousin_yaw_mirror_mode_from_kwargs(kwargs: Dict[str, Any]) -> str:
    """Resolve ``cousin_yaw_mirror_mode``; legacy ``cousin_yaw_final_flip=-1`` → negate."""
    raw = kwargs.get("cousin_yaw_mirror_mode") or kwargs.get("cousin_yaw_mirror")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().lower()
    try:
        ff = float(kwargs.get("cousin_yaw_final_flip", 1.0))
    except Exception:
        ff = 1.0
    if ff < 0:
        return "negate"
    return "none"


def resolve_cousin_yaw_mirror_mode(task_args: Optional[Dict[str, Any]] = None) -> str:
    """Public helper for tasks: read mirror mode from task_args dict."""
    if not task_args:
        return "none"
    return _cousin_yaw_mirror_mode_from_kwargs(task_args)


def _snapshot_index_to_yaw(
    snapshot_index: int,
    step_degrees: float = 22.5,
    layout: Optional[Dict[str, Any]] = None,
    spin_sign: float = -1.0,
    cam_azimuth_sign: float = 1.0,
    add_pi: float = 1.0,
    mirror_mode: str = "none",
) -> float:
    """
    Map ``best_snapshot_index`` to RoboTwin yaw input (radians) for ``qmult(Rx(90°), Ry(yaw))``.

    Default ``spin_sign=-1, cam_azimuth_sign=1, add_pi=1`` matches the tuned RoboTwin
    ``our_assets`` mesh vs layout frame tuning. Adjust in YAML, not in env code.

    ``mirror_mode`` (YAML ``cousin_yaw_mirror_mode``): optional correction after the sum
    above — ``negate`` is ``yaw -> -yaw``; ``pi_minus`` is ``yaw -> π - yaw`` (different
    symmetry than negate; use the one that matches what you see).
    """
    try:
        ss = float(spin_sign)
    except Exception:
        ss = -1.0
    try:
        cas = float(cam_azimuth_sign)
    except Exception:
        cas = 1.0
    try:
        ap = float(add_pi)
    except Exception:
        ap = 1.0
    rot = ss * math.radians(float(snapshot_index) * float(step_degrees))
    out = rot + cas * _layout_cam_horizontal_azimuth_rad(layout)
    out = out + ap * math.pi
    return _apply_cousin_yaw_mirror(out, mirror_mode)


def get_target_object_fixed_yaw_for_instance_id(
    label: str,
    instance_id: str,
    global_yaw_offset_deg: float = 0.0,
    spin_sign: float = -1.0,
    cam_azimuth_sign: float = 1.0,
    add_pi: float = 1.0,
    mirror_mode: str = "none",
) -> Optional[float]:
    """
    Yaw (radians) for the **same** ``instance_id`` folder you load under ``our_assets/...``.

    ``get_target_object_fixed_yaw`` used to pick a *random* instance from
    ``rotation_estimation.instances`` while ``load_actors`` picked another
    random instance for the mesh — yaw and snapshot index could refer to
    different URDFs, so tuning angles looked like it “never worked” or
    “same as before”. Always pair instance_id from ``_pick_random_instance_id``
    with this function.
    """
    layout_path = os.getenv("COUSIN_RELATIVE_LAYOUT_JSON", DEFAULT_RELATIVE_LAYOUT_PATH)
    if not layout_path or not os.path.exists(layout_path):
        return None
    try:
        layout = _load_relative_layout_json(layout_path)
    except Exception:
        return None

    objs = layout.get("objects", []) or []
    selected = set((layout.get("filter", {}) or {}).get("selected_names", []) or [])
    if selected:
        objs = [o for o in objs if o.get("name") in selected]

    target_norm = _norm_label(label)
    matched_obj = None
    for o in objs:
        if _norm_label(str(o.get("label", ""))) == target_norm:
            matched_obj = o
            break
    if matched_obj is None:
        return None

    actor_match = _require_actor_only_match(matched_obj, label)
    instances = actor_match.get("instances") or []
    if not instances:
        raise RuntimeError(
            f"label {label!r}: our_objects_actor_only_match has empty instances"
        )

    step_degrees = _layout_rotation_snapshot_step_degrees(layout)
    want = str(instance_id).strip()
    picked = None
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        iid = inst.get("instance_id")
        if iid is None:
            continue
        if str(iid).strip() == want:
            picked = inst
            break
    if picked is None:
        raise RuntimeError(
            f"label {label!r}: instance_id {want!r} not in our_objects_actor_only_match.instances"
        )
    snap_idx = picked.get("best_snapshot_index")
    if snap_idx is None:
        raise RuntimeError(
            f"label {label!r}: instance {want!r} missing best_snapshot_index"
        )
    yaw = _snapshot_index_to_yaw(
        int(snap_idx),
        step_degrees,
        layout=layout,
        spin_sign=spin_sign,
        cam_azimuth_sign=cam_azimuth_sign,
        add_pi=add_pi,
        mirror_mode=mirror_mode,
    )
    try:
        yaw += math.radians(float(global_yaw_offset_deg))
    except Exception:
        pass
    return yaw


def get_target_object_fixed_yaw(
    label: str,
    seed: int = 0,
    rng: Optional[random.Random] = None,
    global_yaw_offset_deg: float = 0.0,
    spin_sign: float = -1.0,
    cam_azimuth_sign: float = 1.0,
    add_pi: float = 1.0,
    mirror_mode: str = "none",
) -> Optional[float]:
    """
    Return a fixed yaw (radians) by picking a random entry from
    ``our_objects_actor_only_match.instances`` (prefer
    :func:`get_target_object_fixed_yaw_for_instance_id` when the instance folder is known).

    Returns None only if layout file is missing or label is not in the JSON.
    """
    layout_path = os.getenv("COUSIN_RELATIVE_LAYOUT_JSON", DEFAULT_RELATIVE_LAYOUT_PATH)
    if not layout_path or not os.path.exists(layout_path):
        return None
    try:
        layout = _load_relative_layout_json(layout_path)
    except Exception:
        return None

    objs = layout.get("objects", []) or []
    selected = set((layout.get("filter", {}) or {}).get("selected_names", []) or [])
    if selected:
        objs = [o for o in objs if o.get("name") in selected]

    target_norm = _norm_label(label)
    matched_obj = None
    for o in objs:
        if _norm_label(str(o.get("label", ""))) == target_norm:
            matched_obj = o
            break
    if matched_obj is None:
        return None

    actor_match = _require_actor_only_match(matched_obj, label)
    instances = actor_match.get("instances") or []
    if not instances:
        raise RuntimeError(
            f"label {label!r}: our_objects_actor_only_match has empty instances"
        )

    step_degrees = _layout_rotation_snapshot_step_degrees(layout)
    _rng = rng if rng is not None else random.Random(int(seed))
    picked = _rng.choice(instances)
    if not isinstance(picked, dict):
        return None
    snap_idx = picked.get("best_snapshot_index")
    if snap_idx is None:
        return None
    yaw = _snapshot_index_to_yaw(
        int(snap_idx),
        step_degrees,
        layout=layout,
        spin_sign=spin_sign,
        cam_azimuth_sign=cam_azimuth_sign,
        add_pi=add_pi,
        mirror_mode=mirror_mode,
    )
    try:
        yaw += math.radians(float(global_yaw_offset_deg))
    except Exception:
        pass
    return yaw


def get_placement_coordinates(episode_idx, seed, task_name, task_config=None, **kwargs):
    """
    Return placement coordinates for *task objects* (e.g. basket, bread) this episode.

    Called by task load_actors when task_args["use_cousin_coordinate"] is True.
    Return None to fall back to default random placement.

    Args:
        episode_idx: Current episode index (now_ep_num).
        seed: Random seed used for this episode.
        task_name: Task name (e.g. "place_bread_basket").
        task_config: Task config name from YAML (e.g. "demo_randomized").
        **kwargs: Extra args (e.g. table_xy_bias, table_z_bias).

    Returns:
        None: use default random placement.
        Otherwise: task-defined structure. Current implementation returns a dict:
            {
              "objects": [
                  {"name": ..., "label": ..., "x": ..., "y": ..., "yaw": ...},
                  ...
              ]
            }

        Selection is controlled by kwargs:
            - target_labels / primary_labels: iterable of label strings to select (e.g. ["clock"]).
    """
    # Resolve layout JSON path.
    layout_path = os.getenv("COUSIN_RELATIVE_LAYOUT_JSON", DEFAULT_RELATIVE_LAYOUT_PATH)
    if not layout_path or not os.path.exists(layout_path):
        return None

    try:
        layout = _load_relative_layout_json(layout_path)
    except Exception:
        return None

    # Normalize helper.
    norm = _norm_label

    # Determine which labels to select for task objects.
    target_labels_raw = (
        kwargs.get("target_labels")
        or kwargs.get("primary_labels")
        or kwargs.get("task_object_labels")
    )
    if not target_labels_raw:
        # If caller didn't specify, we don't override default random placement.
        return None

    if isinstance(target_labels_raw, (str, bytes)):
        target_labels = {norm(str(target_labels_raw))}
    else:
        target_labels = {norm(str(t)) for t in target_labels_raw}

    # Table XY bias can be passed through kwargs from Base_Task / task env.
    table_xy_bias = kwargs.get("table_xy_bias", (0.0, 0.0))
    if isinstance(table_xy_bias, (list, tuple)) and len(table_xy_bias) >= 2:
        table_xy_bias_xy = (float(table_xy_bias[0]), float(table_xy_bias[1]))
    else:
        table_xy_bias_xy = (0.0, 0.0)

    region_axes = (layout.get("region", {}) or {}).get("axes", {}) or {}

    # Optional filter block in JSON.
    objs = layout.get("objects", []) or []
    selected = set((layout.get("filter", {}) or {}).get("selected_names", []) or [])
    if selected:
        objs = [o for o in objs if o.get("name") in selected]

    task_objects = []
    for o in objs:
        label = norm(str(o.get("label", "")))
        if label not in target_labels:
            continue
        rel = o.get("region_position_rel", None)
        if not isinstance(rel, (list, tuple)) or len(rel) < 2:
            continue
        try:
            u, v = float(rel[0]), float(rel[1])
        except Exception:
            continue

        x, y = _rel_uv_to_robotwin_xy(
            u=u,
            v=v,
            table_xy_bias=table_xy_bias_xy,
            table_length_m=ROBOTWIN_TABLE_LENGTH_M,
            table_width_m=ROBOTWIN_TABLE_WIDTH_M,
            region_axes=region_axes,
        )

        task_objects.append(
            {
                "name": o.get("name"),
                "label": o.get("label"),
                "x": float(x),
                "y": float(y),
                "yaw": 0.0,
            }
        )

    if not task_objects:
        return None

    return {"objects": task_objects}


def get_clutter_placements(
    episode_idx,
    seed,
    task_name,
    task_config=None,
    cluttered_numbers=10,
    obj_names=None,
    cluttered_item_info=None,
    table_z_bias=0.0,
    xlim=None,
    ylim=None,
    zlim=None,
    **kwargs,
):
    """
    Return placement list for *cluttered_table* background objects this episode.

    Called by Base_Task.get_cluttered_table() when task_args["use_cousin_coordinate"] is True.
    Return None to fall back to default random clutter placement.

    Args:
        episode_idx: Current episode index (now_ep_num).
        seed: Random seed used for this episode.
        task_name: Task name (e.g. "place_bread_basket").
        task_config: Task config name from YAML.
        cluttered_numbers: Target number of clutter objects (for length of list).
        obj_names: List of available clutter object type names (from get_available_cluttered_objects).
        cluttered_item_info: Dict of object_type -> {ids, type, root, params} (for z_offset etc.).
        table_z_bias: Table height bias (m).
        xlim, ylim, zlim: Table bounds in world (for reference / scaling).
        **kwargs: Extra args from task.
            cousin_instance_score_pool_top_k: first N ids from each object's
                `instance_score_rank` to randomly choose from (default 3; unset uses default).
            cousin_emit_full_layout: if True, emit every layout object (ignore ``cluttered_numbers`` cap).

        ``exclude_labels`` does not remove objects from the list; those rows use actor-only assets
        (``spawn_as_actor_only``) so pick_up matches preview stacking while still using the
        manipulation URDF.

    Returns:
        None: use default random clutter placement.
        Otherwise: list of placement dicts, one per clutter object. Each dict may contain:
            - "x", "y" (float, required): world XY position.
            - "yaw" (float, optional): rotation around Z in radians; default 0.
            - "object_type" (str, optional): clutter type name (must be in obj_names); if omitted, chosen randomly.
            - "object_index" (str/int, optional): model id for that type; if omitted, chosen randomly.
    """
    layout_path = os.getenv("COUSIN_RELATIVE_LAYOUT_JSON", DEFAULT_RELATIVE_LAYOUT_PATH)
    if not layout_path or not os.path.exists(layout_path):
        return None

    rng = random.Random(int(seed) if seed is not None else 0)
    _pool_top = kwargs.get("cousin_instance_score_pool_top_k")
    instance_score_pool_top_k = (
        int(_pool_top)
        if _pool_top is not None
        else DEFAULT_COUSIN_INSTANCE_SCORE_POOL_TOP_K
    )

    try:
        layout = _load_relative_layout_json(layout_path)
    except Exception:
        return None

    region_axes = (layout.get("region", {}) or {}).get("axes", {}) or {}

    # Prefer selected_names if present (same semantics as the JSON's filter block).
    objs = layout.get("objects", []) or []
    selected = set((layout.get("filter", {}) or {}).get("selected_names", []) or [])
    if selected:
        objs = [o for o in objs if o.get("name") in selected]

    # Labels for task grasp / actor-only spawn (e.g. pick_up): still appear in placements so
    # ontop stacking and RNG order match preview; marked ``spawn_as_actor_only`` instead of removed.
    exclude_raw = (
        kwargs.get("exclude_labels")
        or kwargs.get("clutter_exclude_labels")
        or kwargs.get("task_object_labels")
    )
    exclude_labels: Optional[Set[str]] = None
    if exclude_raw:
        if isinstance(exclude_raw, (str, bytes)):
            exclude_labels = {_norm_label(str(exclude_raw))}
        else:
            exclude_labels = {_norm_label(str(e)) for e in exclude_raw}

    repo_root = Path(__file__).resolve().parent.parent

    # table_xy_bias: Base_Task can pass it through kwargs; otherwise default (0,0).
    table_xy_bias = kwargs.get("table_xy_bias", (0.0, 0.0))
    if isinstance(table_xy_bias, (list, tuple)) and len(table_xy_bias) >= 2:
        table_xy_bias_xy = (float(table_xy_bias[0]), float(table_xy_bias[1]))
    else:
        table_xy_bias_xy = (0.0, 0.0)

    # Build quick lookup for digital-cousins Z and ontop graph.
    obj_by_name: Dict[str, Dict[str, Any]] = {}
    for o in (layout.get("objects", []) or []):
        n = o.get("name")
        if isinstance(n, str) and n:
            obj_by_name[n] = o

    child_to_parent: Dict[str, str] = {}
    name_to_depth: Dict[str, int] = {}
    for level in (layout.get("ontop", {}) or {}).get("levels", []) or []:
        d = level.get("depth", None)
        try:
            depth_i = int(d)
        except Exception:
            continue
        for reln in level.get("relations", []) or []:
            child = reln.get("child")
            parent = reln.get("parent")
            if isinstance(child, str) and isinstance(parent, str) and child:
                child_to_parent[child] = parent
                name_to_depth[child] = depth_i

    def z_center_from_layout(name: str) -> Optional[float]:
        o = obj_by_name.get(name)
        if not o:
            return None
        c = o.get("center_world_xyz", None)
        if not isinstance(c, (list, tuple)) or len(c) < 3:
            return None
        try:
            return float(c[2])
        except Exception:
            return None

    placements: List[Dict[str, Any]] = []
    try:
        global_yaw_offset_rad = math.radians(float(kwargs.get("cousin_global_yaw_offset_deg", 0.0)))
    except Exception:
        global_yaw_offset_rad = 0.0
    try:
        spin_sign = float(kwargs.get("cousin_snapshot_spin_sign", -1.0))
    except Exception:
        spin_sign = -1.0
    try:
        cam_azimuth_sign = float(kwargs.get("cousin_cam_azimuth_sign", 1.0))
    except Exception:
        cam_azimuth_sign = 1.0
    try:
        yaw_add_pi = float(kwargs.get("cousin_yaw_add_pi", 1.0))
    except Exception:
        yaw_add_pi = 1.0
    yaw_mirror_mode = _cousin_yaw_mirror_mode_from_kwargs(kwargs)
    name_to_yaw: Dict[str, float] = {}
    name_to_instance_id: Dict[str, str] = {}
    step_degrees_layout = _layout_rotation_snapshot_step_degrees(layout)
    cousin_emit_full_layout = bool(kwargs.get("cousin_emit_full_layout", False))
    uniform_instance_sampling = bool(kwargs.get("cousin_uniform_instance_sampling", False))
    cousin_instance_indices = kwargs.get("cousin_instance_indices")

    for o in objs:
        obj_instance_name = o.get("name", None)
        label = o.get("label", None)
        rel = o.get("region_position_rel", None)
        if not isinstance(obj_instance_name, str) or not obj_instance_name:
            continue
        if not label or not isinstance(rel, (list, tuple)) or len(rel) < 2:
            continue

        label_norm = _norm_label(str(label))
        spawn_as_actor_only = bool(exclude_labels and label_norm in exclude_labels)
        if spawn_as_actor_only:
            # Some layouts (e.g. anchor layouts) only mark the grasp target with
            # ``our_objects_actor_only_match``. If a task excludes a reference label
            # (B) from clutter, fall back to the combined match for spawning while still
            # treating it as "actor-only" from the task's perspective.
            try:
                comb = _require_actor_only_match(o, str(label))
                model_dir = _model_dir_from_match(
                    repo_root,
                    comb,
                    context=f"our_objects_actor_only_match (name={obj_instance_name!r})",
                )
            except Exception:
                comb = _require_combined_match(o, obj_instance_name)
                model_dir = _model_dir_from_match(
                    repo_root,
                    comb,
                    context=f"our_objects_combined_match (name={obj_instance_name!r})",
                )
        else:
            comb = _require_combined_match(o, obj_instance_name)
            model_dir = _model_dir_from_match(
                repo_root,
                comb,
                context=f"our_objects_combined_match (name={obj_instance_name!r})",
            )
        try:
            actor_repo_relpath = model_dir.relative_to(repo_root).as_posix()
        except ValueError:
            actor_repo_relpath = ""

        randomize_clutter_yaw = kwargs.get("randomize_clutter_yaw", True)

        instance_id = _pick_random_instance_id(
            model_dir,
            rng,
            layout_obj=comb,
            instance_score_pool_top_k=instance_score_pool_top_k,
            uniform_sampling=uniform_instance_sampling,
            allowed_instance_ids=_resolve_allowed_instance_ids_for_label(
                cousin_instance_indices, str(label)
            ),
        )
        fixed_instance_yaw = None
        if not randomize_clutter_yaw:
            instances = comb.get("instances") or []
            if isinstance(instances, list) and len(instances) > 0:
                picked = None
                want = str(instance_id).strip()
                for inst in instances:
                    if not isinstance(inst, dict):
                        continue
                    iid = inst.get("instance_id", None)
                    if iid is None:
                        continue
                    if str(iid).strip() == want:
                        picked = inst
                        break
                if picked is None:
                    cand = rng.choice(instances)
                    picked = cand if isinstance(cand, dict) else None
                if isinstance(picked, dict):
                    snap_idx = picked.get("best_snapshot_index", None)
                    if snap_idx is not None:
                        fixed_instance_yaw = _snapshot_index_to_yaw(
                            int(snap_idx),
                            step_degrees_layout,
                            layout=layout,
                            spin_sign=spin_sign,
                            cam_azimuth_sign=cam_azimuth_sign,
                            add_pi=yaw_add_pi,
                            mirror_mode=yaw_mirror_mode,
                        )

        # 3) map region_position_rel (unit region projection) to RoboTwin world XY
        u, v = float(rel[0]), float(rel[1])
        x, y = _rel_uv_to_robotwin_xy(
            u=u,
            v=v,
            table_xy_bias=table_xy_bias_xy,
            table_length_m=ROBOTWIN_TABLE_LENGTH_M,
            table_width_m=ROBOTWIN_TABLE_WIDTH_M,
            region_axes=region_axes,
        )

        depth = int(name_to_depth.get(obj_instance_name, 1))
        # Layout ontop graph: instance name of the support object below this one (JSON key "parent").
        support_instance = child_to_parent.get(obj_instance_name, None)

        # Determine yaw (Z-axis rotation in RoboTwin world frame).
        # Priority: support instance yaw > fixed layout yaw > random > 0.
        if isinstance(support_instance, str) and support_instance and support_instance in name_to_yaw:
            yaw = float(name_to_yaw[support_instance])
        elif randomize_clutter_yaw:
            yaw = float(rng.uniform(-math.pi, math.pi))
        elif fixed_instance_yaw is not None:
            yaw = fixed_instance_yaw
        else:
            yaw = 0.0
        yaw = float(yaw) + float(global_yaw_offset_rad)

        dz_to_parent = None
        if depth > 1 and isinstance(support_instance, str) and support_instance:
            support_obj = obj_by_name.get(support_instance)
            parent_model_dir = None
            parent_comb = None
            if isinstance(support_obj, dict):
                spl = _norm_label(str(support_obj.get("label", "")))
                support_is_actor_only = bool(exclude_labels and spl in exclude_labels)
                if support_is_actor_only:
                    try:
                        parent_comb = _require_actor_only_match(
                            support_obj, str(support_obj.get("label"))
                        )
                    except Exception:
                        parent_comb = _require_combined_match(support_obj, support_instance)
                else:
                    parent_comb = _require_combined_match(support_obj, support_instance)
                parent_model_dir = _model_dir_from_match(
                    repo_root,
                    parent_comb,
                    context=(
                        "our_objects_actor_only_match (support name="
                        if support_is_actor_only
                        else "our_objects_combined_match (support name="
                    )
                    + f"{support_instance!r})",
                )
            parent_instance_id = name_to_instance_id.get(support_instance)
            if parent_instance_id is None and parent_model_dir is not None and parent_comb is not None:
                parent_instance_id = str(
                    _pick_random_instance_id(
                        parent_model_dir,
                        rng,
                        layout_obj=parent_comb,
                        instance_score_pool_top_k=instance_score_pool_top_k,
                        uniform_sampling=uniform_instance_sampling,
                        allowed_instance_ids=_resolve_allowed_instance_ids_for_label(
                            cousin_instance_indices,
                            str(support_obj.get("label", "")) if isinstance(support_obj, dict) else "",
                        ),
                    )
                )
            elif parent_instance_id is None:
                parent_instance_id = "0"
            child_data = _load_model_data_extent_scale_y(model_dir, instance_id)
            parent_data = (
                _load_model_data_extent_scale_y(parent_model_dir, parent_instance_id)
                if parent_model_dir
                else None
            )
            if child_data is not None and parent_data is not None:
                extent_child, scale_y_child = child_data
                extent_parent, scale_y_parent = parent_data
                dz_to_parent = (extent_parent * scale_y_parent + extent_child * scale_y_child) / 2.0
            else:
                zc = z_center_from_layout(obj_instance_name)
                zp = z_center_from_layout(support_instance)
                if zc is not None and zp is not None:
                    dz_to_parent = float(zc - zp)

        name_to_instance_id[str(obj_instance_name)] = str(instance_id)
        placements.append(
            {
                "name": obj_instance_name,
                "label": label,
                "x": x,
                "y": y,
                "yaw": yaw,
                # Must match cluttered registry keys (class folder name, or ``{name}__na`` if duplicated across buckets)
                "object_type": model_dir.name,
                "object_index": instance_id,
                "depth": depth,
                "parent": support_instance,
                "dz_to_parent": dz_to_parent,
                "spawn_as_actor_only": spawn_as_actor_only,
                "actor_repo_relpath": actor_repo_relpath if spawn_as_actor_only else None,
            }
        )
        name_to_yaw[obj_instance_name] = yaw

        if (
            not cousin_emit_full_layout
            and cluttered_numbers is not None
            and len(placements) >= int(cluttered_numbers)
        ):
            break

    return placements if placements else None
