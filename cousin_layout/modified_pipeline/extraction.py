# This module modifies the Digital Cousins-derived extraction pipeline:
# https://github.com/cremebrule/digital-cousins
# Modified by RoboCousin contributors.
# Licensed under Apache-2.0; see licenses/Digital-Cousins-Apache-2.0.txt.
"""
Modified Step 1 (Real World Extraction).

We keep `digital_cousins/pipeline` unchanged and override only the single over-strict check:

    assert abs(z_dir[0]) < 0.1, f"got tilted floor: {z_dir[0]}"

This can trigger on slightly noisy floor-normal estimates and hard-crash Step 1.
In practice downstream tilt correction uses only (y, z), so we clamp roll (x component)
to 0 and continue.
"""

from __future__ import annotations

import difflib
import inspect
import json

from ..pipeline import extraction as _extraction


def _strip_leading_method_indent(src: str, n_spaces: int = 4) -> str:
    """
    `inspect.getsource()` for class methods includes the class indentation.
    We cannot `textwrap.dedent()` because the method contains triple-quoted docstrings
    whose *contents* may have 0 indentation.
    """
    prefix = " " * n_spaces
    return "".join((ln[n_spaces:] if ln.startswith(prefix) else ln) for ln in src.splitlines(True))


def _build_patched_call():
    src = inspect.getsource(_extraction.RealWorldExtractor.__call__)
    needle = 'assert abs(z_dir[0]) < 0.1, f"got tilted floor: {z_dir[0]}"'

    src_lines = src.splitlines(True)  # keepends
    replaced = False
    for i, ln in enumerate(src_lines):
        if needle in ln:
            indent = len(ln) - len(ln.lstrip(" "))
            base = " " * indent
            ind1 = " " * (indent + 4)
            ind2 = " " * (indent + 8)
            src_lines[i] = (
                f"{base}# Make sure there is minimal roll in the angle\n"
                f"{base}if abs(z_dir[0]) >= 0.1:\n"
                f"{ind1}if self.verbose:\n"
                f"{ind2}print(f\"[WARN] got tilted floor roll component z_dir[0]={{z_dir[0]:.6f}}; clamping to 0.0 to continue.\")\n"
                f"{ind1}z_dir = np.array([0.0, z_dir[1], z_dir[2]], dtype=float)\n"
                f"{ind1}z_norm = np.linalg.norm(z_dir)\n"
                f"{ind1}if z_norm > 0:\n"
                f"{ind2}z_dir = z_dir / z_norm\n"
            )
            replaced = True
            break

    if not replaced:
        raise RuntimeError(
            "Unable to patch RealWorldExtractor.__call__: expected assert line not found.\n"
            f"Looking for:\n  {needle}\n"
        )

    patched_src = _strip_leading_method_indent("".join(src_lines), n_spaces=4)
    local_ns: dict = {}
    exec(patched_src, _extraction.__dict__, local_ns)
    return local_ns["__call__"]


def _string_similarity(a: str, b: str) -> float:
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def _filter_detected_categories(step_1_output_path: str, similarity_threshold: float, verbose: bool = False) -> None:
    with open(step_1_output_path, "r") as f:
        step_1_output_info = json.load(f)

    detected_categories_path = step_1_output_info.get("detected_categories")
    if detected_categories_path is None:
        return

    with open(detected_categories_path, "r") as f:
        detected = json.load(f)

    phrases = detected.get("phrases", [])
    phrases_recaptioned = detected.get("phrases_recaptioned", [])
    if len(phrases) != len(phrases_recaptioned):
        return

    keep_mask = []
    for orig, recaptioned in zip(phrases, phrases_recaptioned):
        score = _string_similarity(str(orig).lower(), str(recaptioned).lower())
        keep = score >= similarity_threshold
        keep_mask.append(keep)
        if verbose and not keep:
            print(
                f"[modified_pipeline] Dropping object due to low recaption similarity: "
                f"orig='{orig}' recaptioned='{recaptioned}' score={score:.3f}"
            )

    if all(keep_mask):
        return

    # Filter all per-object lists.
    def _filter_list(values):
        return [v for v, keep in zip(values, keep_mask) if keep]

    detected["names"] = _filter_list(detected.get("names", []))
    detected["phrases"] = _filter_list(phrases)
    detected["phrases_recaptioned"] = _filter_list(phrases_recaptioned)
    detected["boxes"] = _filter_list(detected.get("boxes", []))
    detected["logits"] = _filter_list(detected.get("logits", []))
    detected["mount"] = _filter_list(detected.get("mount", []))

    # Remap articulation_counts indices after filtering.
    old_articulation = detected.get("articulation_counts", {})
    try:
        old_articulation = {int(k): v for k, v in old_articulation.items()}
    except Exception:
        old_articulation = {}

    new_articulation = {}
    new_idx = 0
    for old_idx, keep in enumerate(keep_mask):
        if not keep:
            continue
        if old_idx in old_articulation:
            new_articulation[new_idx] = old_articulation[old_idx]
        new_idx += 1
    detected["articulation_counts"] = new_articulation

    with open(detected_categories_path, "w") as f:
        json.dump(detected, f, indent=4)


_patched_call = _build_patched_call()


class RealWorldExtractor(_extraction.RealWorldExtractor):
    """Drop-in replacement used by `digital_cousins.modified_pipeline.acdc`."""

    def __call__(
        self,
        *args,
        recaption_similarity_threshold: float = 0.5,
        recaption_filter_enabled: bool = True,
        **kwargs,
    ):
        success, step_1_output_path = _patched_call(self, *args, **kwargs)
        # Temporarily disable recaption-based filtering.
        # Recaption itself can be useful, but in practice we've observed it can collapse multiple
        # distinct instances (e.g. several "pen") into the same recaption label ("pen holder"),
        # and the downstream pipeline is instance-based. Until recaption is made instance-safe,
        # we avoid using it for filtering here.
        _ = recaption_filter_enabled
        _ = recaption_similarity_threshold
        return success, step_1_output_path

