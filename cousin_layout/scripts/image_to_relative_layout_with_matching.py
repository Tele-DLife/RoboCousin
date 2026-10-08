from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

try:
    from script.path_config import COUSIN_LAYOUT_DESK_LAYOUT_DIR, HF_CACHE_ROOT, REPO_ROOT
except ModuleNotFoundError:
    _FALLBACK_REPO_ROOT = Path(__file__).resolve().parents[2]
    COUSIN_LAYOUT_DESK_LAYOUT_DIR = (_FALLBACK_REPO_ROOT / "cousin_layout" / "desk_layout").resolve()
    _hf_cache_env = os.getenv("ROBOTWIN_HF_CACHE_DIR", "/data/huggingface_cache")
    HF_CACHE_ROOT = Path(_hf_cache_env).expanduser().resolve()
    REPO_ROOT = _FALLBACK_REPO_ROOT

_hf_cache_root_env = os.getenv("ROBOTWIN_HF_CACHE_DIR", "").strip()
if _hf_cache_root_env:
    HF_CACHE_ROOT = Path(_hf_cache_root_env).expanduser().resolve()

# Force HuggingFace / Transformers to use local caches only.
# This prevents any accidental network access for tokenizers / models (e.g. bert-base-uncased).
os.environ["HF_HOME"] = str(HF_CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_ROOT / "hub")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LOCAL_MODELS_ONLY", os.getenv("EMBODIEDGEN_LOCAL_MODELS_ONLY", "1"))
os.environ.setdefault("EMBODIEDGEN_LOCAL_MODELS_ONLY", os.environ["LOCAL_MODELS_ONLY"])


def _use_local_models_only() -> bool:
    return os.getenv("LOCAL_MODELS_ONLY", "1") != "0"


def _hf_local_files_kwargs() -> Dict[str, Any]:
    return {"local_files_only": True} if _use_local_models_only() else {}

if __package__ in (None, ""):
    # Allow running as a script:
    #   python cousin_layout/scripts/image_to_relative_layout_with_matching.py
    # by adding the repo root (parent of `cousin_layout/`) to sys.path.
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from cousin_layout.modified_pipeline.relative_layout import (  # type: ignore
        extract_relative_layout_from_image,
        extract_relative_layout_from_step_outputs,
    )
    from cousin_layout.scripts.estimate_object_rotation import (  # type: ignore
        estimate_rotation_for_object,
        preload_rotation_models,
    )
    from cousin_layout.scripts.snapshot_generation import (  # type: ignore
        render_snapshots_for_asset_instance,
    )
else:
    from ..modified_pipeline.relative_layout import (
        extract_relative_layout_from_image,
        extract_relative_layout_from_step_outputs,
    )
    from .estimate_object_rotation import (
        estimate_rotation_for_object,
        preload_rotation_models,
    )
    from .snapshot_generation import render_snapshots_for_asset_instance

DEFAULT_QWEN_EMBEDDING_LOCAL_DIR = str(HF_CACHE_ROOT / "hub" / "models--Qwen--Qwen3-Embedding-4B")
DEFAULT_QWEN_EMBEDDING_HF_ID = "Qwen/Qwen3-Embedding-4B"

DEFAULT_BERT_HF_ID = "bert-base-uncased"

SEMANTIC_MODEL_BERT = "bert"
SEMANTIC_MODEL_QWEN = "qwen"

# Local singleton cache for this script/process.
_MATCH_EMBEDDER = None
_MATCH_EMBEDDER_MODEL_ID: Optional[str] = None
_MATCH_EMBEDDER_FAILED: bool = False
_BERT_RUNTIME: Optional[Tuple[Any, Any, Any]] = None  # (tokenizer, model, device)
_BERT_FAILED: bool = False
_MATCH_DOC_INDEX_BY_ASSETS_DIR: Dict[Tuple[str, str], Tuple[List[Path], List[str], Any]] = {}
_LABEL_RESOLVE_CACHE: Dict[str, Tuple[Optional[Path], float]] = {}


def _get_cousin_object_match_threshold() -> float:
    thr_s = os.getenv("COUSIN_OBJECT_MATCH_MIN_SIM", "").strip()
    try:
        return float(thr_s) if thr_s else 0.5
    except Exception:
        return 0.5


def _list_immediate_subdirs_sorted(parent: Path) -> List[Path]:
    if not parent.is_dir():
        return []
    return sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.name)


def _collect_class_dirs_from_asset_roots(roots_in_priority_order: List[Path]) -> List[Path]:
    """
    Collect immediate child class directories from multiple roots.
    Earlier roots win on exact name ties; embedding argmax uses this list order for ties.
    """
    out: List[Path] = []
    for root in roots_in_priority_order:
        out.extend(_list_immediate_subdirs_sorted(root))
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def _norm_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ")
    s = " ".join(s.split())
    return s


def _object_match_label(obj: Dict[str, Any]) -> str:
    """
    Choose a stable label for name matching.

    Prefer recaption label when it is available.
    Then fallback to `label` (usually class-like caption). Finally fallback to `name` but strip common
    numeric suffixes (e.g. "mug_2" -> "mug") to avoid re-running name matching
    for duplicated instances of the same object type in a scene.
    """
    recaption_label = _norm_label(str(obj.get("step1_recaption_label", "")))
    orig_label = _norm_label(str(obj.get("label", "")))
    raw = recaption_label or orig_label or (obj.get("name") or "")

    s = str(raw) if raw is not None else ""
    s_norm = _norm_label(s)
    if not s_norm:
        return ""

    # Strip trailing instance indices: "mug 2" / "mug_2" / "mug-2"
    parts = s_norm.split(" ")
    if len(parts) >= 2 and parts[-1].isdigit():
        return " ".join(parts[:-1]).strip()
    return s_norm


def _resolve_qwen_embedding_model_path() -> str:
    """
    Resolve Qwen3-Embedding-0.6B path for local sentence-transformers runtime.
    """
    local_dir = os.getenv("COUSIN_QWEN_EMBEDDING_LOCAL_DIR", DEFAULT_QWEN_EMBEDDING_LOCAL_DIR).strip()
    if not local_dir or not os.path.exists(local_dir):
        return DEFAULT_QWEN_EMBEDDING_HF_ID

    snapshots = Path(local_dir) / "snapshots"
    if snapshots.is_dir():
        snap_dirs = [p for p in snapshots.iterdir() if p.is_dir()]
        if snap_dirs:
            snap_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(snap_dirs[0])
    return local_dir


def _normalize_semantic_model(name: Optional[str]) -> str:
    s = (name or SEMANTIC_MODEL_BERT).strip().lower()
    if s in (SEMANTIC_MODEL_BERT, "bert-base", "bert_base"):
        return SEMANTIC_MODEL_BERT
    if s in (SEMANTIC_MODEL_QWEN, "qwen3", "qwen3-embedding", "qwen3_embedding"):
        return SEMANTIC_MODEL_QWEN
    raise ValueError(f"semantic_model must be '{SEMANTIC_MODEL_BERT}' or '{SEMANTIC_MODEL_QWEN}', got {name!r}")


def _get_bert_embedding_runtime(verbose: bool = False) -> Optional[Tuple[Any, Any, Any]]:
    global _BERT_RUNTIME, _BERT_FAILED
    if _BERT_RUNTIME is not None:
        return _BERT_RUNTIME
    if _BERT_FAILED:
        return None
    try:
        import torch
        from transformers import BertModel, BertTokenizer
    except Exception as e:
        if verbose:
            print(f"[AssetMatch][BERT] import failed: {e}")
        _BERT_FAILED = True
        return None

    model_id = os.getenv("COUSIN_BERT_EMBEDDING_MODEL", DEFAULT_BERT_HF_ID).strip() or DEFAULT_BERT_HF_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[AssetMatch][BERT] loading {model_id} on {device}")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_id, **_hf_local_files_kwargs())
        model = BertModel.from_pretrained(
            model_id,
            output_hidden_states=True,
            **_hf_local_files_kwargs(),
        )
        model.eval()
        model.to(device)
    except Exception as e:
        if verbose:
            print(f"[AssetMatch][BERT] model load failed: {e}")
        _BERT_FAILED = True
        return None

    _BERT_RUNTIME = (tokenizer, model, device)
    if verbose:
        print("[AssetMatch][BERT] model loaded")
    return _BERT_RUNTIME


def _bert_encode_texts(texts: List[str], verbose: bool = False) -> Any:
    """
    Mean-pool hidden_states[-2] per sequence (masked mean when batching).
    Returns CPU float tensor [len(texts), 768].
    """
    import torch
    import torch.nn.functional as F

    rt = _get_bert_embedding_runtime(verbose=verbose)
    if rt is None:
        return None
    tokenizer, model, device = rt
    if not texts:
        return torch.empty(0, 768, dtype=torch.float32)

    batch_size = 32
    chunks: List[Any] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            layer_hidden = outputs.hidden_states[-2]
            attn = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (layer_hidden * attn).sum(dim=1)
            counts = attn.sum(dim=1).clamp(min=1e-9)
            mean_embedding = summed / counts
            mean_embedding = F.normalize(mean_embedding, p=2, dim=1)
            chunks.append(mean_embedding.detach().cpu())
    return torch.cat(chunks, dim=0)


def _get_local_qwen_embedder(verbose: bool = False) -> Any:
    """
    Lazily create one local Qwen embedder for asset matching only.
    This model is intentionally separated from Step-1 GPT API model usage.
    """
    global _MATCH_EMBEDDER, _MATCH_EMBEDDER_MODEL_ID, _MATCH_EMBEDDER_FAILED
    if _MATCH_EMBEDDER is not None:
        return _MATCH_EMBEDDER
    if _MATCH_EMBEDDER_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import inspect
    except Exception as e:
        if verbose:
            print(f"[AssetMatch][QWEN] sentence_transformers import failed: {e}")
        _MATCH_EMBEDDER_FAILED = True
        return None

    model_id = _resolve_qwen_embedding_model_path()
    _MATCH_EMBEDDER_MODEL_ID = model_id
    if verbose:
        print(f"[AssetMatch][QWEN] loading model from: {model_id}")
    try:
        force_offline_s = os.getenv("COUSIN_QWEN_OFFLINE", "").strip().lower()
        force_offline = force_offline_s in ("1", "true", "yes", "y", "on")
        is_local_path = os.path.exists(model_id)
        st_kwargs: Dict[str, Any] = {}
        supports_local_only = "local_files_only" in inspect.signature(SentenceTransformer.__init__).parameters
        want_local_only = _use_local_models_only() or force_offline or is_local_path
        if supports_local_only and want_local_only:
            st_kwargs["local_files_only"] = True
        try:
            _MATCH_EMBEDDER = SentenceTransformer(model_id, trust_remote_code=True, **st_kwargs)
        except TypeError:
            _MATCH_EMBEDDER = SentenceTransformer(model_id, **st_kwargs)
    except Exception as e:
        if verbose:
            print(f"[AssetMatch][QWEN] model load failed: {e}")
        _MATCH_EMBEDDER_FAILED = True
        return None

    if verbose:
        print("[AssetMatch][QWEN] model loaded")
    return _MATCH_EMBEDDER


def _build_assets_doc_index_from_class_list(
    index_cache_id: str,
    dir_paths: List[Path],
    semantic_model: str = SEMANTIC_MODEL_BERT,
    verbose: bool = False,
    log_label: str = "AssetMatch",
) -> Tuple[List[Path], List[str], Any]:
    """
    Build and cache candidate directory embeddings keyed by (index_cache_id, semantic_model).
    """
    sm = _normalize_semantic_model(semantic_model)
    cache_key = (index_cache_id, sm)
    cached = _MATCH_DOC_INDEX_BY_ASSETS_DIR.get(cache_key)
    if cached is not None:
        return cached

    dir_names_norm: List[str] = [_norm_label(p.name) for p in dir_paths]
    doc_embeddings = None

    if sm == SEMANTIC_MODEL_BERT:
        bert_embs = _bert_encode_texts(dir_names_norm, verbose=verbose)
        if bert_embs is None:
            if verbose:
                print(f"[{log_label}][BERT] embedder unavailable, skip index build")
            result = (dir_paths, dir_names_norm, doc_embeddings)
            _MATCH_DOC_INDEX_BY_ASSETS_DIR[cache_key] = result
            return result
        if dir_names_norm:
            if verbose:
                print(f"[{log_label}][BERT] building candidate embeddings, candidates={len(dir_names_norm)}")
            doc_embeddings = bert_embs
        elif verbose:
            print(f"[{log_label}][BERT] no candidate class dirs (empty list)")
    else:
        embedder = _get_local_qwen_embedder(verbose=verbose)
        if embedder is None:
            if verbose:
                print(f"[{log_label}][QWEN] embedder unavailable, skip index build")
            result = (dir_paths, dir_names_norm, doc_embeddings)
            _MATCH_DOC_INDEX_BY_ASSETS_DIR[cache_key] = result
            return result

        if dir_names_norm:
            if verbose:
                print(f"[{log_label}][QWEN] building candidate embeddings, candidates={len(dir_names_norm)}")
            doc_embeddings = embedder.encode(dir_names_norm)
        elif verbose:
            print(f"[{log_label}][QWEN] no candidate class dirs (empty list)")

    result = (dir_paths, dir_names_norm, doc_embeddings)
    _MATCH_DOC_INDEX_BY_ASSETS_DIR[cache_key] = result
    return result


def _build_assets_doc_index(
    assets_our_objects_dir: Path,
    semantic_model: str = SEMANTIC_MODEL_BERT,
    verbose: bool = False,
) -> Tuple[List[Path], List[str], Any]:
    """
    Build and cache candidate directory embeddings keyed by (assets directory path, semantic_model).
    """
    if not assets_our_objects_dir.exists():
        if verbose:
            print(f"[AssetMatch] assets dir not found: {assets_our_objects_dir}")
        sm = _normalize_semantic_model(semantic_model)
        empty: Tuple[List[Path], List[str], Any] = ([], [], None)
        _MATCH_DOC_INDEX_BY_ASSETS_DIR[(str(assets_our_objects_dir.expanduser().resolve()), sm)] = empty
        return empty

    dir_paths = _list_immediate_subdirs_sorted(assets_our_objects_dir)
    assets_key = str(assets_our_objects_dir.expanduser().resolve())
    return _build_assets_doc_index_from_class_list(
        index_cache_id=assets_key,
        dir_paths=dir_paths,
        semantic_model=semantic_model,
        verbose=verbose,
        log_label="AssetMatch",
    )


def _initialize_matching_runtime(
    assets_our_objects_dir: Path,
    semantic_model: str = SEMANTIC_MODEL_BERT,
    verbose: bool = False,
) -> None:
    """
    Warm up matching runtime once at pipeline start for the chosen backend only:
    - bert: bert-base-uncased (lazy singleton) + candidate index
    - qwen: Qwen3-Embedding-4B via sentence-transformers (lazy singleton) + candidate index
    """
    sm = _normalize_semantic_model(semantic_model)
    if sm == SEMANTIC_MODEL_BERT:
        _get_bert_embedding_runtime(verbose=verbose)
    else:
        _get_local_qwen_embedder(verbose=verbose)
    _build_assets_doc_index(
        assets_our_objects_dir=assets_our_objects_dir,
        semantic_model=sm,
        verbose=verbose,
    )


def _initialize_robotwin_our_assets_matching(
    actor_root: Path,
    non_actor_root: Path,
    semantic_model: str = SEMANTIC_MODEL_BERT,
    verbose: bool = False,
) -> None:
    """Warm up embedder + doc indices for combined (actor+non-actor) and actor-only pools."""
    sm = _normalize_semantic_model(semantic_model)
    if sm == SEMANTIC_MODEL_BERT:
        _get_bert_embedding_runtime(verbose=verbose)
    else:
        _get_local_qwen_embedder(verbose=verbose)
    combined_id = f"robotwin_our_assets_combined:{actor_root.resolve()}|{non_actor_root.resolve()}"
    actor_id = f"robotwin_our_assets_actor:{actor_root.resolve()}"
    combined_dirs = _collect_class_dirs_from_asset_roots([actor_root, non_actor_root])
    actor_only_dirs = _collect_class_dirs_from_asset_roots([actor_root])
    _build_assets_doc_index_from_class_list(
        index_cache_id=combined_id,
        dir_paths=combined_dirs,
        semantic_model=sm,
        verbose=verbose,
        log_label="AssetMatch[combined]",
    )
    _build_assets_doc_index_from_class_list(
        index_cache_id=actor_id,
        dir_paths=actor_only_dirs,
        semantic_model=sm,
        verbose=verbose,
        log_label="AssetMatch[actor]",
    )


def _resolve_label_to_class_dir(
    index_cache_id: str,
    class_dirs: List[Path],
    label: str,
    strict: bool = False,
    verbose: bool = False,
    semantic_model: str = SEMANTIC_MODEL_BERT,
    log_prefix: str = "AssetMatch",
) -> Tuple[Optional[Path], float]:
    """
    Match label to one of class_dirs using exact normalized name then embedding similarity.
    Returns (accepted_path_or_none, best_raw_score). When below threshold, path is None but score is still valid.
    """
    sm = _normalize_semantic_model(semantic_model)
    tag = "BERT" if sm == SEMANTIC_MODEL_BERT else "QWEN"

    target_norm = _norm_label(label)
    if not target_norm:
        return None, 0.0

    cache_key = f"{index_cache_id}::{target_norm}::{sm}"
    if cache_key in _LABEL_RESOLVE_CACHE:
        cached_path, cached_score = _LABEL_RESOLVE_CACHE[cache_key]
        if verbose:
            print(
                f"[{log_prefix}][CACHE] label='{label}' norm='{target_norm}' -> "
                f"{'None' if cached_path is None else cached_path.name} score={cached_score:.4f}"
            )
        return cached_path, cached_score

    if not class_dirs:
        _LABEL_RESOLVE_CACHE[cache_key] = (None, 0.0)
        if verbose:
            print(f"[{log_prefix}] label='{label}' norm='{target_norm}' -> no candidates")
        return None, 0.0

    for d in class_dirs:
        if _norm_label(d.name) == target_norm:
            _LABEL_RESOLVE_CACHE[cache_key] = (d, 1.0)
            if verbose:
                print(f"[{log_prefix}][EXACT] label='{label}' norm='{target_norm}' -> '{d.name}'")
            return d, 1.0

    if sm == SEMANTIC_MODEL_BERT:
        if _get_bert_embedding_runtime(verbose=verbose) is None:
            _LABEL_RESOLVE_CACHE[cache_key] = (None, 0.0)
            if strict:
                raise RuntimeError(
                    f"Target label '{label}' cannot run bert matching because embedder is unavailable"
                )
            return None, 0.0
    else:
        embedder = _get_local_qwen_embedder(verbose=verbose)
        if embedder is None:
            _LABEL_RESOLVE_CACHE[cache_key] = (None, 0.0)
            if strict:
                raise RuntimeError(
                    f"Target label '{label}' cannot run qwen matching because embedder is unavailable"
                )
            return None, 0.0

    dir_paths, _dir_names_norm, doc_embeddings = _build_assets_doc_index_from_class_list(
        index_cache_id=index_cache_id,
        dir_paths=class_dirs,
        semantic_model=sm,
        verbose=verbose,
        log_label=log_prefix,
    )
    if doc_embeddings is None or not dir_paths:
        _LABEL_RESOLVE_CACHE[cache_key] = (None, 0.0)
        if strict:
            raise RuntimeError(
                f"Target label '{label}' cannot run {sm} matching because candidate embeddings are unavailable"
            )
        if verbose:
            print(
                f"[{log_prefix}][{tag}] label='{label}' norm='{target_norm}' -> None "
                "(candidate embeddings unavailable)"
            )
        return None, 0.0

    best_idx = -1
    best_score = 0.0
    try:
        if sm == SEMANTIC_MODEL_BERT:
            import torch

            q_emb = _bert_encode_texts([target_norm], verbose=verbose)
            if q_emb is None:
                raise RuntimeError("BERT query embedding failed")
            doc_t = doc_embeddings
            if not isinstance(doc_t, torch.Tensor):
                doc_t = torch.as_tensor(doc_embeddings, dtype=torch.float32)
            sim_1d = (q_emb @ doc_t.T).squeeze(0)
            best_idx = int(sim_1d.argmax().item())
            best_score = float(sim_1d[best_idx].item())
            if verbose:
                sim_row = sim_1d
                topk = min(5, len(dir_paths))
                ranked = sorted(
                    [(i, float(sim_row[i].item())) for i in range(len(dir_paths))],
                    key=lambda x: x[1],
                    reverse=True,
                )[:topk]
                ranked_text = ", ".join([f"{dir_paths[i].name}:{score:.4f}" for i, score in ranked])
                best_name = dir_paths[best_idx].name if 0 <= best_idx < len(dir_paths) else "N/A"
                print(
                    f"[{log_prefix}][BERT] label='{label}' norm='{target_norm}' "
                    f"best='{best_name}' score={best_score:.4f} top{len(ranked)}=[{ranked_text}]"
                )
        else:
            embedder = _get_local_qwen_embedder(verbose=verbose)
            assert embedder is not None
            q_emb = embedder.encode([target_norm])

            if hasattr(embedder, "similarity"):
                sim = embedder.similarity(q_emb, doc_embeddings)
            else:
                from sentence_transformers import util  # type: ignore

                sim = util.cos_sim(q_emb, doc_embeddings)
            best_idx = int(sim[0].argmax().item())
            best_score = float(sim[0][best_idx].item())
            if verbose:
                sim_row = sim[0]
                topk = min(5, len(dir_paths))
                ranked = sorted(
                    [(i, float(sim_row[i].item())) for i in range(len(dir_paths))],
                    key=lambda x: x[1],
                    reverse=True,
                )[:topk]
                ranked_text = ", ".join([f"{dir_paths[i].name}:{score:.4f}" for i, score in ranked])
                best_name = dir_paths[best_idx].name if 0 <= best_idx < len(dir_paths) else "N/A"
                print(
                    f"[{log_prefix}][QWEN] label='{label}' norm='{target_norm}' "
                    f"best='{best_name}' score={best_score:.4f} top{len(ranked)}=[{ranked_text}]"
                )
    except Exception as e:
        if verbose:
            print(f"[{log_prefix}][{tag}] similarity failed for label='{label}' norm='{target_norm}': {e}")
        _LABEL_RESOLVE_CACHE[cache_key] = (None, 0.0)
        return None, 0.0

    thr = _get_cousin_object_match_threshold()

    if verbose:
        print(
            f"[{log_prefix}][{tag}] label='{label}' norm='{target_norm}' "
            f"best_score={best_score:.4f} threshold={thr:.4f}"
        )

    if best_score < thr:
        _LABEL_RESOLVE_CACHE[cache_key] = (None, best_score)
        if verbose and 0 <= best_idx < len(dir_paths):
            print(
                f"[{log_prefix}][{tag}][REJECT] label='{label}' norm='{target_norm}' "
                f"best='{dir_paths[best_idx].name}' score={best_score:.4f} < threshold={thr:.4f}"
            )
        return None, best_score

    if 0 <= best_idx < len(dir_paths):
        matched = dir_paths[best_idx]
        _LABEL_RESOLVE_CACHE[cache_key] = (matched, best_score)
        if verbose:
            print(f"[{log_prefix}][{tag}][ACCEPT] label='{label}' norm='{target_norm}' -> '{matched.name}'")
        return matched, best_score

    _LABEL_RESOLVE_CACHE[cache_key] = (None, best_score)
    return None, best_score


def _our_assets_match_record(path: Optional[Path], best_score: float, thr: float) -> Dict[str, Any]:
    return {
        "class_name": path.name if path else None,
        "class_path": str(path) if path else None,
        "best_score": float(best_score),
        "threshold": float(thr),
        "accepted": path is not None,
    }


def _path_under_root(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (ValueError, OSError):
        return False


def _run_snapshot_match_instances_for_object(
    matched_dir_path: Path,
    obj: Dict[str, Any],
    step1_output_path: str,
    cam_pose_world: Dict[str, Any],
    instance_dirs_by_class_dir: Dict[str, List[Path]],
    rendered_snapshot_instances: set[str],
    snapshot_overwrite: bool,
    snapshot_distance_factor: float,
    snapshot_step_degrees: float,
    estimate_device: str,
    estimate_encoder: str,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[str]]:
    """
    Per-object snapshot rendering + rotation index search. Does not store transform_matrix_3x3.
    Returns (instances, instance_score_rank, error_message_if_aborted_early).
    """
    class_key = str(matched_dir_path.resolve())
    instance_dirs = instance_dirs_by_class_dir.get(class_key)
    if instance_dirs is None:
        instance_dirs = sorted(
            [p for p in matched_dir_path.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        instance_dirs_by_class_dir[class_key] = instance_dirs

    if not instance_dirs:
        return (
            [],
            [],
            f"No numeric instance dirs found under {matched_dir_path}",
        )

    instance_results: List[Dict[str, Any]] = []
    instance_scores: dict[str, float] = {}
    for inst_dir in instance_dirs:
        inst_key = str(inst_dir.resolve())
        try:
            if inst_key not in rendered_snapshot_instances:
                render_snapshots_for_asset_instance(
                    asset_instance_dir=str(inst_dir),
                    overwrite=bool(snapshot_overwrite),
                    distance_factor=float(snapshot_distance_factor),
                    camera_height=0.2,
                    step_degrees=float(snapshot_step_degrees),
                    fixed_camera_pose_world=cam_pose_world,
                )
                rendered_snapshot_instances.add(inst_key)

            est = estimate_rotation_for_object(
                step1_output_path=step1_output_path,
                object_name=str(obj.get("name", "")),
                asset_dir=str(inst_dir),
                step_degrees=float(snapshot_step_degrees),
                device=estimate_device,
                encoder=estimate_encoder,
                rotation_mode="object_spin",
                compute_transform_matrix=False,
            )
            instance_results.append(
                {
                    "instance_id": inst_dir.name,
                    "best_snapshot_index": est.get("best_snapshot_index"),
                }
            )
            score = est.get("best_match_score", None)
            if score is not None:
                try:
                    instance_scores[inst_dir.name] = float(score)
                except Exception:
                    pass
        except Exception as e:
            instance_results.append(
                {
                    "instance_id": inst_dir.name,
                    "asset_instance_path": str(inst_dir),
                    "error": str(e),
                }
            )

    instance_score_rank = [
        inst_id
        for inst_id, _score in sorted(
            instance_scores.items(),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
    ]
    return instance_results, instance_score_rank, None


def _fill_match_rotation_from_class_dir(
    match_dict: Dict[str, Any],
    class_dir: Path,
    obj: Dict[str, Any],
    step1_output_path: str,
    cam_pose_world: Dict[str, Any],
    instance_dirs_by_class_dir: Dict[str, List[Path]],
    rendered_snapshot_instances: set[str],
    snapshot_overwrite: bool,
    snapshot_distance_factor: float,
    snapshot_step_degrees: float,
    estimate_device: str,
    estimate_encoder: str,
) -> None:
    instances, rank, err = _run_snapshot_match_instances_for_object(
        matched_dir_path=class_dir,
        obj=obj,
        step1_output_path=step1_output_path,
        cam_pose_world=cam_pose_world,
        instance_dirs_by_class_dir=instance_dirs_by_class_dir,
        rendered_snapshot_instances=rendered_snapshot_instances,
        snapshot_overwrite=snapshot_overwrite,
        snapshot_distance_factor=snapshot_distance_factor,
        snapshot_step_degrees=snapshot_step_degrees,
        estimate_device=estimate_device,
        estimate_encoder=estimate_encoder,
    )
    match_dict["instances"] = instances
    match_dict["instance_score_rank"] = rank
    if err:
        match_dict["rotation_error"] = err
    else:
        match_dict.pop("rotation_error", None)


def _robotwin_dual_resolve_matches(
    label: str,
    actor_root: Path,
    non_actor_root: Path,
    semantic_model: str,
    verbose: bool,
    strict: bool,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Path]]:
    """
    Combined pool (actor+non-actor) then actor-only (only if combined accepted).
    If combined class path already lies under actor_root, actor_only duplicates combined (no second embedding pass).
    Second JSON object is null if combined rejected, or if actor-only best is below threshold.
    """
    thr = _get_cousin_object_match_threshold()
    combined_id = f"robotwin_our_assets_combined:{actor_root.resolve()}|{non_actor_root.resolve()}"
    combined_dirs = _collect_class_dirs_from_asset_roots([actor_root, non_actor_root])
    path1, score1 = _resolve_label_to_class_dir(
        index_cache_id=combined_id,
        class_dirs=combined_dirs,
        label=label,
        strict=strict,
        verbose=verbose,
        semantic_model=semantic_model,
        log_prefix="AssetMatch[combined]",
    )
    rec1 = _our_assets_match_record(path1, score1, thr)
    rec2: Optional[Dict[str, Any]] = None
    if rec1["accepted"] and path1 is not None:
        if _path_under_root(path1, actor_root):
            rec2 = {**rec1}
        else:
            actor_id = f"robotwin_our_assets_actor:{actor_root.resolve()}"
            actor_dirs = _collect_class_dirs_from_asset_roots([actor_root])
            path2, score2 = _resolve_label_to_class_dir(
                index_cache_id=actor_id,
                class_dirs=actor_dirs,
                label=label,
                strict=strict,
                verbose=verbose,
                semantic_model=semantic_model,
                log_prefix="AssetMatch[actor]",
            )
            if path2 is not None:
                rec2 = _our_assets_match_record(path2, score2, thr)
    return rec1, rec2, path1


def _find_matching_our_object_dir(
    assets_our_objects_dir: Path,
    label: str,
    strict: bool = False,
    verbose: bool = False,
    semantic_model: str = SEMANTIC_MODEL_BERT,
) -> Optional[Path]:
    """
    Match label to an assets/our_objects subdirectory name (single root; legacy override).
    """
    if not assets_our_objects_dir.exists():
        if verbose:
            print(f"[AssetMatch] assets dir not found: {assets_our_objects_dir}")
        return None
    class_dirs = _list_immediate_subdirs_sorted(assets_our_objects_dir)
    index_id = str(assets_our_objects_dir.expanduser().resolve())
    path, _score = _resolve_label_to_class_dir(
        index_cache_id=index_id,
        class_dirs=class_dirs,
        label=label,
        strict=strict,
        verbose=verbose,
        semantic_model=semantic_model,
        log_prefix="AssetMatch",
    )
    return path


def debug_match_labels(
    assets_our_objects_dir: str,
    labels: List[str],
    semantic_model: str = SEMANTIC_MODEL_BERT,
) -> None:
    """
    Lightweight debug helper to test matching without importing the full
    digital-cousins runtime (e.g. OmniGibson). Intended to be called via:
        python -c "from ... import debug_match_labels; debug_match_labels(...)".
    """
    sm = _normalize_semantic_model(semantic_model)
    assets_dir = Path(assets_our_objects_dir).expanduser().resolve()
    _initialize_matching_runtime(assets_our_objects_dir=assets_dir, semantic_model=sm, verbose=True)
    for lab in labels:
        d = _find_matching_our_object_dir(
            assets_our_objects_dir=assets_dir,
            label=str(lab),
            strict=False,
            verbose=True,
            semantic_model=sm,
        )
        print(lab, "->", None if d is None else str(d))


def _load_step1_label_maps(step1_output_path: str, verbose: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Build name -> {phrase, recaption} map from step_1_detected_categories.json.
    """
    try:
        step1_info = _load_json(Path(step1_output_path))
    except Exception as e:
        if verbose:
            print(f"[Step1Labels] failed to read step1 output: {e}")
        return {}

    detected_path = str(step1_info.get("detected_categories", "") or "").strip()
    if not detected_path:
        return {}

    try:
        detected = _load_json(Path(detected_path))
    except Exception as e:
        if verbose:
            print(f"[Step1Labels] failed to read detected categories: {e}")
        return {}

    names = detected.get("names", []) or []
    phrases = detected.get("phrases", []) or []
    recaptioned = detected.get("phrases_recaptioned", []) or []
    out: Dict[str, Dict[str, str]] = {}
    for i, nm in enumerate(names):
        if not isinstance(nm, str) or not nm.strip():
            continue
        out[nm] = {
            "phrase": str(phrases[i]) if i < len(phrases) else "",
            "recaption": str(recaptioned[i]) if i < len(recaptioned) else "",
        }
    if verbose:
        print(f"[Step1Labels] loaded mappings for {len(out)} objects")
    return out


def run_pipeline_with_matching(
    input_image_path: str,
    save_dir: str,
    region_a: float = 6.0,
    region_b: float = 4.0,
    layout_to_region_mode: str = "normalize_to_region",
    footprint_mode: str = "robust_xy_span",
    support_footprint_slice_frac: float | None = None,
    gpt_api_key: str | None = None,
    gpt_version: str = "qwen-vl-max",
    step_1_output_path: str | None = None,
    robo_twin_root: str = str(REPO_ROOT),
    assets_our_objects_dir: str | None = None,
    recaption_similarity_threshold: float = 0.5,
    recaption_filter_enabled: bool = True,
    snapshot_overwrite: bool = True,
    snapshot_distance_factor: float = 1.5,
    snapshot_step_degrees: float = 22.5,
    estimate_device: str = "cuda",
    estimate_encoder: str = "DinoV2Encoder",
    preload_models: bool = True,
    verbose: bool = False,
    semantic_model: str = SEMANTIC_MODEL_BERT,
) -> Path:
    """
    End-to-end:
      1) From RGB image, run modified relative_layout Step-1 to produce ontop layout JSON.
      2) For each object, match RoboTwin our_assets: combined actor+non-actor pool, then (if combined
         accepts) actor-only pool. Threshold matches COUSIN_OBJECT_MATCH_MIN_SIM (default 0.5).
      3) For each mapped class (combined + optional actor-only), iterate numeric instances:
         - render snapshots, estimate best_snapshot_index (step_degrees recorded once at layout root).
      4) Write mapping + per-match instances into the same JSON (no rotation_estimation block).
    """
    save_dir_path = Path(save_dir).expanduser().resolve()
    sm = _normalize_semantic_model(semantic_model)
    robo_twin_root_path = Path(robo_twin_root).expanduser().resolve()
    actor_assets_root = robo_twin_root_path / "our_assets" / "actor"
    non_actor_assets_root = robo_twin_root_path / "our_assets" / "non-actor"
    use_dual_our_assets_layout = assets_our_objects_dir is None

    if preload_models:
        if use_dual_our_assets_layout:
            _initialize_robotwin_our_assets_matching(
                actor_root=actor_assets_root,
                non_actor_root=non_actor_assets_root,
                semantic_model=sm,
                verbose=verbose,
            )
        else:
            _initialize_matching_runtime(
                assets_our_objects_dir=Path(assets_our_objects_dir).expanduser().resolve(),
                semantic_model=sm,
                verbose=verbose,
            )
        preload_rotation_models(device=estimate_device, encoder=estimate_encoder, encoder_kwargs={})

    # Step 1: produce ontop layout JSON (either by running Step-1, or by reusing an existing Step-1 output).
    if step_1_output_path:
        # Reuse Step-1 results and only compute the layout + ontop graph.
        out_dir = save_dir_path / "relative_layout"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "relative_layout.json"
        out_png = out_dir / "relative_layout_topdown.png"
        ontop_json = out_dir / "relative_layout_ontop_desktable.json"
        ontop_png = out_dir / "relative_layout_ontop_desktable_topdown.png"
        ontop_graph = out_dir / "support_graph_step1_desktable.json"
        layout_info = extract_relative_layout_from_step_outputs(
            step_1_output_path=str(Path(step_1_output_path).expanduser().resolve()),
            step_2_output_path=None,
            region_a=float(region_a),
            region_b=float(region_b),
            cousin_index=0,
            save_path=str(out_json),
            visualization_path=str(out_png),
            footprint_mode=str(footprint_mode),
            ontop_base_label="desk,table",
            ontop_base_name=None,
            ontop_include_base=False,
            support_footprint_slice_frac=support_footprint_slice_frac,
            ontop_save_path=str(ontop_json),
            ontop_visualization_path=str(ontop_png),
            ontop_graph_save_path=str(ontop_graph),
        )
    else:
        layout_info, _step1_out = extract_relative_layout_from_image(
            input_image_path=input_image_path,
            region_a=region_a,
            region_b=region_b,
            gpt_api_key=gpt_api_key,
            save_dir=str(save_dir_path),
            captions=None,
            gpt_version=gpt_version,
            max_retries=3,
            retry_wait_time=5.0,
            recaption_similarity_threshold=recaption_similarity_threshold,
            recaption_filter_enabled=recaption_filter_enabled,
            layout_to_region_mode=str(layout_to_region_mode),
            footprint_mode=str(footprint_mode),
            ontop_base_label="desk,table",
            ontop_base_name=None,
            ontop_include_base=False,
            support_footprint_slice_frac=support_footprint_slice_frac,
            verbose=verbose,
        )

    # The ontop JSON we care about is the same as used by RoboTwin by default.
    relative_layout_dir = save_dir_path / "relative_layout"
    ontop_json_path = relative_layout_dir / "relative_layout_ontop_desktable.json"

    if not ontop_json_path.exists():
        # Fallback: if the caller previously wrote a different ontop json,
        # prefer the returned layout_info and dump it here.
        _save_json(ontop_json_path, layout_info)

    layout = _load_json(ontop_json_path)

    # Step 2: initialize local matching runtime once (separate from GPT API model).
    if use_dual_our_assets_layout:
        assets_dir: Optional[Path] = None
        if not preload_models:
            _initialize_robotwin_our_assets_matching(
                actor_root=actor_assets_root,
                non_actor_root=non_actor_assets_root,
                semantic_model=sm,
                verbose=verbose,
            )
    else:
        assets_dir = Path(assets_our_objects_dir).expanduser().resolve()
        if not preload_models:
            _initialize_matching_runtime(assets_our_objects_dir=assets_dir, semantic_model=sm, verbose=verbose)

    # Step 3: for each object, find top-1 matching class (combined + optional actor-only).
    objects = layout.get("objects", []) or []
    cam_pose_world = ((layout.get("source", {}) or {}).get("cam_pose_world", {}) or {})
    step1_output_path = str((layout.get("source", {}) or {}).get("step_1_output_path", "") or "")
    step1_label_maps: Dict[str, Dict[str, str]] = {}
    if step1_output_path and Path(step1_output_path).exists():
        step1_label_maps = _load_step1_label_maps(step1_output_path=step1_output_path, verbose=verbose)

    # Cache for this run:
    # - match_label -> (combined template dict, actor-only template or None, accepted class Path or None)
    #   templates hold only class_name/path/scores; per-object copies get instances filled below.
    # - mapped class dir -> its numeric instance dirs (avoid repeated listing)
    # - instance dir -> whether snapshots already rendered in this run (avoid duplicate snapshot generation)
    match_bundle_by_label: Dict[str, Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Path]]] = {}
    instance_dirs_by_class_dir: Dict[str, List[Path]] = {}
    rendered_snapshot_instances: set[str] = set()

    for obj in objects:
        obj.pop("rotation_estimation", None)
        obj_name = str(obj.get("name", "") or "")
        if obj_name and obj_name in step1_label_maps:
            lab_info = step1_label_maps[obj_name]
            # Keep original phrase as a debug field and provide recaption label for matching logic.
            obj["step1_phrase_label"] = str(lab_info.get("phrase", "") or "")
            obj["step1_recaption_label"] = str(lab_info.get("recaption", "") or "")

        match_label = _object_match_label(obj)
        if not match_label:
            continue

        if match_label in match_bundle_by_label:
            tpl_combined, tpl_actor, matched_dir = match_bundle_by_label[match_label]
        else:
            if use_dual_our_assets_layout:
                tpl_combined, tpl_actor, matched_dir = _robotwin_dual_resolve_matches(
                    label=match_label,
                    actor_root=actor_assets_root,
                    non_actor_root=non_actor_assets_root,
                    semantic_model=sm,
                    verbose=verbose,
                    strict=False,
                )
            else:
                assert assets_dir is not None
                class_dirs = _list_immediate_subdirs_sorted(assets_dir)
                path_s, score_s = _resolve_label_to_class_dir(
                    index_cache_id=str(assets_dir.resolve()),
                    class_dirs=class_dirs,
                    label=match_label,
                    strict=False,
                    verbose=verbose,
                    semantic_model=sm,
                    log_prefix="AssetMatch",
                )
                thr_s = _get_cousin_object_match_threshold()
                tpl_combined = _our_assets_match_record(path_s, score_s, thr_s)
                tpl_actor = None
                matched_dir = path_s
            match_bundle_by_label[match_label] = (tpl_combined, tpl_actor, matched_dir)

        # Match legacy behavior: when the first combined pass misses the
        # threshold, clear only the class field and skip extended match fields.
        if matched_dir is None:
            obj["our_objects_class_name"] = None
            obj["our_objects_class_path"] = None
            obj.pop("our_objects_combined_match", None)
            obj.pop("our_objects_actor_only_match", None)
            continue

        rec_combined = copy.deepcopy(tpl_combined)
        rec_actor_only = copy.deepcopy(tpl_actor) if tpl_actor is not None else None

        matched_dir_path = Path(matched_dir)
        obj["our_objects_class_name"] = matched_dir_path.name
        obj["our_objects_class_path"] = str(matched_dir_path)

        if not step1_output_path or not Path(step1_output_path).exists():
            msg = f"Missing step_1_output_path in layout source: {step1_output_path}"
            rec_combined["instances"] = []
            rec_combined["instance_score_rank"] = []
            rec_combined["rotation_error"] = msg
            if rec_actor_only is not None:
                rec_actor_only["instances"] = []
                rec_actor_only["instance_score_rank"] = []
                rec_actor_only["rotation_error"] = msg
            obj["our_objects_combined_match"] = rec_combined
            obj["our_objects_actor_only_match"] = rec_actor_only
            continue

        path_c = Path(str(rec_combined["class_path"]))
        _fill_match_rotation_from_class_dir(
            match_dict=rec_combined,
            class_dir=path_c,
            obj=obj,
            step1_output_path=step1_output_path,
            cam_pose_world=cam_pose_world,
            instance_dirs_by_class_dir=instance_dirs_by_class_dir,
            rendered_snapshot_instances=rendered_snapshot_instances,
            snapshot_overwrite=snapshot_overwrite,
            snapshot_distance_factor=snapshot_distance_factor,
            snapshot_step_degrees=snapshot_step_degrees,
            estimate_device=estimate_device,
            estimate_encoder=estimate_encoder,
        )

        if rec_actor_only is not None:
            p_a = rec_actor_only.get("class_path")
            p_c = rec_combined.get("class_path")
            if (
                p_a
                and p_c
                and Path(str(p_a)).resolve() == Path(str(p_c)).resolve()
            ):
                rec_actor_only = copy.deepcopy(rec_combined)
            else:
                _fill_match_rotation_from_class_dir(
                    match_dict=rec_actor_only,
                    class_dir=Path(str(p_a)),
                    obj=obj,
                    step1_output_path=step1_output_path,
                    cam_pose_world=cam_pose_world,
                    instance_dirs_by_class_dir=instance_dirs_by_class_dir,
                    rendered_snapshot_instances=rendered_snapshot_instances,
                    snapshot_overwrite=snapshot_overwrite,
                    snapshot_distance_factor=snapshot_distance_factor,
                    snapshot_step_degrees=snapshot_step_degrees,
                    estimate_device=estimate_device,
                    estimate_encoder=estimate_encoder,
                )

        obj["our_objects_combined_match"] = rec_combined
        obj["our_objects_actor_only_match"] = rec_actor_only

    layout["rotation_snapshot_step_degrees"] = float(snapshot_step_degrees)
    layout["objects"] = objects
    _save_json(ontop_json_path, layout)

    return ontop_json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate relative_layout_ontop_desktable.json from an input image "
            "and semantically match each object against RoboTwin our_assets "
            "(actor + non-actor, with optional actor-only matching)."
        )
    )
    parser.add_argument(
        "--input-image-path",
        type=str,
        required=False,
        help="Path to the input RGB image.",
    )
    parser.add_argument(
        "--step-1-output-path",
        type=str,
        default=None,
        help="Optional step_1_output_info.json path to skip Step 1. The image is not processed again; only relative_layout, ontop, matching, and rotation are computed.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(COUSIN_LAYOUT_DESK_LAYOUT_DIR / str(time.time())),
        help="Directory for relative_layout and intermediate Step 1 outputs.",
    )
    parser.add_argument(
        "--region-a",
        type=float,
        default=1.2,
        help="Target region length along x (meters); must match the RoboTwin tabletop size.",
    )
    parser.add_argument(
        "--region-b",
        type=float,
        default=0.7,
        help="Target region width along y (meters); must match the RoboTwin tabletop size.",
    )
    parser.add_argument(
        "--layout-to-region-mode",
        type=str,
        default="center_preserving_uniform_scale",
        choices=["normalize_to_region", "center_preserving_uniform_scale"],
        help="How to map the Step 1 layout to the tabletop region. The default "
        "center_preserving_uniform_scale first attempts centered placement at "
        "the original scale, then uniformly shrinks if needed; "
        "normalize_to_region is the legacy behavior that normalizes to [0, 1] "
        "before mapping to the region.",
    )
    parser.add_argument(
        "--footprint-mode",
        type=str,
        default="robust_xy_span",
        choices=["robust_xy_span", "support_plane"],
        help="Top-down footprint method: robust_xy_span (legacy) or "
        "support_plane (fit a support plane and retain nearby points to avoid "
        "vertical laptop-screen contamination).",
    )
    parser.add_argument(
        "--support-footprint-slice-frac",
        type=float,
        default=None,
        dest="support_footprint_slice_frac",
        help=(
            "Optional: when constructing the Step 1 support/ontop graph, use "
            "only the bottom slice of each object point cloud to compute its "
            "top-down footprint. For example, 0.2 uses the bottom 20%% of "
            "height and can reduce false positives for tall irregular objects "
            "such as lamps. The default None disables slicing and uses the "
            "full AABB footprint (legacy behavior)."
        ),
    )
    # Backward/typo-friendly alias (underscore style)
    parser.add_argument(
        "--support_footprint_slice_frac",
        type=float,
        default=None,
        dest="support_footprint_slice_frac",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gpt-api-key",
        type=str,
        default=None,
        help="GPT API key for Step 1 captioning; uses default.yaml when omitted.",
    )
    parser.add_argument(
        "--gpt-version",
        type=str,
        default="qwen-vl-max",
        help="Model version. Default: qwen-vl-max.",
    )
    parser.add_argument(
        "--robo-twin-root",
        type=str,
        default=str(REPO_ROOT),
        help="RoboTwin repository root. By default, matches under "
        "<robo-twin-root>/our_assets/actor and non-actor.",
    )
    parser.add_argument(
        "--assets-our-objects-dir",
        type=str,
        default=None,
        help="Optional single asset root (each child directory is a class) for "
        "legacy single-pool matching. When omitted, uses two-stage matching "
        "over <robo-twin-root>/our_assets/actor and non-actor.",
    )
    parser.add_argument(
        "--no-recaption-filter",
        action="store_true",
        help="Disable the recaption filter (recaption_filter_enabled=False).",
    )
    parser.add_argument(
        "--recaption-similarity-threshold",
        type=float,
        default=0.5,
        help="Recaption-filter similarity threshold passed to "
        "relative_layout.extract_relative_layout_from_image.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional debugging information.",
    )
    parser.add_argument(
        "--snapshot-overwrite",
        action="store_true",
        help="Regenerate snapshots for mapped class instances.",
    )
    parser.add_argument(
        "--snapshot-distance-factor",
        type=float,
        default=1.5,
        help="Snapshot rendering distance factor.",
    )
    parser.add_argument(
        "--snapshot-step-degrees",
        type=float,
        default=22.5,
        help="Snapshot angular interval per frame; must match rotation estimation.",
    )
    parser.add_argument(
        "--estimate-device",
        type=str,
        default="cuda",
        help="Device for rotation estimation. Default: cuda; cpu is supported.",
    )
    parser.add_argument(
        "--estimate-encoder",
        type=str,
        default="DinoV2Encoder",
        help="Vision encoder for rotation estimation. Default: DinoV2Encoder.",
    )
    parser.add_argument(
        "--no-preload-models",
        action="store_true",
        help="Disable model preloading at pipeline startup (semantic BERT/Qwen "
        "matching and the rotation-estimation FeatureMatcher).",
    )
    parser.add_argument(
        "--semantic-model",
        "--semantic_model",
        dest="semantic_model",
        type=str,
        default=SEMANTIC_MODEL_QWEN,
        choices=[SEMANTIC_MODEL_BERT, SEMANTIC_MODEL_QWEN],
        help=(
            "Semantic matching for assets/our_objects class names: bert "
            "(default; local bert-base-uncased, without loading Qwen) or qwen "
            "(Qwen3-Embedding-4B, lazily loaded only for this option)."
        ),
    )

    args = parser.parse_args()

    if not args.input_image_path and not args.step_1_output_path:
        raise ValueError("Must provide --input-image-path or --step-1-output-path.")
    if args.step_1_output_path and args.input_image_path:
        # Prefer skipping Step-1 when an explicit step output is provided.
        pass

    ontop_json_path = run_pipeline_with_matching(
        input_image_path=args.input_image_path or "",
        save_dir=args.save_dir,
        region_a=float(args.region_a),
        region_b=float(args.region_b),
        layout_to_region_mode=str(args.layout_to_region_mode),
        footprint_mode=str(args.footprint_mode),
        support_footprint_slice_frac=args.support_footprint_slice_frac,
        gpt_api_key=args.gpt_api_key,
        gpt_version=args.gpt_version,
        step_1_output_path=args.step_1_output_path,
        robo_twin_root=args.robo_twin_root,
        assets_our_objects_dir=args.assets_our_objects_dir,
        recaption_similarity_threshold=float(args.recaption_similarity_threshold),
        recaption_filter_enabled=not bool(args.no_recaption_filter),
        snapshot_overwrite=bool(args.snapshot_overwrite),
        snapshot_distance_factor=float(args.snapshot_distance_factor),
        snapshot_step_degrees=float(args.snapshot_step_degrees),
        estimate_device=args.estimate_device,
        estimate_encoder=args.estimate_encoder,
        preload_models=not bool(args.no_preload_models),
        verbose=bool(args.verbose),
        semantic_model=str(args.semantic_model),
    )

    print(f"Matching results written to: {ontop_json_path}")


if __name__ == "__main__":
    main()

# python -m digital_cousins.scripts.image_to_relative_layout_with_matching \
#   --input-image-path /path/to/desk.png \
#   --snapshot-overwrite \
#   --verbose \
#   --support-footprint-slice-frac 0.2

# python -m digital_cousins.scripts.image_to_relative_layout_with_matching \
#   --step-1-output-path cousin_layout/desk_layout/<run_id>/step_1_output/step_1_output_info.json \
#   --save-dir cousin_layout/desk_layout/<run_id> \
#   --snapshot-overwrite  \
#   --verbose\
#   --support-footprint-slice-frac 0.2


