import os
import time
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.layers.dp_attention import get_attention_dp_rank
from sglang.srt.server_args import get_global_server_args


_DUMP_SAMPLING_RUN_ID: Optional[str] = None


def _is_enabled() -> bool:
    v = (os.environ.get("SGLANG_DEBUG_DUMP_SAMPLING", "") or "").lower()
    return v in ("1", "true", "yes", "on")


def _get_run_id() -> str:
    global _DUMP_SAMPLING_RUN_ID
    if _DUMP_SAMPLING_RUN_ID is None:
        _DUMP_SAMPLING_RUN_ID = (
            os.environ.get("SGLANG_DEBUG_SAVE_LAYER_HIDDENS_RUN_ID")
            or os.environ.get("SGLANG_DEBUG_DUMP_SAMPLING_RUN_ID")
            or f"{os.getpid()}_{int(time.time() * 1000)}"
        )
    return _DUMP_SAMPLING_RUN_ID


def _get_base_dir() -> str:
    return (
        os.environ.get("SGLANG_DEBUG_DUMP_SAMPLING_DIR")
        or os.environ.get("SGLANG_DEBUG_SAVE_LAYER_HIDDENS_DIR")
        or "/tmp/sglang_sampling_dump"
    )


def _get_topk() -> int:
    try:
        return int(os.environ.get("SGLANG_DEBUG_DUMP_SAMPLING_TOPK", "50"))
    except Exception:
        return 50


def _mode_tag() -> str:
    args = get_global_server_args()
    # Prefer a stable, human-readable tag for A/B runs.
    if getattr(args, "enable_special_dp_attention", False):
        return "special_dp_attention"
    if getattr(args, "enable_dp_attention", False):
        return "dp_attention"
    return "tp"


def _extract_sampling_info_dict(sampling_info: Any) -> Dict[str, Any]:
    """Extract a small, torch.save-friendly view of sampling_info.

    Only includes fields that matter for sampling determinism (temperature/top-p/top-k/min-p/seed).
    """
    out: Dict[str, Any] = {}
    if sampling_info is None:
        return out

    # Tensors (per-seq)
    for k in ("temperatures", "top_ps", "top_ks", "min_ps", "sampling_seed"):
        v = getattr(sampling_info, k, None)
        if torch.is_tensor(v):
            out[k] = v.detach().contiguous().cpu()
        else:
            out[k] = v

    # Booleans / derived flags
    for k in (
        "need_top_p_sampling",
        "need_top_k_sampling",
        "need_min_p_sampling",
        "is_all_greedy",
        "has_custom_logit_processor",
    ):
        if hasattr(sampling_info, k):
            out[k] = getattr(sampling_info, k)

    # Grammar presence affects TP sync behavior.
    out["has_grammar"] = bool(getattr(sampling_info, "grammars", None))
    return out


def dump_next_token_logits_topk(
    *,
    req_ids: List[str],
    phase: str,
    positions: torch.Tensor,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    stage: str,
) -> None:
    """Dump per-seq next_token_logits top-k + sampling_info to disk.

    Path template (as requested):
      "{run_id}/{mode_tag}/{req_id}/{phase}/pos_{pos}/tp{tp}_{stage}.pt"
    """
    if not _is_enabled():
        return
    if next_token_logits is None:
        return
    if positions is None or not torch.is_tensor(positions):
        return
    if len(req_ids) == 0:
        return

    tp_rank = int(get_tensor_model_parallel_rank())
    dp_rank = int(get_attention_dp_rank())
    tp_size = int(get_tensor_model_parallel_world_size())
    topk = _get_topk()
    run_id = _get_run_id()
    base_dir = _get_base_dir()
    tag = _mode_tag()

    # next_token_logits: [b, vocab]
    b = int(next_token_logits.shape[0])
    if b != len(req_ids):
        # Best effort: truncate to the shared prefix so we never crash production runs.
        b = min(b, len(req_ids))
        req_ids = req_ids[:b]
        next_token_logits = next_token_logits[:b]
        positions = positions[:b]

    # Compute top-k on device then move small tensors to CPU for saving.
    k = min(int(topk), int(next_token_logits.shape[-1]))
    top_vals, top_idx = torch.topk(next_token_logits, k=k, dim=-1)
    top_vals_cpu = top_vals.detach().contiguous().cpu()
    top_idx_cpu = top_idx.detach().contiguous().cpu()
    pos_cpu = positions.detach().contiguous().cpu().to(torch.int64)

    sampling_dict = _extract_sampling_info_dict(sampling_info)

    for i in range(b):
        rid = str(req_ids[i])
        pos_i = int(pos_cpu[i].item())
        out_dir = os.path.join(
            base_dir,
            f"run_{run_id}",
            tag,
            rid,
            phase,
            f"pos_{pos_i}",
        )
        os.makedirs(out_dir, exist_ok=True)
        fname = f"tp{tp_rank}_{stage}.pt"
        out_path = os.path.join(out_dir, fname)

        payload = {
            "req_id": rid,
            "phase": phase,
            "pos": pos_i,
            "tp_rank": tp_rank,
            "dp_rank": dp_rank,
            "tp_size": tp_size,
            "mode_tag": tag,
            "stage": stage,
            "topk": k,
            "topk_vals": top_vals_cpu[i],
            "topk_idx": top_idx_cpu[i],
            "sampling_info": sampling_dict,
        }
        torch.save(payload, out_path)

