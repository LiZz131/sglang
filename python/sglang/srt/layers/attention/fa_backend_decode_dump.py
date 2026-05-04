# Copyright 2023-2025 SGLang Team
# Optional full CPU snapshot for FlashAttentionBackend.forward_decode (decode only).
#
# Env:
#   SGLANG_DEBUG_DUMP_FA_BACKEND=1
#   SGLANG_DEBUG_DUMP_FA_BACKEND_LAYERS=0,1,...   (empty = all layers)
#   SGLANG_DEBUG_DUMP_FA_BACKEND_DIR=...          (fallback: SGLANG_DEBUG_SAVE_LAYER_HIDDENS_DIR, then /tmp)
#   SGLANG_DEBUG_SAVE_LAYER_HIDDENS_RUN_ID        (subdir run_<id>, default unknown_run)
#   SGLANG_DEBUG_SAVE_LAYER_HIDDENS_SYNC=1        (optional cuda sync before read)
#   SGLANG_DEBUG_DUMP_FA_BACKEND_KV=used|full    (default used: only KV slots for page_table
#                                                 union + out_cache_loc rows; full = entire layer pool)
#
# Note: MLA decode passes ``k_rope`` as a view over the *entire* KV row dimension (size+page_size).
# We persist ``inputs_k_rope`` as the same used-page gather as ``mla_key_used_page_slots_concat``'s
# rope slice would be, not the full pool (which would be hundreds of MB per .pt file).

from __future__ import annotations

import logging
import os
from dataclasses import fields
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_FB_SKIP_FIELDS = frozenset(
    {
        "token_to_kv_pool",
        "req_to_token_pool",
        "attn_backend",
        "sampling_info",
        "mm_inputs",
        "spec_info",
        "tbo_children",
        "model_specific_states",
        "next_token_logits_buffer",
        "input_embeds",
        "temperature",
        "top_p",
        "mrope_positions",
        "hidden_states",
        "residual",
        "encoder_lens_cpu",
        "extend_prefix_lens_cpu",
        "extend_seq_lens_cpu",
        "extend_logprob_start_lens_cpu",
        "lora_ids",
        "dimensions",
        "nsa_cp_metadata",
    }
)

_META_TENSOR_ATTRS: List[str] = [
    "cache_seqlens_int32",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "page_table",
    "swa_page_table",
    "encoder_cu_seqlens_k",
    "encoder_lens_int32",
    "encoder_page_table",
]

_META_SCALAR_ATTRS: List[str] = [
    "max_seq_len_q",
    "max_seq_len_k",
    "encoder_max_seq_len_k",
]


def _enabled() -> bool:
    v = (os.environ.get("SGLANG_DEBUG_DUMP_FA_BACKEND", "") or "").lower()
    return v in ("1", "true", "yes", "on")


def _layer_allowed(layer_id: int) -> bool:
    raw = (os.environ.get("SGLANG_DEBUG_DUMP_FA_BACKEND_LAYERS", "") or "").strip()
    if not raw:
        return True
    try:
        return layer_id in {int(x.strip()) for x in raw.split(",") if x.strip()}
    except Exception:
        return True


def _dump_root() -> str:
    return (
        os.environ.get("SGLANG_DEBUG_DUMP_FA_BACKEND_DIR")
        or os.environ.get("SGLANG_DEBUG_SAVE_LAYER_HIDDENS_DIR")
        or "/tmp/sglang_fa_decode_dump"
    )


def _run_id() -> str:
    return (os.environ.get("SGLANG_DEBUG_SAVE_LAYER_HIDDENS_RUN_ID") or "").strip() or "unknown_run"


def _path_tag(fb: ForwardBatch) -> str:
    fm = getattr(fb, "forward_mode", None)
    if fm is not None and fm.is_split_prefill():
        return "split_prefill"
    return "normal_forward"


def _maybe_sync_cuda() -> None:
    if (os.environ.get("SGLANG_DEBUG_SAVE_LAYER_HIDDENS_SYNC", "") or "").strip() == "1":
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _fa_decode_dump_filename_suffix(
    forward_batch: ForwardBatch, positions: Optional[torch.Tensor]
) -> str:
    """Filesystem suffix for FA decode dumps.

    **Must stay aligned** with ``deepseek_v2._debug_decode_dump_filename_suffix`` so that
    ``compare_fa_backend_decode_dump`` can reuse the same decode-signature / req-id pairing
    as ``compare_decode_layer_hiddens.py``.
    """
    req_keys = list(getattr(forward_batch, "_debug_req_ids", None) or [])
    pos_src = positions if torch.is_tensor(positions) else getattr(
        forward_batch, "positions", None
    )
    if torch.is_tensor(pos_src):
        pos_flat = pos_src.detach().contiguous().reshape(-1).cpu()
    else:
        pos_flat = torch.tensor([], dtype=torch.long)
    if pos_flat.numel() == 0:
        pos_part = "nopos"
    elif pos_flat.numel() == 1:
        pos_part = f"pos{int(pos_flat.item())}"
    else:
        shown = "_".join(str(int(x)) for x in pos_flat.tolist()[:8])
        if pos_flat.numel() > 8:
            shown += f"_etc{pos_flat.numel()}"
        pos_part = f"npos{pos_flat.numel()}_{shown}"
    pos_part = pos_part.replace(os.sep, "_").replace(" ", "_")

    if not req_keys:
        rpart = "noreq"
    elif len(req_keys) == 1:
        rpart = str(req_keys[0])[:56]
    else:
        tail = "_".join(str(x)[:10] for x in req_keys[:4])
        rpart = f"nreq{len(req_keys)}_{tail}"
        if len(req_keys) > 4:
            rpart += "_etc"
    rpart = (
        rpart.replace(os.sep, "_")
        .replace(" ", "_")
        .replace("/", "_")
    )
    return f"_decode_{pos_part}_req{rpart}"


def _snapshot_decode_align_fields(
    forward_batch: ForwardBatch,
    positions: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Fields aligned with ``deepseek_v2`` layer decode dumps for offline pairing."""
    pos_t = positions if torch.is_tensor(positions) else getattr(
        forward_batch, "positions", None
    )
    positions_cpu = (
        pos_t.detach().contiguous().cpu() if torch.is_tensor(pos_t) else None
    )
    slc = getattr(forward_batch, "seq_lens_cpu", None)
    if slc is None:
        slc_cpu = None
    elif isinstance(slc, torch.Tensor):
        slc_cpu = slc.detach().contiguous().cpu()
    else:
        slc_cpu = torch.tensor(slc, dtype=torch.int32)
    seq_lens_gpu = getattr(forward_batch, "seq_lens", None)
    seq_lens_t = (
        seq_lens_gpu.detach().contiguous().cpu()
        if torch.is_tensor(seq_lens_gpu)
        else None
    )
    fm = getattr(forward_batch, "forward_mode", None)
    try:
        from sglang.srt.distributed.parallel_state import get_attention_dp_rank

        adp_r = int(get_attention_dp_rank())
    except Exception:
        adp_r = -1
    return {
        "debug_req_ids": list(getattr(forward_batch, "_debug_req_ids", None) or []),
        "positions": positions_cpu,
        "seq_lens_cpu": slc_cpu,
        "seq_lens": seq_lens_t,
        "attn_dp_rank": adp_r,
        "forward_mode": str(fm) if fm is not None else None,
        "global_num_tokens": list(getattr(forward_batch, "global_num_tokens", []) or []),
        "global_seq_lens_sum_per_dp": list(
            getattr(forward_batch, "global_seq_lens_sum_per_dp", []) or []
        ),
    }


def _to_cpu(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None or not torch.is_tensor(t):
        return None
    return t.detach().contiguous().cpu()


def _snapshot_forward_batch_tensors(fb: ForwardBatch) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for f in fields(fb):
        name = f.name
        if name in _FB_SKIP_FIELDS:
            continue
        v = getattr(fb, name, None)
        if torch.is_tensor(v):
            out[name] = _to_cpu(v)
    return out


def _snapshot_forward_batch_scalars(fb: ForwardBatch) -> Dict[str, Any]:
    fm = getattr(fb, "forward_mode", None)
    out: Dict[str, Any] = {
        "forward_mode": str(fm) if fm is not None else None,
        "batch_size": fb.batch_size,
        "seq_lens_sum": fb.seq_lens_sum,
        "return_logprob": bool(getattr(fb, "return_logprob", False)),
        "special_dp_attention": bool(getattr(fb, "special_dp_attention", False)),
        "dp_local_token_start": getattr(fb, "dp_local_token_start", None),
        "dp_local_token_end": getattr(fb, "dp_local_token_end", None),
        "dp_padding_mode": repr(getattr(fb, "dp_padding_mode", None)),
        "is_extend_in_batch": bool(getattr(fb, "is_extend_in_batch", False)),
        "global_dp_buffer_len": getattr(fb, "global_dp_buffer_len", None),
        "padded_static_len": getattr(fb, "padded_static_len", None),
        "num_token_non_padded_cpu": getattr(fb, "num_token_non_padded_cpu", None),
        "is_prefill_only": bool(getattr(fb, "is_prefill_only", False)),
        "global_seq_lens_sum_per_dp": getattr(fb, "global_seq_lens_sum_per_dp", None),
        "global_num_tokens_cpu": getattr(fb, "global_num_tokens_cpu", None),
        "original_global_num_tokens_cpu": getattr(fb, "original_global_num_tokens_cpu", None),
    }
    return out


def _snapshot_metadata_obj(metadata: Any, prefix: str) -> Dict[str, Any]:
    if metadata is None:
        return {}
    out: Dict[str, Any] = {"_prefix": prefix, "_type": type(metadata).__name__}
    for attr in _META_TENSOR_ATTRS:
        v = getattr(metadata, attr, None)
        if torch.is_tensor(v):
            out[attr] = _to_cpu(v)
    for attr in _META_SCALAR_ATTRS:
        if hasattr(metadata, attr):
            out[attr] = getattr(metadata, attr)
    if hasattr(metadata, "window_size"):
        out["window_size"] = getattr(metadata, "window_size")
    lam = getattr(metadata, "local_attn_metadata", None)
    if lam is not None:
        sub: Dict[str, Any] = {"_type": type(lam).__name__}
        for a in (
            "local_query_start_loc",
            "local_seqused_k",
            "local_block_table",
            "local_max_query_len",
            "local_max_seq_len",
        ):
            v = getattr(lam, a, None)
            if torch.is_tensor(v):
                sub[a] = _to_cpu(v)
            elif v is not None:
                sub[a] = v
        out["local_attn_metadata"] = sub
    return out


def _kv_dump_mode() -> str:
    v = (os.environ.get("SGLANG_DEBUG_DUMP_FA_BACKEND_KV", "") or "").strip().lower()
    if v in ("full", "all", "whole", "pool"):
        return "full"
    return "used"


def _page_tables_for_dump(
    fb: ForwardBatch,
    backend: Any,
    fa_kv_page_tables: Optional[List[torch.Tensor]],
) -> List[torch.Tensor]:
    pts: List[torch.Tensor] = list(fa_kv_page_tables or [])
    if not pts:
        md = getattr(backend, "forward_metadata", None)
        pt0 = getattr(md, "page_table", None) if md is not None else None
        if torch.is_tensor(pt0):
            pts.append(pt0)
    return pts


def _unique_pages_union(
    page_tables: Optional[List[torch.Tensor]], batch_size: int
) -> torch.Tensor:
    if not page_tables:
        return torch.tensor([], dtype=torch.long)
    chunks: List[torch.Tensor] = []
    bs = int(batch_size)
    for pt in page_tables:
        if not torch.is_tensor(pt):
            continue
        take = min(bs, int(pt.shape[0]))
        if take <= 0:
            continue
        flat = pt[:take].flatten().long()
        flat = flat[flat >= 0]
        if flat.numel():
            chunks.append(flat)
    if not chunks:
        return torch.tensor([], dtype=torch.long)
    return torch.unique(torch.cat(chunks, dim=0))


def _gather_slot_blocks(
    kb: torch.Tensor, unique_pages: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Concatenate kb[p*ps:(p+1)*ps] for each physical page index p (paged KV layout)."""
    if unique_pages.numel() == 0 or page_size <= 0:
        return kb[:0].clone()
    num_slots = int(kb.shape[0])
    max_page = num_slots // page_size
    parts: List[torch.Tensor] = []
    for p in torch.sort(unique_pages.long()).values.tolist():
        p = int(p)
        if p < 0 or p >= max_page:
            continue
        s = p * page_size
        e = min(s + page_size, num_slots)
        parts.append(kb[s:e].contiguous())
    if not parts:
        return kb[:0].clone()
    return torch.cat(parts, dim=0)


def _mla_out_cache_slot_rows(kb: torch.Tensor, fb: ForwardBatch) -> Optional[torch.Tensor]:
    ocl = getattr(fb, "out_cache_loc", None)
    if not torch.is_tensor(ocl):
        return None
    idx = ocl.long().flatten()
    idx = idx[(idx >= 0) & (idx < kb.shape[0])]
    if idx.numel() == 0:
        return None
    u = torch.unique(idx)
    return kb[u].contiguous()


def _build_kv_payload(
    fb: ForwardBatch,
    backend: Any,
    layer_id: int,
    use_mla: bool,
    page_tables: Optional[List[torch.Tensor]],
) -> Dict[str, Any]:
    pool = getattr(fb, "token_to_kv_pool", None)
    if pool is None:
        return {"error": "no token_to_kv_pool"}

    mode = _kv_dump_mode()
    page_size = int(getattr(backend, "page_size", 1) or 1)
    bs = int(fb.batch_size)
    pts: List[torch.Tensor] = _page_tables_for_dump(fb, backend, page_tables)

    out: Dict[str, Any] = {"kv_dump_mode": mode}

    if mode == "full":
        try:
            if use_mla:
                kb = pool.get_key_buffer(layer_id)
                out["mla_key_buffer_full"] = _to_cpu(kb)
            else:
                kbuf, vbuf = pool.get_kv_buffer(layer_id)
                out["mha_key_buffer_full"] = _to_cpu(kbuf)
                out["mha_value_buffer_full"] = _to_cpu(vbuf)
        except Exception as e:
            out["error"] = repr(e)
        return out

    uniq = _unique_pages_union(pts, bs)
    out["used_unique_page_ids"] = uniq.cpu().tolist()
    out["used_num_unique_pages"] = int(uniq.numel())

    try:
        if use_mla:
            kb = pool.get_key_buffer(layer_id)
            out["mla_key_used_page_slots_concat"] = _to_cpu(
                _gather_slot_blocks(kb, uniq, page_size)
            )
            rows = _mla_out_cache_slot_rows(kb, fb)
            if rows is not None:
                out["mla_key_out_cache_slot_rows"] = _to_cpu(rows)
        else:
            kbuf, vbuf = pool.get_kv_buffer(layer_id)
            out["mha_key_used_page_slots_concat"] = _to_cpu(
                _gather_slot_blocks(kbuf, uniq, page_size)
            )
            out["mha_value_used_page_slots_concat"] = _to_cpu(
                _gather_slot_blocks(vbuf, uniq, page_size)
            )
    except Exception as e:
        out["used_gather_error"] = repr(e)
    return out


def _req_to_token_rows(fb: ForwardBatch, backend: Any) -> Optional[torch.Tensor]:
    pool = getattr(fb, "req_to_token_pool", None)
    if pool is None or not torch.is_tensor(getattr(pool, "req_to_token", None)):
        return None
    rpi = fb.req_pool_indices
    if not torch.is_tensor(rpi):
        return None
    md = getattr(backend, "forward_metadata", None)
    max_len = int(pool.req_to_token.shape[1])
    nk = max_len
    if md is not None:
        msk = getattr(md, "max_seq_len_k", None)
        if msk is not None and int(msk) > 0:
            nk = min(int(msk), max_len)
    rt = pool.req_to_token[rpi, :nk]
    return _to_cpu(rt)


def _to_cpu_inputs_k_rope(
    k_rope: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    backend: Any,
    fa_kv_page_tables: Optional[List[torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Save k_rope for decode dumps without materializing the full KV pool row dimension.

    FlashAttention MLA path uses ``k_rope = kv_cache[:, :, v_head_dim:]`` where ``kv_cache`` spans
    all allocator slots; naive ``_to_cpu`` would copy ~size*rope_dim per file.
    """
    if not torch.is_tensor(k_rope):
        return _to_cpu(k_rope)
    pool = getattr(forward_batch, "token_to_kv_pool", None)
    page_size = int(getattr(backend, "page_size", 1) or 1)
    sz = int(getattr(pool, "size", 0) or 0) if pool is not None else 0
    ps_pool = (
        int(getattr(pool, "page_size", page_size) or page_size)
        if pool is not None
        else page_size
    )
    full_rows = sz + ps_pool if sz > 0 else 0
    if pool is None or full_rows <= 0 or int(k_rope.shape[0]) < full_rows - 2:
        return _to_cpu(k_rope)
    pts = _page_tables_for_dump(forward_batch, backend, fa_kv_page_tables)
    uniq = _unique_pages_union(pts, int(forward_batch.batch_size))
    if uniq.numel() == 0:
        return _to_cpu(k_rope)
    gathered = _gather_slot_blocks(k_rope, uniq, page_size)
    return _to_cpu(gathered)


def maybe_dump_forward_decode(
    backend: Any,
    layer: Any,
    forward_batch: ForwardBatch,
    *,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_rope: Optional[torch.Tensor],
    k_rope: Optional[torch.Tensor],
    sinks: Optional[torch.Tensor],
    save_kv_cache: bool,
    o: torch.Tensor,
    dbg_flags: Dict[str, Any],
    fa_kv_page_tables: Optional[List[torch.Tensor]] = None,
) -> None:
    if not _enabled():
        return
    if not forward_batch.forward_mode.is_decode():
        return
    layer_id = int(getattr(layer, "layer_id", -1))
    if not _layer_allowed(layer_id):
        return

    try:
        _maybe_sync_cuda()
        root = os.path.abspath(os.path.expanduser(_dump_root()))
        run = _run_id()
        tag = _path_tag(forward_batch)
        out_dir = os.path.join(root, f"run_{run}", tag, "fa_backend_decode")
        os.makedirs(out_dir, exist_ok=True)

        tp_r = int(get_tensor_model_parallel_rank())
        pos = getattr(forward_batch, "positions", None)
        pos_tag = -1
        if torch.is_tensor(pos) and pos.numel() > 0:
            pos_tag = int(pos.view(-1)[0].item())
        decode_suffix = _fa_decode_dump_filename_suffix(forward_batch, pos)
        fn = f"fa_decode_layer_{layer_id:04d}_tp{tp_r}{decode_suffix}.pt"
        path = os.path.join(out_dir, fn)

        decode_align = _snapshot_decode_align_fields(forward_batch, pos)
        payload: Dict[str, Any] = {
            "dump_kind": "fa_backend_decode",
            "version": 2,
            "layer_id": layer_id,
            "tp_rank": tp_r,
            "pos_tag": pos_tag,
            "decode_filename_suffix": decode_suffix,
            **decode_align,
            "dbg_flags": dict(dbg_flags),
            "forward_batch_scalars": _snapshot_forward_batch_scalars(forward_batch),
            "forward_batch_tensors": _snapshot_forward_batch_tensors(forward_batch),
            "flash_forward_metadata": _snapshot_metadata_obj(
                getattr(backend, "forward_metadata", None), "forward_metadata"
            ),
            "req_to_token_rows": _req_to_token_rows(forward_batch, backend),
            "kv_buffers": _build_kv_payload(
                forward_batch,
                backend,
                layer_id,
                bool(dbg_flags.get("use_mla")),
                fa_kv_page_tables,
            ),
            "inputs_q": _to_cpu(q),
            "inputs_k": _to_cpu(k),
            "inputs_v": _to_cpu(v),
            "inputs_q_rope": _to_cpu(q_rope),
            "inputs_k_rope": _to_cpu_inputs_k_rope(
                k_rope, forward_batch, backend, fa_kv_page_tables
            ),
            "inputs_sinks": _to_cpu(sinks),
            "out_o_before_view": _to_cpu(o),
        }
        mexp = getattr(backend, "forward_metadata_spec_decode_expand", None)
        if mexp is not None and bool(dbg_flags.get("use_cascade_attn")):
            payload["flash_forward_metadata_spec_decode_expand"] = _snapshot_metadata_obj(
                mexp, "forward_metadata_spec_decode_expand"
            )

        torch.save(payload, path)
        logger.info(
            "[fa_backend_decode_dump] saved layer=%s tp=%s pos=%s path=%s",
            layer_id,
            tp_r,
            pos_tag,
            path,
        )
    except Exception:
        logger.exception("[fa_backend_decode_dump] failed layer=%s", layer_id)
