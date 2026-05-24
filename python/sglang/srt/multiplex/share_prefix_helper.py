"""
Share Prefix Helper for Special DP Attention.

Implements cross-DP-rank prefix KV cache sharing during split_prefill.

Design overview
---------------
In special DP attention, each DP group has its own radix-tree / KV cache, so different
DP ranks may have prefix KVs of different lengths for the same request.  During
split_prefill we run every rank's extend tokens through the full TP computation.  By
sharing the **longest** prefix across all DP ranks (all-reduce on a transfer buffer),
every rank can attend to more prefix tokens and therefore compute shorter extends.

Key data flow per forward batch
--------------------------------
1. [schedule_batch.prepare_for_extend] All-gather local prefix_lens across DP ranks.
2. Compute per-request metadata: max_prefix, min_prefix, transfer_len, max_rank.
3. Pre-allocate transfer_buffer  [total_transfer_tokens, 1, kv_cache_dim].
4. [per-layer, in deepseek_v2.py]
   a. fill_transfer_buffer_for_layer  – max_rank (attn_tp_rank==0 only) copies
      KV[min_prefix : max_prefix] into transfer_buffer; others leave zeros.
   b. dist.all_reduce(transfer_buffer) – every rank now has complete transfer KV.
   c. build_combined_kv_for_layer  – concatenate [local_prefix | transfer | extend].
   d. flash_attn_with_kvcache on the combined KV.
5. Save extend KV to dp-local out_cache_loc as usual.

Communication groups
--------------------
* All-gather  : tp_group (spans all DP groups within one worker).
* All-reduce  : tp_group (same).
* Only attn_tp_rank == 0 fills transfer_buffer to avoid N-fold duplication when
  attn_tp_size > 1 (all ranks within one DP group share the same latent KV cache).

Buffer layout (no request reorder for simplicity)
--------------------------------------------------
transfer_buffer slots are laid out in original batch order:
  [req0_transfer | req1_transfer | req2_transfer | ...]
with transfer_offsets[i] = sum(transfer_len[0:i]).

Combined KV per request (in build_combined_kv_for_layer) is also in original order:
  [req0_local_prefix | req0_transfer | req0_extend | req1_... | ...]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GatheredPrefixData: result of the early all-gather step
# ---------------------------------------------------------------------------

@dataclass
class GatheredPrefixData:
    """Results of the early prefix all-gather, usable by both prepare_for_extend
    (to adjust input_ids) and compute_share_prefix_info (to skip re-gathering).

    Computed by gather_prefix_data() which must run before input_ids are sliced in
    prepare_for_extend so that all ranks use max_prefix as the effective prefix.
    """
    local_prefix_lens: List[int]   # len(req.prefix_indices) per request, THIS rank
    seq_lens: List[int]            # total len(req.fill_ids) per request
    max_prefix_list: List[int]     # max prefix across all DP ranks per request
    min_prefix_list: List[int]     # min prefix across all DP ranks per request
    transfer_len_list: List[int]   # = max_prefix - min_prefix per request
    max_rank_list: List[int]       # DP rank that holds the longest prefix
    total_transfer: int            # sum of transfer_len_list
    local_dp_rank: int
    local_attn_tp_rank: int


@dataclass
class SharePrefixBatchInfo:
    """Per-batch metadata and buffers for share-prefix in special DP attention.

    The transfer_buffer is NOT allocated during compute_share_prefix_info because
    kv_cache_dim is only known inside the model (deepseek_v2.py).  Call
    ensure_transfer_buffer_allocated() from the model before the first layer.

    When total_transfer_tokens == 0 (all DP ranks have identical prefix lengths),
    the transfer path is skipped but we still use this struct so that:
      - combined_kv is built from local prefix + extend (no real transfer)
      - attention uses the custom combined_kv page_table (avoids the broken
        req_to_token_pool page_table for non-dp-local requests)
      - KV is correctly written back via save_dp_local_kv
    """

    # ---------- per-request lists (original batch order) ----------
    seq_lens: List[int]             # total sequence lengths
    local_prefix: List[int]         # prefix len on THIS DP rank
    min_prefix: List[int]           # min across all DP ranks
    max_prefix: List[int]           # max across all DP ranks
    transfer_len: List[int]         # = max_prefix - min_prefix
    max_rank: List[int]             # attn_dp_rank with max prefix

    # ---------- rank identity ----------
    local_dp_rank: int
    local_attn_tp_rank: int

    # ---------- buffer layout ----------
    transfer_offsets: List[int]     # exclusive prefix-sum of transfer_lens
    total_transfer_tokens: int

    # cumulative original extend starts (for slicing k_nope / k_pe)
    # extend_hidden_starts[i] = sum(seq_lens[j] - local_prefix[j]  for j in 0..i-1)
    extend_hidden_starts: List[int]

    # ---------- prefix slot indices (GPU tensors, one per request) ----------
    # prefix_indices_list[i] = GPU int64 tensor of token_to_kv_pool slot indices
    # for the prefix tokens of request i.  len(prefix_indices_list[i]) == local_prefix[i].
    prefix_indices_list: List[torch.Tensor]

    # ---------- dp-local request indices ----------
    # dp_local_req_global_indices[j] = global batch index of the j-th dp-local request.
    # A request is "dp-local" if its decode_dp_rank matches this rank's dp rank.
    # out_cache_loc is laid out in this order with extend_len = seq_len - local_prefix
    # slots per dp-local request.
    dp_local_req_global_indices: List[int]

    # ---------- preallocated reusable buffer (overwritten each layer) ----------
    # shape: [total_transfer_tokens, 1, kv_cache_dim]
    # None until ensure_transfer_buffer_allocated() is called.
    transfer_buffer: Optional[torch.Tensor]

    # ---------- misc ----------
    # Set by ensure_transfer_buffer_allocated()
    kv_cache_dim: int = 0           # kv_lora_rank + qk_rope_head_dim
    kv_lora_rank: int = 0           # for splitting combined KV into rope / nope
    device: Optional[torch.device] = None

    def ensure_transfer_buffer_allocated(
        self,
        kv_cache_dim: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate transfer_buffer if not yet done; idempotent.

        When total_transfer_tokens == 0, allocates a zero-element tensor
        (shape [0, 1, kv_cache_dim]) which is safe for all downstream ops.
        """
        if self.transfer_buffer is not None:
            return
        self.kv_cache_dim = kv_cache_dim
        self.kv_lora_rank = kv_lora_rank
        self.device = device
        self.transfer_buffer = torch.zeros(
            (self.total_transfer_tokens, 1, kv_cache_dim),
            dtype=dtype,
            device=device,
        )
        logger.debug(
            "[share_prefix] allocated transfer_buffer shape=%s dtype=%s",
            tuple(self.transfer_buffer.shape), dtype,
        )


# ---------------------------------------------------------------------------
# Helper: gather_prefix_data  (early all-gather, runs before input_ids)
# ---------------------------------------------------------------------------

def gather_prefix_data(
    reqs,             # list of Req objects with .prefix_indices and .fill_ids
    tp_group,         # GroupCoordinator (get_tp_group())
    device: torch.device,
) -> "GatheredPrefixData":
    """
    All-gather local prefix lengths across the full TP/DP group and compute
    per-request metadata (max_prefix, min_prefix, etc.).

    This is extracted so that prepare_for_extend can call it BEFORE slicing
    input_ids – every rank then uses max_prefix as the effective base so all
    ranks feed the same number of tokens to the model per request.

    Returns GatheredPrefixData (never None, but may be empty if n==0).
    """
    from sglang.srt.layers.dp_attention import (
        get_attention_dp_rank,
        get_attention_dp_size,
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    n = len(reqs)
    local_dp_rank: int = int(get_attention_dp_rank())
    local_attn_tp_rank: int = int(get_attention_tp_rank())
    dp_size: int = int(get_attention_dp_size())
    tp_size: int = int(get_attention_tp_size())

    local_prefix_lens_cpu = [len(req.prefix_indices) for req in reqs]
    seq_lens_cpu = [len(req.fill_ids) for req in reqs]

    logger.debug(
        "[share_prefix] early-gather dp_rank=%d attn_tp_rank=%d "
        "local_prefix_lens=%s seq_lens=%s",
        local_dp_rank, local_attn_tp_rank, local_prefix_lens_cpu, seq_lens_cpu,
    )

    # All-gather across tp_group (covers all DP ranks × attn_tp ranks)
    local_tensor = torch.tensor(
        local_prefix_lens_cpu, dtype=torch.int64, device=device
    )
    gathered = [torch.empty_like(local_tensor) for _ in range(tp_size * dp_size)]
    dist.all_gather(gathered, local_tensor, group=tp_group.device_group)

    # dp_prefix_matrix[dp_rank][req_idx] = prefix len (use attn_tp_rank 0 per DP group)
    dp_prefix_matrix: List[List[int]] = []
    for dp_r in range(dp_size):
        idx = dp_r * tp_size + 0
        dp_prefix_matrix.append(gathered[idx].tolist())

    if local_dp_rank == 0 and local_attn_tp_rank == 0:
        logger.info(
            "[share_prefix] all-gathered prefix matrix (dp_rank x req_idx):\n%s",
            "\n".join(
                f"  dp{dp_r}: {dp_prefix_matrix[dp_r]}" for dp_r in range(dp_size)
            ),
        )

    max_prefix_list: List[int] = []
    min_prefix_list: List[int] = []
    transfer_len_list: List[int] = []
    max_rank_list: List[int] = []

    for i in range(n):
        prefix_per_rank = [dp_prefix_matrix[r][i] for r in range(dp_size)]
        max_p = max(prefix_per_rank)
        min_p = min(prefix_per_rank)
        max_r = prefix_per_rank.index(max_p)
        t_len = max_p - min_p
        max_prefix_list.append(max_p)
        min_prefix_list.append(min_p)
        transfer_len_list.append(t_len)
        max_rank_list.append(max_r)

    total_transfer = sum(transfer_len_list)

    if local_dp_rank == 0 and local_attn_tp_rank == 0:
        for i in range(n):
            logger.info(
                "[share_prefix] req[%d]: seq_len=%d local_prefix=%d "
                "min=%d max=%d transfer_len=%d max_rank=%d",
                i, seq_lens_cpu[i], local_prefix_lens_cpu[i],
                min_prefix_list[i], max_prefix_list[i],
                transfer_len_list[i], max_rank_list[i],
            )

    if total_transfer == 0:
        logger.debug(
            "[share_prefix] dp_rank=%d total_transfer=0; no actual transfer needed.",
            local_dp_rank,
        )

    return GatheredPrefixData(
        local_prefix_lens=local_prefix_lens_cpu,
        seq_lens=seq_lens_cpu,
        max_prefix_list=max_prefix_list,
        min_prefix_list=min_prefix_list,
        transfer_len_list=transfer_len_list,
        max_rank_list=max_rank_list,
        total_transfer=total_transfer,
        local_dp_rank=local_dp_rank,
        local_attn_tp_rank=local_attn_tp_rank,
    )


# ---------------------------------------------------------------------------
# Helper: compute_share_prefix_info
# ---------------------------------------------------------------------------

def compute_share_prefix_info(
    batch,                        # ScheduleBatch
    tp_group,                     # GroupCoordinator (get_tp_group())
    device: torch.device,
    gathered_data: Optional["GatheredPrefixData"] = None,
) -> Optional["SharePrefixBatchInfo"]:
    """
    Build a SharePrefixBatchInfo for the batch.  If *gathered_data* is provided
    (from an earlier call to gather_prefix_data()), the all-gather step is
    skipped entirely – this avoids a second collective when prepare_for_extend
    already ran gather_prefix_data() to adjust input_ids.

    Always returns a SharePrefixBatchInfo (never None for non-empty batches), even
    when total_transfer_tokens == 0.  This ensures the share-prefix attention path
    is always used, avoiding the broken req_to_token_pool page_table for non-dp-local
    requests.  When total_transfer_tokens == 0, fill + all_reduce are skipped and
    combined_kv contains only local-prefix + extend KV.

    Returns None only if the batch is empty.

    Parameters
    ----------
    batch        : ScheduleBatch – must have .reqs, .dp_local_req_indices set.
    tp_group     : GroupCoordinator – wraps the full TP process group.
    device       : CUDA device for tensor allocation.
    gathered_data: Optional pre-computed GatheredPrefixData from gather_prefix_data().
    """
    from sglang.srt.layers.dp_attention import (
        get_attention_dp_rank,
        get_attention_dp_size,
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    reqs = batch.reqs
    n = len(reqs)
    if n == 0:
        return None

    # ------------------------------------------------------------------ #
    # 1. All-gather (or reuse cached data)                                  #
    # ------------------------------------------------------------------ #
    if gathered_data is None:
        gathered_data = gather_prefix_data(reqs, tp_group, device)

    local_prefix_lens_cpu = gathered_data.local_prefix_lens
    seq_lens_cpu = gathered_data.seq_lens
    max_prefix_list = gathered_data.max_prefix_list
    min_prefix_list = gathered_data.min_prefix_list
    transfer_len_list = gathered_data.transfer_len_list
    max_rank_list = gathered_data.max_rank_list
    total_transfer = gathered_data.total_transfer
    local_dp_rank = gathered_data.local_dp_rank
    local_attn_tp_rank = gathered_data.local_attn_tp_rank

    # ------------------------------------------------------------------ #
    # 2. Build GPU prefix_indices_list                                      #
    # ------------------------------------------------------------------ #
    prefix_indices_list: List[torch.Tensor] = []
    for req in reqs:
        if len(req.prefix_indices) > 0:
            pi = torch.tensor(
                req.prefix_indices
                if not isinstance(req.prefix_indices, torch.Tensor)
                else req.prefix_indices.tolist(),
                dtype=torch.int64, device=device,
            )
        else:
            pi = torch.empty(0, dtype=torch.int64, device=device)
        prefix_indices_list.append(pi)

    # ------------------------------------------------------------------ #
    # 3. Compute offsets                                                    #
    # ------------------------------------------------------------------ #
    transfer_offsets: List[int] = []
    acc = 0
    for tl in transfer_len_list:
        transfer_offsets.append(acc)
        acc += tl

    # extend_hidden_starts[i] = offset into the flattened hidden_states / k_nope.
    #
    # IMPORTANT: Because input_ids is sliced from max_prefix (not local_prefix),
    # the model processes exactly (seq_len - max_prefix) tokens per request.
    # extend_hidden_starts must therefore accumulate (seq_len - max_prefix) per
    # request so that Part C of build_combined_kv_for_layer and save_dp_local_kv
    # index k_nope/k_pe correctly.
    extend_hidden_starts: List[int] = []
    acc = 0
    for i in range(n):
        extend_hidden_starts.append(acc)
        acc += seq_lens_cpu[i] - max_prefix_list[i]   # <-- max_prefix, not local_prefix

    logger.debug(
        "[share_prefix] dp_rank=%d extend_hidden_starts=%s (based on max_prefix)",
        local_dp_rank, extend_hidden_starts,
    )

    # ------------------------------------------------------------------ #
    # 4. Collect dp-local request global indices                            #
    # ------------------------------------------------------------------ #
    dp_local_req_global_indices: List[int] = list(
        getattr(batch, "dp_local_req_indices", [])
    )
    logger.debug(
        "[share_prefix] dp_rank=%d dp_local_req_global_indices=%s",
        local_dp_rank, dp_local_req_global_indices,
    )

    # ------------------------------------------------------------------ #
    # 5. Return metadata (transfer_buffer allocated lazily later)          #
    # ------------------------------------------------------------------ #
    return SharePrefixBatchInfo(
        seq_lens=seq_lens_cpu,
        local_prefix=local_prefix_lens_cpu,
        min_prefix=min_prefix_list,
        max_prefix=max_prefix_list,
        transfer_len=transfer_len_list,
        max_rank=max_rank_list,
        local_dp_rank=local_dp_rank,
        local_attn_tp_rank=local_attn_tp_rank,
        transfer_offsets=transfer_offsets,
        total_transfer_tokens=total_transfer,
        extend_hidden_starts=extend_hidden_starts,
        prefix_indices_list=prefix_indices_list,
        dp_local_req_global_indices=dp_local_req_global_indices,
        transfer_buffer=None,  # allocated lazily by ensure_transfer_buffer_allocated()
        device=device,
    )


# ---------------------------------------------------------------------------
# Helper: fill_transfer_buffer_for_layer
# ---------------------------------------------------------------------------

def fill_transfer_buffer_for_layer(
    info: SharePrefixBatchInfo,
    layer_id: int,
    kv_buf: torch.Tensor,  # token_to_kv_pool.get_key_buffer(layer_id)
) -> None:
    """
    Fill info.transfer_buffer with KV[min_prefix : max_prefix] from the local
    KV cache for requests where this rank is max_rank.

    Only fills when:  info.local_dp_rank == max_rank[i]
                  AND info.local_attn_tp_rank == 0

    All other positions in transfer_buffer remain zero (for all_reduce correctness).

    Parameters
    ----------
    kv_buf : token_to_kv_pool.get_key_buffer(layer_id)
             shape [total_slots, 1, kv_cache_dim]
    """
    if info.local_attn_tp_rank != 0:
        # Other attn_tp_ranks within this DP group share the same KV cache.
        # Only rank 0 fills to avoid N-fold accumulation after all_reduce.
        return

    n = len(info.seq_lens)
    kv_dtype = info.transfer_buffer.dtype

    filled_reqs = []
    for i in range(n):
        if info.max_rank[i] != info.local_dp_rank:
            continue
        t_len = info.transfer_len[i]
        if t_len == 0:
            continue

        min_p = info.min_prefix[i]
        max_p = info.max_prefix[i]  # = min_p + t_len
        t_start = info.transfer_offsets[i]

        # pref_idx[min_p:max_p] are the KV pool slot indices for the transfer region
        pref_idx = info.prefix_indices_list[i]  # length = local_prefix[i] >= max_p
        slots = pref_idx[min_p:max_p]  # [t_len] int64

        kv_slice = kv_buf[slots]  # [t_len, 1, kv_cache_dim]
        if kv_slice.dtype != kv_dtype:
            kv_slice = kv_slice.to(kv_dtype)

        info.transfer_buffer[t_start : t_start + t_len] = kv_slice
        filled_reqs.append((i, min_p, max_p, t_len))

    if filled_reqs:
        logger.debug(
            "[share_prefix] layer=%d dp_rank=%d filled transfer_buffer for reqs: %s",
            layer_id, info.local_dp_rank,
            [f"req{i}: [{mn},{mx}) len={tl}" for i, mn, mx, tl in filled_reqs],
        )


# ---------------------------------------------------------------------------
# Helper: build_combined_kv_for_layer
# ---------------------------------------------------------------------------

def build_combined_kv_for_layer(
    info: SharePrefixBatchInfo,
    layer_id: int,
    k_nope: torch.Tensor,  # [total_extend_tokens, 1, kv_lora_rank]
    k_pe: torch.Tensor,    # [total_extend_tokens, 1, qk_rope_head_dim]
    kv_buf: torch.Tensor,  # token_to_kv_pool.get_key_buffer(layer_id)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a contiguous combined KV tensor for all requests in the batch.

    For request i the combined KV has three parts:
      - Part A [0, min_prefix[i])          : local prefix from kv pool
      - Part B [min_prefix[i], max_prefix[i]): transfer KV (all-reduced)
      - Part C [max_prefix[i], seq_lens[i]) : new extend KV (from k_nope / k_pe)

    Returns
    -------
    combined_kv : [total_combined_tokens, 1, kv_cache_dim]
    page_table  : [batch_size, max_seq_len] int32 – slot index into combined_kv
    cache_seqlens : [batch_size] int32 – = seq_lens[i]
    """
    n = len(info.seq_lens)
    kv_cache_dim = info.kv_cache_dim
    kv_lora_rank = info.kv_lora_rank
    device = info.device
    dtype = k_nope.dtype

    total_combined = sum(info.seq_lens)
    combined_kv = torch.empty(
        (total_combined, 1, kv_cache_dim), dtype=dtype, device=device
    )

    # Cast pool buffer dtype if needed
    if kv_buf.dtype != dtype:
        kv_buf_cast = kv_buf.to(dtype)
    else:
        kv_buf_cast = kv_buf

    transfer_buf = info.transfer_buffer
    if transfer_buf.dtype != dtype:
        transfer_buf = transfer_buf.to(dtype)

    current_kv_offset = 0  # offset into combined_kv

    for i in range(n):
        min_p = info.min_prefix[i]
        max_p = info.max_prefix[i]
        local_p = info.local_prefix[i]
        seq_len = info.seq_lens[i]
        pref_idx = info.prefix_indices_list[i]

        # ---- Part A: local prefix [0, min_p) ----
        if min_p > 0:
            slots_a = pref_idx[:min_p]
            combined_kv[current_kv_offset : current_kv_offset + min_p] = (
                kv_buf_cast[slots_a]
            )

        # ---- Part B: transfer [min_p, max_p) ----
        t_len = info.transfer_len[i]
        if t_len > 0:
            t_start = info.transfer_offsets[i]
            combined_kv[current_kv_offset + min_p : current_kv_offset + max_p] = (
                transfer_buf[t_start : t_start + t_len]
            )

        # ---- Part C: extend [max_p, seq_len) ----
        e_len = seq_len - max_p
        if e_len > 0:
            h_start = info.extend_hidden_starts[i]  # start in k_nope for this req
            # input_ids is sliced from max_prefix on all ranks, so k_nope/k_pe for
            # request i begins at extend_hidden_starts[i] with no intra-request
            # offset – the very first token in k_nope for this request corresponds
            # to sequence position max_p.
            h_slice_start = h_start
            h_slice_end = h_start + e_len

            kv_offset = current_kv_offset + max_p

            # k_nope/k_pe are [total_extend, 1, dim]
            combined_kv[kv_offset : kv_offset + e_len, 0, :kv_lora_rank] = (
                k_nope[h_slice_start:h_slice_end, 0]
            )
            combined_kv[kv_offset : kv_offset + e_len, 0, kv_lora_rank:] = (
                k_pe[h_slice_start:h_slice_end, 0]
            )

        current_kv_offset += seq_len

    # ------------------------------------------------------------------ #
    # Build page_table [batch_size, max_seq_len]                           #
    # Each request i occupies consecutive slots [start_i, start_i + seq_lens[i]).
    # ------------------------------------------------------------------ #
    max_seq_len = max(info.seq_lens)
    page_table = torch.zeros(
        (n, max_seq_len), dtype=torch.int32, device=device
    )
    current_kv_offset = 0
    for i in range(n):
        seq_len = info.seq_lens[i]
        page_table[i, :seq_len] = torch.arange(
            current_kv_offset, current_kv_offset + seq_len,
            dtype=torch.int32, device=device,
        )
        current_kv_offset += seq_len

    cache_seqlens = torch.tensor(info.seq_lens, dtype=torch.int32, device=device)

    if layer_id == getattr(build_combined_kv_for_layer, "_log_layer", None):
        logger.debug(
            "[share_prefix] build_combined_kv: combined_kv.shape=%s "
            "page_table.shape=%s cache_seqlens=%s",
            tuple(combined_kv.shape), tuple(page_table.shape),
            cache_seqlens.tolist(),
        )

    return combined_kv, page_table, cache_seqlens


# ---------------------------------------------------------------------------
# Convenience: zero out transfer_buffer before filling (to be safe)
# ---------------------------------------------------------------------------

def reset_transfer_buffer(info: SharePrefixBatchInfo) -> None:
    """Zero the transfer buffer before each layer's fill + all_reduce."""
    info.transfer_buffer.zero_()


# ---------------------------------------------------------------------------
# Helper: save_dp_local_kv
# ---------------------------------------------------------------------------

def save_dp_local_kv(
    info: SharePrefixBatchInfo,
    layer,                        # RadixAttention – used for layer_id lookup
    k_nope: torch.Tensor,         # [total_extend_tokens, 1, kv_lora_rank]
    k_pe: torch.Tensor,           # [total_extend_tokens, 1, qk_rope_head_dim]
    out_cache_loc: torch.Tensor,  # [total_dp_local_extend_tokens] – flat slot indices
    token_to_kv_pool,             # MLATokenToKVPool (or wrapper)
) -> None:
    """
    Write KV cache for all dp-local requests after share-prefix attention.

    For each dp-local request (decode_dp_rank == this rank), we write two
    segments into the pre-allocated out_cache_loc slots:

    Segment 1 – Transfer region [local_prefix, max_prefix):
        KV comes from info.transfer_buffer (already all-reduced).
        transfer_buffer holds positions [min_prefix, max_prefix) per request;
        for this rank, we need the sub-range [local_prefix, max_prefix), which
        sits at offset (local_prefix - min_prefix) inside the request's transfer
        slice.

    Segment 2 – Extend region [max_prefix, seq_len):
        KV comes from k_nope / k_pe at the appropriate extend_hidden_starts offset.

    Parameters
    ----------
    info           : SharePrefixBatchInfo – must have transfer_buffer allocated.
    layer          : RadixAttention object for layer_id (same as used in flash_attn).
    k_nope         : [total_extend, 1, kv_lora_rank] – new k_nope for all extend tokens.
    k_pe           : [total_extend, 1, qk_rope_head_dim] – new k_pe for all extend tokens.
    out_cache_loc  : flat token pool slot indices for dp-local requests, in dp-local order.
    token_to_kv_pool : KV pool; must expose set_mla_kv_buffer(layer, loc, k_nope, k_rope).
    """
    oc_offset = 0           # running index into out_cache_loc

    total_transfer_written = 0
    total_extend_written = 0

    for g in info.dp_local_req_global_indices:
        local_p = info.local_prefix[g]
        max_p   = info.max_prefix[g]
        min_p   = info.min_prefix[g]
        seq_len = info.seq_lens[g]
        extend_len = seq_len - local_p          # total out_cache_loc slots for this req

        oc_slice = out_cache_loc[oc_offset : oc_offset + extend_len]
        oc_offset += extend_len

        # ---- Segment 1: transfer region [local_p, max_p) ----
        transfer_local_len = max_p - local_p    # 0 for donor rank (local_p == max_p)
        if transfer_local_len > 0:
            # transfer_buffer contains positions [min_p, max_p) for this request.
            # This rank's [local_p, max_p) starts at offset (local_p - min_p).
            t_src_start = info.transfer_offsets[g] + (local_p - min_p)
            t_src_end   = info.transfer_offsets[g] + (max_p - min_p)
            transfer_kv = info.transfer_buffer[t_src_start : t_src_end]  # [tl, 1, kv_cache_dim]

            transfer_slots  = oc_slice[:transfer_local_len]
            xfer_k_nope = transfer_kv[:, :, : info.kv_lora_rank]
            xfer_k_rope = transfer_kv[:, :, info.kv_lora_rank :]
            token_to_kv_pool.set_mla_kv_buffer(layer, transfer_slots, xfer_k_nope, xfer_k_rope)
            total_transfer_written += transfer_local_len

        # ---- Segment 2: extend region [max_p, seq_len) ----
        true_ext_len = seq_len - max_p          # 0 if all tokens were cached/transferred
        if true_ext_len > 0:
            h_start = info.extend_hidden_starts[g]
            # input_ids is sliced from max_prefix on every rank, so k_nope for
            # request g starts exactly at h_start with no intra-request offset.
            ext_k_nope = k_nope[h_start : h_start + true_ext_len]   # [true_ext_len, 1, kv_lora_rank]
            ext_k_pe   = k_pe  [h_start : h_start + true_ext_len]   # [true_ext_len, 1, qk_rope_head_dim]

            extend_slots = oc_slice[transfer_local_len:]
            token_to_kv_pool.set_mla_kv_buffer(layer, extend_slots, ext_k_nope, ext_k_pe)
            total_extend_written += true_ext_len

    logger.debug(
        "[share_prefix] save_dp_local_kv: dp_rank=%d layer=%d "
        "dp_local_reqs=%d transfer_tokens=%d extend_tokens=%d",
        info.local_dp_rank,
        getattr(layer, "layer_id", -1),
        len(info.dp_local_req_global_indices),
        total_transfer_written,
        total_extend_written,
    )
