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
3. Pre-allocate layout tensors (page_table, cache_seqlens, cu_seqlens_k_new,
   all_local_src_indices) once – these do not change across layers.
4. [per-layer, in deepseek_v2.py]
   a. fill_transfer_region_for_layer – max_rank (attn_tp_rank==0 only) writes
      KV[min_prefix : max_prefix] directly into combined_kv_buf's transfer_block;
      others leave zeros.
   b. dist.all_reduce(transfer_block) – every rank now has complete transfer KV.
   c. fill_local_and_extend_for_layer – batched-gather local_block from kv_pool,
      two contiguous writes for extend_block from k_nope / k_pe.
   d. flash_attn_with_kvcache on combined_kv_buf using pre-computed page_table.
5. save_dp_local_kv reads transfer/extend KV directly from combined_kv_buf and
   writes to dp-local out_cache_loc slots.

Buffer layout (block layout, single allocation per batch)
---------------------------------------------------------
combined_kv_buf  [total_combined, 1, kv_cache_dim]
  [0 .. local_block_size)           local_block:    req0_local | req1_local | ...
  [local_block_size .. extend_block_start)  transfer_block: req0_xfer | req1_xfer | ...
  [extend_block_start .. total_combined)    extend_block:   req0_ext  | req1_ext  | ...

where:
  local_block_size  = sum(min_prefix)
  extend_block_start = local_block_size + total_transfer_tokens
  total_combined    = sum(seq_lens)

This replaces the old design that had:
  - a per-layer torch.empty for combined_kv      (N_layers mallocs eliminated)
  - a per-layer torch.zeros for page_table       (N_layers mallocs eliminated)
  - a per-layer torch.tensor for cache_seqlens   (N_layers mallocs eliminated)
  - a separate transfer_buffer + per-layer copy  (N_layers copies eliminated)

Communication groups
--------------------
* All-gather  : tp_group (spans all DP groups within one worker).
* All-reduce  : tp_group (same).
* Only attn_tp_rank == 0 fills transfer_region to avoid N-fold duplication when
  attn_tp_size > 1 (all ranks within one DP group share the same latent KV cache).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

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

    Layout
    ------
    combined_kv_buf [total_combined, 1, kv_cache_dim] is pre-allocated once per
    batch (lazily on first layer when kv_cache_dim is known) and reused for all
    layers.  It has a block layout:

        [local_block | transfer_block | extend_block]

    page_table_cached, cache_seqlens_tensor, cu_seqlens_k_new_tensor and
    all_local_src_indices are computed once in compute_share_prefix_info and
    reused each layer.

    When total_transfer_tokens == 0 (all DP ranks have identical prefix lengths),
    the transfer path is skipped (transfer_block is empty) but we still use this
    struct so that:
      - combined_kv_buf is built from local prefix + extend (no real transfer)
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

    # ---------- buffer layout offsets (CPU scalars) ----------
    transfer_offsets: List[int]     # exclusive prefix-sum of transfer_lens
    total_transfer_tokens: int

    # local_block_size  = sum(min_prefix)
    # extend_block_start = local_block_size + total_transfer_tokens
    local_block_size: int
    extend_block_start: int

    # Per-request start offsets inside local_block (cumsum of min_prefix)
    local_block_starts: List[int]

    # cumulative original extend starts (for indexing extend_block)
    # extend_hidden_starts[i] = sum(seq_lens[j] - max_prefix[j]  for j in 0..i-1)
    extend_hidden_starts: List[int]

    # ---------- prefix slot indices (GPU tensors, one per request) ----------
    prefix_indices_list: List[torch.Tensor]

    # ---------- dp-local request indices ----------
    dp_local_req_global_indices: List[int]

    # ---------- precomputed GPU tensors (layout-only, no kv_cache_dim needed) ----------
    # all_local_src_indices[j] = kv_buf slot for the j-th token in local_block
    # shape [local_block_size], dtype int64
    all_local_src_indices: Optional[torch.Tensor]

    # page_table[i, pos] = slot index in combined_kv_buf for request i, position pos
    # shape [batch_size, max_seq_len], dtype int32
    page_table_cached: Optional[torch.Tensor]

    # cache_seqlens[i] = seq_lens[i]
    # shape [batch_size], dtype int32
    cache_seqlens_tensor: Optional[torch.Tensor]

    # cu_seqlens_k_new = pad(cumsum(cache_seqlens_tensor))
    # shape [batch_size + 1], dtype int32
    cu_seqlens_k_new_tensor: Optional[torch.Tensor]

    # ---------- preallocated reusable buffer (overwritten each layer) ----------
    # combined_kv_buf: [total_combined, 1, kv_cache_dim]
    # None until ensure_combined_buf_allocated() is called.
    combined_kv_buf: Optional[torch.Tensor]

    # ---------- misc ----------
    kv_cache_dim: int = 0           # kv_lora_rank + qk_rope_head_dim
    kv_lora_rank: int = 0           # for splitting combined KV into rope / nope
    device: Optional[torch.device] = None

    def ensure_combined_buf_allocated(
        self,
        kv_cache_dim: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Allocate combined_kv_buf if not yet done; idempotent.

        Only combined_kv_buf requires kv_cache_dim and is deferred to the first
        model layer.  All layout tensors (page_table_cached, etc.) are already
        allocated in compute_share_prefix_info.
        """
        if self.combined_kv_buf is not None:
            return
        self.kv_cache_dim = kv_cache_dim
        self.kv_lora_rank = kv_lora_rank
        self.device = device
        total_combined = sum(self.seq_lens)
        self.combined_kv_buf = torch.empty(
            (total_combined, 1, kv_cache_dim),
            dtype=dtype,
            device=device,
        )
        logger.debug(
            "[share_prefix] allocated combined_kv_buf shape=%s dtype=%s",
            tuple(self.combined_kv_buf.shape), dtype,
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
    combined_kv_buf contains only local-prefix + extend KV.

    Returns None only if the batch is empty.

    In addition to the basic metadata, this function pre-computes all layout
    tensors that are constant across layers:
      - all_local_src_indices  : batched kv_buf slot indices for local_block fill
      - page_table_cached      : [n, max_seq_len] int32, block-layout mapping
      - cache_seqlens_tensor   : [n] int32
      - cu_seqlens_k_new_tensor: [n+1] int32

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
    # 3. Compute transfer offsets                                           #
    # ------------------------------------------------------------------ #
    transfer_offsets: List[int] = []
    acc = 0
    for tl in transfer_len_list:
        transfer_offsets.append(acc)
        acc += tl

    # extend_hidden_starts[i] = offset into k_nope/k_pe for request i.
    # Because input_ids is sliced from max_prefix on all ranks, the model
    # processes exactly (seq_len - max_prefix) tokens per request.
    extend_hidden_starts: List[int] = []
    acc = 0
    for i in range(n):
        extend_hidden_starts.append(acc)
        acc += seq_lens_cpu[i] - max_prefix_list[i]

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
    # 5. Block-layout offsets (CPU scalars)                                 #
    # ------------------------------------------------------------------ #
    # local_block_starts[i] = starting slot index in local_block for request i
    local_block_starts: List[int] = []
    acc = 0
    for mp in min_prefix_list:
        local_block_starts.append(acc)
        acc += mp
    local_block_size: int = acc  # = sum(min_prefix)

    extend_block_start: int = local_block_size + total_transfer

    # ------------------------------------------------------------------ #
    # 6. Pre-compute GPU layout tensors (all layer-invariant)              #
    # ------------------------------------------------------------------ #

    # all_local_src_indices: concatenation of pref_idx[:min_p] for each req
    # Shape: [local_block_size], dtype int64
    if local_block_size > 0:
        all_local_src_indices: torch.Tensor = torch.cat([
            prefix_indices_list[i][:min_prefix_list[i]]
            for i in range(n)
            if min_prefix_list[i] > 0
        ])
    else:
        all_local_src_indices = torch.empty(0, dtype=torch.int64, device=device)

    # page_table_cached [n, max_seq_len] int32
    # Block layout: position j in request i maps to:
    #   [0, min_p)   → local_block_starts[i] + j
    #   [min_p, max_p) → local_block_size + transfer_offsets[i] + (j - min_p)
    #   [max_p, seq_len) → extend_block_start + extend_hidden_starts[i] + (j - max_p)
    max_seq_len = max(seq_lens_cpu)
    page_table_cached = torch.zeros(
        (n, max_seq_len), dtype=torch.int32, device=device
    )
    for i in range(n):
        min_p = min_prefix_list[i]
        max_p = max_prefix_list[i]
        seq_len = seq_lens_cpu[i]
        t_len = transfer_len_list[i]
        e_len = seq_len - max_p

        if min_p > 0:
            page_table_cached[i, :min_p] = torch.arange(
                local_block_starts[i],
                local_block_starts[i] + min_p,
                dtype=torch.int32, device=device,
            )
        if t_len > 0:
            page_table_cached[i, min_p:max_p] = torch.arange(
                local_block_size + transfer_offsets[i],
                local_block_size + transfer_offsets[i] + t_len,
                dtype=torch.int32, device=device,
            )
        if e_len > 0:
            page_table_cached[i, max_p:seq_len] = torch.arange(
                extend_block_start + extend_hidden_starts[i],
                extend_block_start + extend_hidden_starts[i] + e_len,
                dtype=torch.int32, device=device,
            )

    cache_seqlens_tensor = torch.tensor(
        seq_lens_cpu, dtype=torch.int32, device=device
    )
    cu_seqlens_k_new_tensor = F.pad(
        torch.cumsum(cache_seqlens_tensor, dim=0, dtype=torch.int32), (1, 0)
    )

    logger.debug(
        "[share_prefix] dp_rank=%d precomputed page_table=%s "
        "local_block_size=%d extend_block_start=%d",
        local_dp_rank,
        tuple(page_table_cached.shape),
        local_block_size,
        extend_block_start,
    )

    # ------------------------------------------------------------------ #
    # 7. Return metadata (combined_kv_buf allocated lazily later)          #
    # ------------------------------------------------------------------ #
    info = SharePrefixBatchInfo(
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
        local_block_size=local_block_size,
        extend_block_start=extend_block_start,
        local_block_starts=local_block_starts,
        extend_hidden_starts=extend_hidden_starts,
        prefix_indices_list=prefix_indices_list,
        dp_local_req_global_indices=dp_local_req_global_indices,
        all_local_src_indices=all_local_src_indices,
        page_table_cached=page_table_cached,
        cache_seqlens_tensor=cache_seqlens_tensor,
        cu_seqlens_k_new_tensor=cu_seqlens_k_new_tensor,
        combined_kv_buf=None,  # allocated lazily by ensure_combined_buf_allocated()
        device=device,
    )

    logger.debug(
        "[share_prefix] dp_rank=%d share_prefix_info computed: "
        "total_transfer_tokens=%d batch_size=%d",
        local_dp_rank, total_transfer, n,
    )
    return info


# ---------------------------------------------------------------------------
# Helper: fill_transfer_region_for_layer
# ---------------------------------------------------------------------------

def fill_transfer_region_for_layer(
    info: SharePrefixBatchInfo,
    layer_id: int,
    kv_buf: torch.Tensor,  # token_to_kv_pool.get_key_buffer(layer_id)
) -> None:
    """
    Fill the transfer_block of combined_kv_buf with KV[min_prefix : max_prefix]
    from the local KV cache for requests where this rank is max_rank.

    The transfer_block occupies combined_kv_buf[local_block_size : extend_block_start].
    Within it, request i's transfer region starts at transfer_offsets[i].

    Only fills when:  info.local_dp_rank == max_rank[i]
                  AND info.local_attn_tp_rank == 0

    All other positions in transfer_block remain zero (for all_reduce correctness).

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
    kv_dtype = info.combined_kv_buf.dtype
    base = info.local_block_size  # offset of transfer_block in combined_kv_buf

    filled_reqs = []
    for i in range(n):
        if info.max_rank[i] != info.local_dp_rank:
            continue
        t_len = info.transfer_len[i]
        if t_len == 0:
            continue

        min_p = info.min_prefix[i]
        max_p = info.max_prefix[i]
        t_start = info.transfer_offsets[i]

        # pref_idx[min_p:max_p] are the KV pool slot indices for the transfer region
        pref_idx = info.prefix_indices_list[i]  # length = local_prefix[i] >= max_p
        slots = pref_idx[min_p:max_p]  # [t_len] int64

        kv_slice = kv_buf[slots]  # [t_len, 1, kv_cache_dim]
        if kv_slice.dtype != kv_dtype:
            kv_slice = kv_slice.to(kv_dtype)

        info.combined_kv_buf[base + t_start : base + t_start + t_len] = kv_slice
        filled_reqs.append((i, min_p, max_p, t_len))

    if filled_reqs:
        logger.debug(
            "[share_prefix] layer=%d dp_rank=%d filled transfer_buffer for reqs: %s",
            layer_id, info.local_dp_rank,
            [f"req{i}: [{mn},{mx}) len={tl}" for i, mn, mx, tl in filled_reqs],
        )


# ---------------------------------------------------------------------------
# Helper: fill_local_and_extend_for_layer
# ---------------------------------------------------------------------------

def fill_local_and_extend_for_layer(
    info: SharePrefixBatchInfo,
    k_nope: torch.Tensor,   # [total_extend_tokens, 1, kv_lora_rank]
    k_pe: torch.Tensor,     # [total_extend_tokens, 1, qk_rope_head_dim]
    kv_buf: torch.Tensor,   # token_to_kv_pool.get_key_buffer(layer_id)
) -> None:
    """
    Fill the local_block and extend_block of combined_kv_buf for the current layer.

    local_block  [0 .. local_block_size):
        One batched indexed gather from kv_buf using pre-computed all_local_src_indices.
        This replaces the per-request loop in the old build_combined_kv_for_layer.

    extend_block [extend_block_start .. total_combined):
        Two contiguous writes – one for the k_nope component, one for k_pe.
        extend_hidden_starts aligns k_nope/k_pe (in max_prefix-based order) with
        the extend_block positions.

    The transfer_block [local_block_size .. extend_block_start) is handled
    separately by reset_transfer_region + fill_transfer_region_for_layer +
    all_reduce, so this function does NOT touch it.
    """
    dtype = k_nope.dtype

    # ---- local_block ----
    if info.local_block_size > 0:
        info.combined_kv_buf[: info.local_block_size] = (
            kv_buf[info.all_local_src_indices].to(dtype)
        )

    # ---- extend_block ----
    total_extend = k_nope.shape[0]
    if total_extend > 0:
        buf_ext = info.combined_kv_buf[info.extend_block_start :]
        buf_ext[:, 0, : info.kv_lora_rank] = k_nope[:, 0]
        buf_ext[:, 0, info.kv_lora_rank :] = k_pe[:, 0]


# ---------------------------------------------------------------------------
# Convenience: zero out transfer_region before filling (to be safe)
# ---------------------------------------------------------------------------

def reset_transfer_region(info: SharePrefixBatchInfo) -> None:
    """Zero the transfer_block in combined_kv_buf before each layer's fill + all_reduce."""
    info.combined_kv_buf[info.local_block_size : info.extend_block_start].zero_()


# ---------------------------------------------------------------------------
# Helper: save_dp_local_kv
# ---------------------------------------------------------------------------

def save_dp_local_kv(
    info: SharePrefixBatchInfo,
    layer,                        # RadixAttention – used for layer_id lookup
    out_cache_loc: torch.Tensor,  # [total_dp_local_extend_tokens] – flat slot indices
    token_to_kv_pool,             # MLATokenToKVPool (or wrapper)
) -> None:
    """
    Write KV cache for all dp-local requests after share-prefix attention.

    For each dp-local request (decode_dp_rank == this rank), we write two
    segments into the pre-allocated out_cache_loc slots.  Both segments are
    read directly from combined_kv_buf (no need for separate k_nope/k_pe refs).

    Segment 1 – Transfer region [local_prefix, max_prefix):
        combined_kv_buf[local_block_size + transfer_offsets[g] + (local_p - min_p) :
                        local_block_size + transfer_offsets[g] + (max_p  - min_p)]

    Segment 2 – Extend region [max_prefix, seq_len):
        combined_kv_buf[extend_block_start + extend_hidden_starts[g] :
                        extend_block_start + extend_hidden_starts[g] + true_ext_len]

    Parameters
    ----------
    info           : SharePrefixBatchInfo – must have combined_kv_buf allocated.
    layer          : RadixAttention object for layer_id (same as used in flash_attn).
    out_cache_loc  : flat token pool slot indices for dp-local requests, in dp-local order.
    token_to_kv_pool : KV pool; must expose set_mla_kv_buffer(layer, loc, k_nope, k_rope).
    """
    oc_offset = 0
    total_transfer_written = 0
    total_extend_written = 0

    kv_lora_rank = info.kv_lora_rank
    base_xfer = info.local_block_size
    base_ext = info.extend_block_start

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
            # transfer_block contains positions [min_p, max_p) for this request.
            # This rank's [local_p, max_p) starts at offset (local_p - min_p).
            t_src_start = base_xfer + info.transfer_offsets[g] + (local_p - min_p)
            t_src_end   = base_xfer + info.transfer_offsets[g] + (max_p - min_p)

            transfer_kv = info.combined_kv_buf[t_src_start : t_src_end]  # [tl, 1, kv_cache_dim]
            transfer_slots = oc_slice[:transfer_local_len]
            xfer_k_nope = transfer_kv[:, :, :kv_lora_rank]
            xfer_k_rope = transfer_kv[:, :, kv_lora_rank:]
            token_to_kv_pool.set_mla_kv_buffer(layer, transfer_slots, xfer_k_nope, xfer_k_rope)
            total_transfer_written += transfer_local_len

        # ---- Segment 2: extend region [max_p, seq_len) ----
        true_ext_len = seq_len - max_p          # 0 if all tokens were cached/transferred
        if true_ext_len > 0:
            h_start = info.extend_hidden_starts[g]
            e_src_start = base_ext + h_start
            e_src_end   = base_ext + h_start + true_ext_len

            ext_kv = info.combined_kv_buf[e_src_start : e_src_end]   # [el, 1, kv_cache_dim]
            extend_slots = oc_slice[transfer_local_len:]
            ext_k_nope = ext_kv[:, :, :kv_lora_rank]
            ext_k_rope = ext_kv[:, :, kv_lora_rank:]
            token_to_kv_pool.set_mla_kv_buffer(layer, extend_slots, ext_k_nope, ext_k_rope)
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
