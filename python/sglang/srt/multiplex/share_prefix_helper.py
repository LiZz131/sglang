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

Persistent buffer (grow-only)
------------------------------
combined_kv_buf is now backed by a module-level grow-only tensor
(_PERSISTENT_COMBINED_KV_BUF).  On each new batch, ensure_combined_buf_allocated()
returns a view (no CUDA malloc) if capacity ≥ total_combined.  Only when a batch
requires more tokens than ever seen before does the buffer grow (2× doubling
strategy).  After the first forward pass, all subsequent batches of equal or
smaller token count incur zero cudaMalloc overhead at layer 0.

Why max_prefill_tokens is not the right static bound:
  total_combined = sum(seq_lens) = sum(max_prefix) + sum(extend_lens)
  Only sum(extend_lens) ≤ max_prefill_tokens.
  sum(max_prefix) ≤ KV pool capacity, which is >> max_prefill_tokens.

Communication groups
--------------------
* All-gather  : tp_group (spans all DP groups within one worker).
* All-reduce  : tp_group (same).
* Only attn_tp_rank == 0 fills transfer_region to avoid N-fold duplication when
  attn_tp_size > 1 (all ranks within one DP group share the same latent KV cache).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistent (grow-only) combined_kv_buf
# ---------------------------------------------------------------------------
# combined_kv_buf is the main per-batch working buffer used in
# _share_prefix_attn_mqa.  Its size is sum(seq_lens) × kv_cache_dim, which
# varies per batch.  Allocating it fresh each batch (even once per forward)
# causes a cudaMalloc visible in nsys at layer 0.
#
# Solution: keep a module-level dict of grow-only tensors, one per
# (device, dtype, kv_cache_dim) key.  On each batch, acquire() returns a
# *view* (no new allocation) if the capacity is sufficient, or reallocates
# at 2× the required size.  After warm-up, no further cudaMalloc occurs.
#
# Why is max_prefill_tokens insufficient as a static upper bound?
#   combined_kv_buf size = sum(seq_lens) = sum(max_prefix) + sum(extend_lens)
#   Only sum(extend_lens) ≤ max_prefill_tokens.
#   sum(max_prefix) is bounded by the KV pool size, which can be >> max_prefill_tokens.
#
# Key: (device_str, dtype_str, kv_cache_dim: int)
_PERSISTENT_COMBINED_KV_BUF: dict[tuple, torch.Tensor] = {}

# Minimum initial capacity (tokens) to avoid growth on tiny warm-up batches.
_PERSISTENT_BUF_MIN_CAPACITY = 44156


def _acquire_combined_kv_buf(
    total_combined: int,
    kv_cache_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a [:total_combined, 1, kv_cache_dim] view of the persistent buffer.

    Grows the persistent buffer (doubling strategy) only when
    total_combined > current capacity.  The returned tensor is a storage-alias
    view; it shares memory with the persistent buffer and will be overwritten
    on the next batch that calls this function.  Callers must NOT hold the
    view past the end of the current forward pass.

    Parameters
    ----------
    total_combined : int
        sum(seq_lens) for the current batch.
    kv_cache_dim   : int
        kv_lora_rank + qk_rope_head_dim.
    dtype          : torch.dtype
        Model activation dtype (fp16 / bf16).
    device         : torch.device
        CUDA device the model runs on.

    Returns
    -------
    torch.Tensor  shape [total_combined, 1, kv_cache_dim], dtype=dtype, device=device
    """
    key = (str(device), str(dtype), kv_cache_dim)
    buf = _PERSISTENT_COMBINED_KV_BUF.get(key)

    if buf is None or buf.shape[0] < total_combined:
        old_capacity = buf.shape[0] if buf is not None else 0
        new_capacity = max(
            total_combined * 2,            # 2× headroom for future batches
            old_capacity * 2,              # double existing if non-zero
            _PERSISTENT_BUF_MIN_CAPACITY,  # minimum floor
        )
        buf = torch.empty(
            (new_capacity, 1, kv_cache_dim),
            dtype=dtype,
            device=device,
        )
        _PERSISTENT_COMBINED_KV_BUF[key] = buf
        logger.info(
            "[share_prefix] persistent combined_kv_buf GROW: "
            "old_capacity=%d new_capacity=%d total_combined=%d "
            "dtype=%s device=%s kv_cache_dim=%d "
            "alloc_MB=%.2f",
            old_capacity, new_capacity, total_combined,
            dtype, device, kv_cache_dim,
            new_capacity * kv_cache_dim * buf.element_size() / (1024 * 1024),
        )
    else:
        logger.debug(
            "[share_prefix] persistent combined_kv_buf REUSE: "
            "capacity=%d total_combined=%d (headroom=%d)",
            buf.shape[0], total_combined, buf.shape[0] - total_combined,
        )

    return buf[:total_combined]


def get_persistent_combined_kv_buf_capacity(
    kv_cache_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> int:
    """Return the current capacity (number of token slots) of the persistent buffer.

    Returns 0 if no buffer has been allocated yet for this (device, dtype, kv_cache_dim).
    Intended for logging, monitoring, and unit tests.
    """
    key = (str(device), str(dtype), kv_cache_dim)
    buf = _PERSISTENT_COMBINED_KV_BUF.get(key)
    return buf.shape[0] if buf is not None else 0


def clear_persistent_combined_kv_buf() -> None:
    """Release all persistent combined_kv_buf allocations.

    Intended for unit tests (call in setup_method) and server shutdown.
    Not safe to call while a forward pass is in-flight.
    """
    _PERSISTENT_COMBINED_KV_BUF.clear()
    logger.debug("[share_prefix] persistent combined_kv_buf cleared")


# ---------------------------------------------------------------------------
# NVTX profiling (optional, gated by ServerArgs)
# ---------------------------------------------------------------------------

@dataclass
class SharePrefixNvtxStats:
    """Byte/token counts for NVTX range labels and batch_stats logging."""

    total_transfer_tokens: int
    transfer_bytes: int
    local_block_tokens: int
    local_bytes: int
    extend_tokens: int
    extend_bytes: int
    combined_bytes: int
    batch_size: int
    local_dp_rank: int
    kv_cache_dim: int
    elem_size: int


def _share_prefix_nvtx_config() -> Tuple[bool, int]:
    try:
        from sglang.srt.server_args import get_global_server_args

        sa = get_global_server_args()
        enabled = bool(getattr(sa, "enable_share_prefix_nvtx", False))
        stride = int(getattr(sa, "share_prefix_nvtx_layer_stride", 1))
        return enabled, max(1, stride)
    except Exception:
        return False, 1


def compute_share_prefix_nvtx_stats(
    info: "SharePrefixBatchInfo",
    elem_size: int,
) -> SharePrefixNvtxStats:
    kv_cache_dim = info.kv_cache_dim
    if kv_cache_dim <= 0:
        kv_cache_dim = info.kv_lora_rank  # fallback before buf alloc
    extend_tokens = sum(
        info.seq_lens[i] - info.max_prefix[i] for i in range(len(info.seq_lens))
    )
    transfer_bytes = info.total_transfer_tokens * kv_cache_dim * elem_size
    local_bytes = info.local_block_size * kv_cache_dim * elem_size
    extend_bytes = extend_tokens * kv_cache_dim * elem_size
    combined_bytes = sum(info.seq_lens) * kv_cache_dim * elem_size
    return SharePrefixNvtxStats(
        total_transfer_tokens=info.total_transfer_tokens,
        transfer_bytes=transfer_bytes,
        local_block_tokens=info.local_block_size,
        local_bytes=local_bytes,
        extend_tokens=extend_tokens,
        extend_bytes=extend_bytes,
        combined_bytes=combined_bytes,
        batch_size=len(info.seq_lens),
        local_dp_rank=info.local_dp_rank,
        kv_cache_dim=kv_cache_dim,
        elem_size=elem_size,
    )


def format_share_prefix_nvtx_label(
    stats: SharePrefixNvtxStats,
    layer_id: int,
    phase: str,
) -> str:
    if layer_id < 0:
        prefix = f"share_prefix/batch/{phase}"
    else:
        prefix = f"share_prefix/L{layer_id}/{phase}"
    return (
        f"{prefix} xfer_toks={stats.total_transfer_tokens} xfer_B={stats.transfer_bytes} "
        f"local_toks={stats.local_block_tokens} local_B={stats.local_bytes} "
        f"ext_toks={stats.extend_tokens} ext_B={stats.extend_bytes} "
        f"combined_B={stats.combined_bytes} bs={stats.batch_size} dp={stats.local_dp_rank}"
    )


@contextmanager
def share_prefix_nvtx_range(
    info: Optional["SharePrefixBatchInfo"],
    layer_id: int,
    phase: str,
    *,
    elem_size: int = 4,
) -> Iterator[None]:
    """NVTX range for share-prefix; batch ranges use layer_id=-1 (no stride filter)."""
    enabled, stride = _share_prefix_nvtx_config()
    if not enabled:
        yield
        return
    if layer_id >= 0 and layer_id % stride != 0:
        yield
        return
    if info is None:
        label = f"share_prefix/batch/{phase}" if layer_id < 0 else f"share_prefix/L{layer_id}/{phase}"
        handle = nvtx.range_start(label)
    else:
        stats = compute_share_prefix_nvtx_stats(info, elem_size)
        handle = nvtx.range_start(format_share_prefix_nvtx_label(stats, layer_id, phase))
    try:
        yield
    finally:
        nvtx.range_end(handle)


def log_share_prefix_batch_stats(
    info: "SharePrefixBatchInfo",
    elem_size: int = 2,
) -> None:
    """One-line INFO summary per batch (rank0 only)."""
    if info.local_dp_rank != 0 or info.local_attn_tp_rank != 0:
        return
    _, stride = _share_prefix_nvtx_config()
    stats = compute_share_prefix_nvtx_stats(info, elem_size)
    if stats.kv_cache_dim > 0:
        logger.info(
            "[share_prefix] batch_stats: total_transfer=%d transfer_B=%d "
            "local_toks=%d local_B=%d ext_toks=%d ext_B=%d combined_B=%d "
            "bs=%d dp=%d kv_dim=%d nvtx_stride_K=%d",
            stats.total_transfer_tokens,
            stats.transfer_bytes,
            stats.local_block_tokens,
            stats.local_bytes,
            stats.extend_tokens,
            stats.extend_bytes,
            stats.combined_bytes,
            stats.batch_size,
            stats.local_dp_rank,
            stats.kv_cache_dim,
            stride,
        )
    else:
        logger.info(
            "[share_prefix] batch_stats: total_transfer=%d local_toks=%d "
            "ext_toks=%d combined_toks=%d bs=%d dp=%d nvtx_stride_K=%d "
            "(bytes logged after combined_kv_buf alloc)",
            stats.total_transfer_tokens,
            stats.local_block_tokens,
            stats.extend_tokens,
            sum(info.seq_lens),
            stats.batch_size,
            stats.local_dp_rank,
            stride,
        )


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

    # ---------- per-batch view into the module-level persistent buffer ----------
    # combined_kv_buf: [total_combined, 1, kv_cache_dim]
    # Set to a view of _PERSISTENT_COMBINED_KV_BUF by ensure_combined_buf_allocated().
    # None until that call.
    combined_kv_buf: Optional[torch.Tensor]

    # ---------- misc ----------
    kv_cache_dim: int = 0           # kv_lora_rank + qk_rope_head_dim
    kv_lora_rank: int = 0           # for splitting combined KV into rope / nope
    device: Optional[torch.device] = None

    # ---------- pipeline overlap state (optional) ----------
    # Set comm_stream to enable computation-communication overlap.  When set,
    # _share_prefix_attn_mqa uses the pipelined path:
    #   Phase A (comm_stream): save_kv(layer i) + reset/fill_xfer/all_reduce/fill_local(layer i+1)
    #   Phase B (main_stream): fill_extend(layer i+1) + flash_attn(layer i+1)
    # Both streams are coordinated via attn_done_event (main→comm) and
    # _pending_ltr_event (comm→main).
    #
    # comm_stream is per-batch but can be a long-lived object reused across batches.
    # Use set_pipeline_comm_stream() to inject or replace the stream.
    comm_stream: Optional[torch.cuda.Stream] = None

    # The PP-rank-local last layer id (self.end_layer - 1).  Used to decide when
    # to drain comm_stream and when to skip Phase A (PP boundary).
    # -1 means not yet initialised.
    pipeline_last_layer_id: int = -1

    # Carries the local-transfer-ready CUDA Event from one layer's comm_stream
    # work to the next layer's main_stream wait.
    # - Reset to None at the start of the FIRST sub-forward of each batch.
    # - Preserved across sub-forwards so comm_stream work started at the tail of
    #   sub-forward N is picked up by sub-forward N+1 without double-launch.
    _pending_ltr_event: Optional[torch.cuda.Event] = None

    # True after the first call to init_pipeline_state_for_forward() for this
    # batch.  Prevents _pending_ltr_event from being cleared on subsequent
    # sub-forward calls in split-prefill.
    pipeline_initialized: bool = False

    def set_pipeline_comm_stream(
        self, stream: Optional[torch.cuda.Stream]
    ) -> None:
        """Inject or replace the comm_stream used for pipeline overlap.

        Pass None to disable pipeline overlap and fall back to synchronous mode.
        The stream is not owned by this object; callers are responsible for its
        lifetime.
        """
        self.comm_stream = stream

    def init_pipeline_state_for_forward(
        self,
        pp_last_layer_id: int,
        *,
        force_reset: bool = False,
    ) -> None:
        """Initialise pipeline state before a forward (or sub-forward) pass.

        Must be called once per forward / split-prefill chunk, BEFORE the layer
        loop.  For split-prefill the same ``SharePrefixBatchInfo`` is reused
        across sub-forwards; this method therefore distinguishes "first call"
        from "subsequent calls" via ``pipeline_initialized``:

        * First call  → reset ``_pending_ltr_event`` and record
          ``pipeline_last_layer_id`` (the PP-rank's true last layer, fixed for
          the entire batch).
        * Later calls → only update ``pipeline_last_layer_id`` if it hasn't been
          set yet (shouldn't happen), but do NOT touch ``_pending_ltr_event`` so
          comm_stream work started during the previous sub-forward is inherited.

        Parameters
        ----------
        pp_last_layer_id:
            ``self.end_layer - 1`` for the current PP rank.  This is the layer
            at which comm_stream is drained and Phase A is no longer launched.
        force_reset:
            If True, always reset as if this were the first call.  Used by the
            regular (non-split) ``forward()`` path which always starts fresh.
        """
        if force_reset or not self.pipeline_initialized:
            self._pending_ltr_event = None
            self.pipeline_last_layer_id = pp_last_layer_id
            self.pipeline_initialized = True
            logger.debug(
                "[share_prefix_pipeline] pipeline init: pp_last_layer=%d "
                "force_reset=%s",
                pp_last_layer_id,
                force_reset,
            )

    def ensure_combined_buf_allocated(
        self,
        kv_cache_dim: int,
        kv_lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Wire combined_kv_buf to the global grow-only persistent buffer.

        On the first call for this batch, this returns a *view* (no new CUDA
        malloc) into _PERSISTENT_COMBINED_KV_BUF when the existing capacity is
        sufficient.  The persistent buffer is reallocated (grow-only, 2×) only
        when total_combined exceeds the current capacity.

        After the model warm-up batch, subsequent prefill batches should never
        trigger a CUDA malloc here.

        Only combined_kv_buf requires kv_cache_dim and is deferred to the first
        model layer.  All other layout tensors (page_table_cached, etc.) are
        already allocated in compute_share_prefix_info.
        """
        if self.combined_kv_buf is not None:
            return
        self.kv_cache_dim = kv_cache_dim
        self.kv_lora_rank = kv_lora_rank
        self.device = device
        total_combined = sum(self.seq_lens)
        self.combined_kv_buf = _acquire_combined_kv_buf(
            total_combined, kv_cache_dim, dtype, device
        )


# ---------------------------------------------------------------------------
# SchedulingGatherResult: result of the scheduling-level all-gather
# ---------------------------------------------------------------------------

@dataclass
class SchedulingGatherResult:
    """Result of gather_scheduling_prefix_data() called during scheduling.

    Contains per-request max_prefix_list (max across all DP ranks) and the
    global minimum KV memory budget, used to make consistent prefill scheduling
    decisions across all DP ranks.
    """
    max_prefix_list: List[int]      # max local_prefix_len across DP ranks, per req
    global_min_raw_budget: int      # min(available + evictable) across all DP ranks


def gather_scheduling_prefix_data(
    waiting_queue,              # List[Req] — the scheduler waiting queue
    tree_cache,                 # BasePrefixCache — for match_prefix
    tp_cpu_group,               # Gloo CPU group spanning all DP ranks × attn_tp ranks
    attn_tp_size: int,          # number of attn_tp ranks per DP group
    dp_size: int,               # number of DP groups
    token_to_kv_pool_allocator, # BaseTokenToKVPoolAllocator — for budget query
) -> "SchedulingGatherResult":
    """All-gather local prefix lengths and KV budget across DP ranks for scheduling.

    This function is called once per scheduling round (inside
    _get_new_batch_prefill_raw) BEFORE calc_priority and PrefillAdder run.
    It ensures every DP rank uses the same per-request max_prefix and the same
    KV memory budget, preventing the prefill batch sizes from diverging.

    Side-effects
    ------------
    - Calls tree_cache.match_prefix() for every request in waiting_queue,
      populating req.prefix_indices / req.last_node / req.last_host_node /
      req.host_hit_length (identical side-effect to SchedulePolicy._compute_prefix_matches).
    - Sets req.sched_max_prefix (int) on every request.

    Communication
    -------------
    One dist.all_gather on the CPU group carrying:
        [prefix_len_0, …, prefix_len_{n-1}, raw_budget]
    Total payload: (n+1) × 8 bytes × tp_size.  Very cheap.

    Parameters
    ----------
    waiting_queue           : scheduler waiting_queue (list of Req).
    tree_cache              : local radix-tree cache.
    tp_cpu_group            : Gloo CPU group spanning all TP ranks (scheduler's
                              tp_cpu_group).  Do NOT use attn_tp_cpu_group here:
                              with dp_attention, attn_tp_size may be 1 per process.
    attn_tp_size            : TP ranks per DP group (to select representative rank).
    dp_size                 : number of DP groups.
    token_to_kv_pool_allocator : allocator whose available_size() gives budget.

    Returns
    -------
    SchedulingGatherResult with max_prefix_list and global_min_raw_budget.
    """
    from sglang.srt.mem_cache.radix_cache import RadixKey

    n = len(waiting_queue)
    world_size = dist.get_world_size(group=tp_cpu_group)

    # ------------------------------------------------------------------ #
    # Step 1: compute local prefix match lengths (same side-effect as     #
    # SchedulePolicy._compute_prefix_matches).                            #
    # ------------------------------------------------------------------ #
    local_prefix_lens: List[int] = []
    for req in waiting_queue:
        prefix_ids = req.origin_input_ids + req.output_ids
        extra_key = req.extra_key
        match_result = tree_cache.match_prefix(
            rid=req.rid,
            key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
        )
        req.prefix_indices = match_result.device_indices
        req.last_node = match_result.last_device_node
        req.last_host_node = match_result.last_host_node
        req.host_hit_length = match_result.host_hit_length
        local_prefix_lens.append(len(req.prefix_indices))

    # ------------------------------------------------------------------ #
    # Step 2: query local KV budget (must match PrefillAdder.rem_total_tokens
    # before rem_total_token_offset: available + evictable).                #
    # ------------------------------------------------------------------ #
    try:
        local_budget = int(
            token_to_kv_pool_allocator.available_size()
            + tree_cache.evictable_size()
        )
    except Exception:
        local_budget = 0

    # ------------------------------------------------------------------ #
    # Step 3: all_gather  [prefix_len_0, …, prefix_len_{n-1}, budget]    #
    #         via CPU (Gloo) group                                        #
    # ------------------------------------------------------------------ #
    local_data = torch.tensor(
        local_prefix_lens + [local_budget], dtype=torch.int64
    )
    gathered = [torch.empty_like(local_data) for _ in range(world_size)]
    dist.all_gather(gathered, local_data, group=tp_cpu_group)

    # ------------------------------------------------------------------ #
    # Step 4: compute per-request max_prefix across DP ranks              #
    # Use only attn_tp_rank==0 representative per DP group                #
    # (same convention as gather_prefix_data).                            #
    # ------------------------------------------------------------------ #
    dp_prefix_matrix: List[List[int]] = []
    dp_budgets: List[int] = []
    for dp_r in range(dp_size):
        rep_idx = dp_r * attn_tp_size + 0  # attn_tp_rank == 0 representative
        row = gathered[rep_idx].tolist()
        dp_prefix_matrix.append(row[:n])   # prefix lens
        dp_budgets.append(int(row[n]))     # budget

    max_prefix_list: List[int] = []
    for i in range(n):
        max_p = max(dp_prefix_matrix[dp_r][i] for dp_r in range(dp_size))
        max_prefix_list.append(max_p)

    # Set convenience attribute on each request so PrefillAdder can read it
    for req, mp in zip(waiting_queue, max_prefix_list):
        req.sched_max_prefix = mp

    global_min_raw_budget = min(dp_budgets) if dp_budgets else local_budget

    logger.debug(
        "[sched_gather] n_reqs=%d world_size=%d dp_size=%d attn_tp_size=%d "
        "local_prefix_lens=%s max_prefix_list=%s "
        "local_budget=%d global_min_raw_budget=%d",
        n, world_size, dp_size, attn_tp_size,
        local_prefix_lens, max_prefix_list,
        local_budget, global_min_raw_budget,
    )
    for i, req in enumerate(waiting_queue):
        logger.debug(
            "[sched_gather] req[%d] rid=%s local_prefix=%d max_prefix=%d",
            i, req.rid, local_prefix_lens[i], max_prefix_list[i],
        )

    return SchedulingGatherResult(
        max_prefix_list=max_prefix_list,
        global_min_raw_budget=global_min_raw_budget,
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
    with share_prefix_nvtx_range(None, -1, "gather_prefix"):
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
            if isinstance(req.prefix_indices, torch.Tensor):
                pi = req.prefix_indices.to(dtype=torch.int64, device=device)
            else:
                pi = torch.tensor(req.prefix_indices, dtype=torch.int64, device=device)
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

    with share_prefix_nvtx_range(None, -1, "compute_info"):
        # all_local_src_indices: concatenation of pref_idx[:min_p] for each req
        if local_block_size > 0:
            all_local_src_indices: torch.Tensor = torch.cat([
                prefix_indices_list[i][:min_prefix_list[i]]
                for i in range(n)
                if min_prefix_list[i] > 0
            ])
        else:
            all_local_src_indices = torch.empty(0, dtype=torch.int64, device=device)

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
    log_share_prefix_batch_stats(info, elem_size=2)

    # Buffer layout summary (token counts) for verifying prefix distribution.
    # kv_cache_dim is not known yet (allocated lazily), so we report token counts.
    total_combined_toks = sum(seq_lens_cpu)
    logger.info(
        "[share_prefix] buffer layout: dp_rank=%d bs=%d local_toks=%d "
        "xfer_toks=%d ext_toks=%d combined_toks=%d | "
        "per_req min_prefix=%s max_prefix=%s transfer_len=%s",
        local_dp_rank, n,
        local_block_size, total_transfer,
        total_combined_toks - extend_block_start, total_combined_toks,
        min_prefix_list, max_prefix_list, transfer_len_list,
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

def fill_local_block_for_layer(
    info: "SharePrefixBatchInfo",
    kv_buf: torch.Tensor,  # token_to_kv_pool.get_key_buffer(layer_id)
) -> None:
    """
    Fill combined_kv_buf[:local_block_size] via a single batched indexed gather.

    This region contains the min-prefix KV tokens that every DP rank already has
    locally.  The fill is independent of the current layer's hidden-state output,
    so it can safely run on a separate comm_stream in the pipeline overlap scheme.

    The transfer_block and extend_block are NOT touched by this function.
    """
    if info.local_block_size > 0:
        info.combined_kv_buf[: info.local_block_size] = kv_buf[
            info.all_local_src_indices
        ].to(info.combined_kv_buf.dtype)


def fill_extend_block_for_layer(
    info: "SharePrefixBatchInfo",
    k_nope: torch.Tensor,  # [total_extend_tokens, 1, kv_lora_rank]
    k_pe: torch.Tensor,    # [total_extend_tokens, 1, qk_rope_head_dim]
) -> None:
    """
    Fill combined_kv_buf[extend_block_start:] from k_nope / k_pe.

    k_nope and k_pe come from the current layer's qkv projection and must
    therefore run on main_stream after the qkv_proj kernel completes.  They
    write to the extend_block region, which is disjoint from the local_block
    and transfer_block regions written by fill_local_block_for_layer and
    fill_transfer_region_for_layer respectively.
    """
    total_extend = k_nope.shape[0]
    if total_extend > 0:
        buf_ext = info.combined_kv_buf[info.extend_block_start :]
        buf_ext[:, 0, : info.kv_lora_rank] = k_nope[:, 0]
        buf_ext[:, 0, info.kv_lora_rank :] = k_pe[:, 0]


def fill_local_and_extend_for_layer(
    info: "SharePrefixBatchInfo",
    k_nope: torch.Tensor,   # [total_extend_tokens, 1, kv_lora_rank]
    k_pe: torch.Tensor,     # [total_extend_tokens, 1, qk_rope_head_dim]
    kv_buf: torch.Tensor,   # token_to_kv_pool.get_key_buffer(layer_id)
) -> None:
    """
    Fill the local_block and extend_block of combined_kv_buf for the current layer.

    Compatibility wrapper that calls fill_local_block_for_layer and
    fill_extend_block_for_layer sequentially on the current stream.  Used by
    the legacy synchronous path in _share_prefix_attn_mqa.

    local_block  [0 .. local_block_size):
        One batched indexed gather from kv_buf using pre-computed all_local_src_indices.

    extend_block [extend_block_start .. total_combined):
        Two contiguous writes – one for the k_nope component, one for k_pe.

    The transfer_block [local_block_size .. extend_block_start) is handled
    separately by reset_transfer_region + fill_transfer_region_for_layer +
    all_reduce, so this function does NOT touch it.
    """
    fill_local_block_for_layer(info, kv_buf)
    fill_extend_block_for_layer(info, k_nope, k_pe)


# ---------------------------------------------------------------------------
# Convenience: zero out transfer_region before filling (to be safe)
# ---------------------------------------------------------------------------

def reset_transfer_region(info: SharePrefixBatchInfo) -> None:
    """Zero the transfer_block in combined_kv_buf before each layer's fill + all_reduce."""
    info.combined_kv_buf[info.local_block_size : info.extend_block_start].zero_()


# ---------------------------------------------------------------------------
# Pipeline overlap helpers: Phase A and Phase C
# ---------------------------------------------------------------------------
#
# Helper: KV pool layer membership check
# ---------------------------------------------------------------------------


def _kv_pool_has_layer(token_to_kv_pool, layer_id: int) -> bool:
    """Return True if token_to_kv_pool contains KV data for layer_id.

    In pipeline-parallel setups each PP rank owns a contiguous sub-range of
    layers [start_layer, start_layer + len(kv_buffer)).  Phase A must NOT call
    get_key_buffer() for layers outside this range.
    """
    start = token_to_kv_pool.start_layer
    return start <= layer_id < start + len(token_to_kv_pool.kv_buffer)


#
# Phase A (runs on comm_stream):
#   reset_transfer_region + fill_transfer_region_for_layer + all_reduce + fill_local_block
#   Independent of the current layer's hidden-state; can overlap with MLP.
#
# Phase C (runs on comm_stream after main_stream signals attn_done):
#   save_dp_local_kv for the just-finished layer, then Phase A for the next layer.
#
# Event flow per layer:
#   main_stream records attn_done_event  ──►  comm_stream waits, runs Phase C + Phase A(next)
#   comm_stream records ltr_event        ──►  main_stream waits, runs fill_extend + flash_attn


def _launch_phase_a(
    info: SharePrefixBatchInfo,
    layer_id: int,
    kv_buf: torch.Tensor,
    tp_group,
    comm_stream: torch.cuda.Stream,
) -> torch.cuda.Event:
    """
    Launch Phase A on comm_stream for the given layer.

    Phase A fills the local_block and (when total_transfer_tokens > 0) also
    resets / fills / all_reduces the transfer_block.  Both writes are
    independent of the current batch's hidden-state computation, so they can
    safely run on comm_stream while main_stream executes MLP or qkv_proj.

    Returns a CUDA Event recorded at the end of Phase A on comm_stream.
    The caller (or the next layer's _share_prefix_attn_mqa) must call
        torch.cuda.current_stream().wait_event(returned_event)
    before issuing fill_extend_block_for_layer / flash_attn.

    Parameters
    ----------
    info        : SharePrefixBatchInfo with combined_kv_buf already allocated.
    layer_id    : transformer layer index (used only for logging).
    kv_buf      : token_to_kv_pool.get_key_buffer(layer_id).
    tp_group    : NCCL group for all_reduce.
    comm_stream : CUDA stream on which to schedule all kernels.
    """
    ltr_event = torch.cuda.Event()
    with torch.cuda.stream(comm_stream):
        if info.total_transfer_tokens > 0:
            reset_transfer_region(info)
            fill_transfer_region_for_layer(info, layer_id, kv_buf)
            dist.all_reduce(
                info.combined_kv_buf[info.local_block_size : info.extend_block_start],
                op=dist.ReduceOp.SUM,
                group=tp_group.device_group,
            )
        fill_local_block_for_layer(info, kv_buf)
        ltr_event.record(comm_stream)

    logger.debug(
        "[share_prefix_pipeline] Phase A launched on comm_stream: "
        "layer=%d xfer_toks=%d local_toks=%d combined_toks=%d stream_id=%d",
        layer_id,
        info.total_transfer_tokens,
        info.local_block_size,
        sum(info.seq_lens),
        comm_stream.cuda_stream,
    )
    return ltr_event


def _launch_phase_c_and_maybe_next_phase_a(
    info: SharePrefixBatchInfo,
    layer_id: int,
    layer,                        # RadixAttention – used for layer_id lookup in save_dp_local_kv
    out_cache_loc: torch.Tensor,
    token_to_kv_pool,
    tp_group,
    attn_done_event: torch.cuda.Event,
    next_layer_id: int,           # -1 means last layer (no Phase A follows)
) -> Optional[torch.cuda.Event]:
    """
    Launch Phase C (save_kv for layer_id) and optionally Phase A (for next_layer_id)
    on info.comm_stream, gated by attn_done_event from main_stream.

    Sequence on comm_stream:
      1. wait(attn_done_event)         — wait for flash_attn(layer_id) to finish
      2. save_dp_local_kv(layer_id)    — write transfer + extend KV to out_cache_loc
      3. if next_layer_id >= 0:
             Phase A(next_layer_id)    — prefetch next layer's local + transfer KV
             record ltr_event

    Returns the ltr_event for next_layer_id, or None if next_layer_id < 0.

    Parameters
    ----------
    attn_done_event : CUDA Event recorded on main_stream after flash_attn.
    next_layer_id   : layer_id + 1 for normal layers; -1 for the last layer.
    """
    comm_stream = info.comm_stream
    next_ltr_event: Optional[torch.cuda.Event] = None

    if next_layer_id >= 0:
        next_ltr_event = torch.cuda.Event()

    with torch.cuda.stream(comm_stream):
        comm_stream.wait_event(attn_done_event)

        # Phase C: save KV for the just-finished layer
        save_dp_local_kv(
            info=info,
            layer=layer,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=token_to_kv_pool,
        )
        logger.debug(
            "[share_prefix_pipeline] Phase C (save_kv) launched on comm_stream: "
            "layer=%d dp_local_reqs=%d stream_id=%d",
            layer_id,
            len(info.dp_local_req_global_indices),
            comm_stream.cuda_stream,
        )

        # Phase A for next layer (if any).
        # Guard: next_layer_id must be within this PP rank's KV pool.  If it falls
        # outside (PP boundary), skip Phase A – the next PP rank owns that layer.
        if next_layer_id >= 0 and not _kv_pool_has_layer(token_to_kv_pool, next_layer_id):
            logger.debug(
                "[share_prefix_pipeline] Phase A skipped: next_layer_id=%d is outside "
                "PP rank KV pool (start=%d len=%d) — treating as last layer",
                next_layer_id,
                token_to_kv_pool.start_layer,
                len(token_to_kv_pool.kv_buffer),
            )
            next_layer_id = -1     # demote to last-layer path
            next_ltr_event = None  # discard the pre-allocated event

        if next_layer_id >= 0:
            next_kv_buf = token_to_kv_pool.get_key_buffer(next_layer_id)
            if info.total_transfer_tokens > 0:
                reset_transfer_region(info)
                fill_transfer_region_for_layer(info, next_layer_id, next_kv_buf)
                dist.all_reduce(
                    info.combined_kv_buf[
                        info.local_block_size : info.extend_block_start
                    ],
                    op=dist.ReduceOp.SUM,
                    group=tp_group.device_group,
                )
            fill_local_block_for_layer(info, next_kv_buf)
            next_ltr_event.record(comm_stream)
            logger.debug(
                "[share_prefix_pipeline] Phase A (next layer) launched on comm_stream: "
                "next_layer=%d xfer_toks=%d local_toks=%d stream_id=%d",
                next_layer_id,
                info.total_transfer_tokens,
                info.local_block_size,
                comm_stream.cuda_stream,
            )

    return next_ltr_event


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
