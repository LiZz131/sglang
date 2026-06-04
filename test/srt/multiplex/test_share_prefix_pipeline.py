"""
Unit tests for share-prefix pipeline overlap helpers.

Tests the split fill functions (fill_local_block_for_layer,
fill_extend_block_for_layer) and the async Phase A / Phase C launchers
(_launch_phase_a, _launch_phase_c_and_maybe_next_phase_a).

These tests use CPU tensors where possible and skip CUDA-only tests when
a GPU is not available.

Run with:
    pytest test/srt/multiplex/test_share_prefix_pipeline.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.multiplex.share_prefix_helper import (
    SharePrefixBatchInfo,
    _launch_phase_a,
    _launch_phase_c_and_maybe_next_phase_a,
    clear_persistent_share_prefix_bufs,
    fill_extend_block_for_layer,
    fill_local_and_extend_for_layer,
    fill_local_block_for_layer,
    fill_transfer_region_for_layer,
    reset_transfer_region,
    save_dp_local_kv,
)

CUDA_AVAILABLE = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

# Use a GPU with sufficient free memory.  GPU 0 may be occupied by training
# workloads in the CI environment; prefer GPU 1 when available.
def _pick_test_cuda_device() -> str:
    if not CUDA_AVAILABLE:
        return "cpu"
    for idx in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(idx)
        if free > 200 * 1024 * 1024:  # at least 200 MB
            return f"cuda:{idx}"
    return "cuda:0"  # fallback

_TEST_CUDA_DEVICE = _pick_test_cuda_device()


# ---------------------------------------------------------------------------
# Shared helper: build mock SharePrefixBatchInfo
# (identical to test_share_prefix_helper.py's _make_info)
# ---------------------------------------------------------------------------

def _make_info(
    seq_lens,
    local_prefix,
    min_prefix,
    max_prefix,
    max_rank,
    local_dp_rank,
    local_attn_tp_rank,
    dp_local_req_global_indices=None,
    device="cpu",
    kv_cache_dim=8,
    kv_lora_rank=6,
) -> SharePrefixBatchInfo:
    n = len(seq_lens)
    transfer_len = [max_prefix[i] - min_prefix[i] for i in range(n)]
    total_transfer = sum(transfer_len)

    transfer_offsets, acc = [], 0
    for tl in transfer_len:
        transfer_offsets.append(acc)
        acc += tl

    extend_hidden_starts, acc = [], 0
    for i in range(n):
        extend_hidden_starts.append(acc)
        acc += seq_lens[i] - max_prefix[i]

    prefix_indices_list, slot = [], 0
    for i in range(n):
        n_slots = local_prefix[i]
        prefix_indices_list.append(torch.arange(slot, slot + n_slots, dtype=torch.int64))
        slot += n_slots

    if dp_local_req_global_indices is None:
        dp_local_req_global_indices = list(range(n))

    dev = torch.device(device)

    local_block_starts, acc = [], 0
    for mp in min_prefix:
        local_block_starts.append(acc)
        acc += mp
    local_block_size = acc
    extend_block_start = local_block_size + total_transfer

    if local_block_size > 0:
        all_local_src_indices = torch.cat([
            prefix_indices_list[i][:min_prefix[i]]
            for i in range(n) if min_prefix[i] > 0
        ])
    else:
        all_local_src_indices = torch.empty(0, dtype=torch.int64)

    max_seq_len = max(seq_lens)
    page_table_cached = torch.zeros((n, max_seq_len), dtype=torch.int32)
    for i in range(n):
        min_p, max_p = min_prefix[i], max_prefix[i]
        seq_len = seq_lens[i]
        t_len = transfer_len[i]
        e_len = seq_len - max_p
        if min_p > 0:
            page_table_cached[i, :min_p] = torch.arange(
                local_block_starts[i], local_block_starts[i] + min_p, dtype=torch.int32)
        if t_len > 0:
            page_table_cached[i, min_p:max_p] = torch.arange(
                local_block_size + transfer_offsets[i],
                local_block_size + transfer_offsets[i] + t_len, dtype=torch.int32)
        if e_len > 0:
            page_table_cached[i, max_p:seq_len] = torch.arange(
                extend_block_start + extend_hidden_starts[i],
                extend_block_start + extend_hidden_starts[i] + e_len, dtype=torch.int32)

    cache_seqlens_tensor = torch.tensor(list(seq_lens), dtype=torch.int32)
    cu_seqlens_k_new_tensor = F.pad(
        torch.cumsum(cache_seqlens_tensor, dim=0, dtype=torch.int32), (1, 0))

    return SharePrefixBatchInfo(
        seq_lens=list(seq_lens),
        local_prefix=list(local_prefix),
        min_prefix=list(min_prefix),
        max_prefix=list(max_prefix),
        transfer_len=list(transfer_len),
        max_rank=list(max_rank),
        local_dp_rank=local_dp_rank,
        local_attn_tp_rank=local_attn_tp_rank,
        transfer_offsets=list(transfer_offsets),
        total_transfer_tokens=total_transfer,
        local_block_size=local_block_size,
        extend_block_start=extend_block_start,
        local_block_starts=list(local_block_starts),
        extend_hidden_starts=list(extend_hidden_starts),
        prefix_indices_list=prefix_indices_list,
        dp_local_req_global_indices=list(dp_local_req_global_indices),
        all_local_src_indices=all_local_src_indices,
        page_table_cached=page_table_cached,
        cache_seqlens_tensor=cache_seqlens_tensor,
        cu_seqlens_k_new_tensor=cu_seqlens_k_new_tensor,
        combined_kv_buf=None,
        kv_cache_dim=kv_cache_dim,
        kv_lora_rank=kv_lora_rank,
        device=dev,
    )


def _standard_batch(device="cpu"):
    """
    2 requests:
      req0: seq=20, local_prefix=4, min_prefix=2, max_prefix=8
      req1: seq=15, local_prefix=3, min_prefix=3, max_prefix=6

    Block layout:
      local_block_size   = 5  (2 + 3)
      total_transfer     = 9  (6 + 3)
      extend_block_start = 14
      total_combined     = 35
    """
    kv_cache_dim, kv_lora_rank = 8, 5
    seq_lens     = [20, 15]
    local_prefix = [4,  3]
    min_prefix   = [2,  3]
    max_prefix   = [8,  6]

    dev = torch.device(device)
    info = _make_info(
        seq_lens=seq_lens, local_prefix=local_prefix,
        min_prefix=min_prefix, max_prefix=max_prefix,
        max_rank=[0, 1], local_dp_rank=1, local_attn_tp_rank=0,
        device=device, kv_cache_dim=kv_cache_dim, kv_lora_rank=kv_lora_rank,
    )

    # Allocate combined_kv_buf directly (bypassing the persistent buffer with its
    # large minimum capacity) to keep the test self-contained and avoid OOM on
    # memory-constrained GPUs.
    total_combined = sum(seq_lens)
    info.combined_kv_buf = torch.zeros(
        total_combined, 1, kv_cache_dim, dtype=torch.float32, device=dev
    )
    info.kv_lora_rank = kv_lora_rank
    info.kv_cache_dim = kv_cache_dim
    info.device = dev

    # Pre-fill transfer_block with 9.0 (simulates post-all_reduce state)
    info.combined_kv_buf[info.local_block_size : info.extend_block_start] = 9.0

    total_slots = 100
    kv_buf = torch.zeros(total_slots, 1, kv_cache_dim, dtype=torch.float32, device=dev)
    for s in range(total_slots):
        kv_buf[s, 0, :] = float(s)

    info.prefix_indices_list[0] = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    info.prefix_indices_list[1] = torch.tensor([10, 11, 12], dtype=torch.int64)
    info.all_local_src_indices = torch.cat([
        info.prefix_indices_list[0][:min_prefix[0]],
        info.prefix_indices_list[1][:min_prefix[1]],
    ])

    total_model_extend = sum(s - mp for s, mp in zip(seq_lens, max_prefix))
    k_nope = torch.full((total_model_extend, 1, kv_lora_rank), 7.0, device=dev)
    k_pe   = torch.full((total_model_extend, 1, kv_cache_dim - kv_lora_rank), 3.0, device=dev)
    return info, kv_buf, k_nope, k_pe


# ---------------------------------------------------------------------------
# 1. TestFillLocalBlockForLayer
# ---------------------------------------------------------------------------

class TestFillLocalBlockForLayer:
    """fill_local_block_for_layer writes only the local_block region."""

    def setup_method(self):
        clear_persistent_share_prefix_bufs()

    def test_local_block_values(self):
        info, kv_buf, _, _ = _standard_batch()
        fill_local_block_for_layer(info, kv_buf)

        # req0 local: slots 0,1 → kv_buf values 0.0, 1.0
        assert info.combined_kv_buf[0, 0, 0].item() == pytest.approx(0.0)
        assert info.combined_kv_buf[1, 0, 0].item() == pytest.approx(1.0)

        # req1 local: slots 10,11,12 → kv_buf values 10.0, 11.0, 12.0
        assert info.combined_kv_buf[2, 0, 0].item() == pytest.approx(10.0)
        assert info.combined_kv_buf[3, 0, 0].item() == pytest.approx(11.0)
        assert info.combined_kv_buf[4, 0, 0].item() == pytest.approx(12.0)

    def test_transfer_block_untouched(self):
        """fill_local_block_for_layer must not touch the transfer_block."""
        info, kv_buf, _, _ = _standard_batch()
        info.combined_kv_buf[info.local_block_size : info.extend_block_start] = 99.0
        fill_local_block_for_layer(info, kv_buf)

        tb = info.combined_kv_buf[info.local_block_size : info.extend_block_start]
        assert tb.min().item() == pytest.approx(99.0)
        assert tb.max().item() == pytest.approx(99.0)

    def test_extend_block_untouched(self):
        """fill_local_block_for_layer must not touch the extend_block."""
        info, kv_buf, _, _ = _standard_batch()
        info.combined_kv_buf[info.extend_block_start :] = 55.0
        fill_local_block_for_layer(info, kv_buf)

        ext = info.combined_kv_buf[info.extend_block_start :]
        assert ext.min().item() == pytest.approx(55.0)
        assert ext.max().item() == pytest.approx(55.0)

    def test_empty_local_block(self):
        """When local_block_size == 0 the function is a no-op."""
        clear_persistent_share_prefix_bufs()
        info = _make_info(
            seq_lens=[5], local_prefix=[0], min_prefix=[0], max_prefix=[3],
            max_rank=[0], local_dp_rank=1, local_attn_tp_rank=0,
            kv_cache_dim=4, kv_lora_rank=3,
        )
        info.ensure_combined_buf_allocated(4, 3, torch.float32, torch.device("cpu"))
        info.combined_kv_buf.fill_(0.0)
        kv_buf = torch.ones(10, 1, 4)
        fill_local_block_for_layer(info, kv_buf)
        # combined_kv_buf should be untouched (all zeros)
        assert info.combined_kv_buf.abs().max().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. TestFillExtendBlockForLayer
# ---------------------------------------------------------------------------

class TestFillExtendBlockForLayer:
    """fill_extend_block_for_layer writes only the extend_block region."""

    def setup_method(self):
        clear_persistent_share_prefix_bufs()

    def test_extend_nope_values(self):
        info, _, k_nope, k_pe = _standard_batch()
        fill_extend_block_for_layer(info, k_nope, k_pe)

        ext = info.combined_kv_buf[info.extend_block_start :]
        nope_part = ext[:, 0, : info.kv_lora_rank]
        assert nope_part.allclose(torch.full_like(nope_part, 7.0))

    def test_extend_rope_values(self):
        info, _, k_nope, k_pe = _standard_batch()
        fill_extend_block_for_layer(info, k_nope, k_pe)

        ext = info.combined_kv_buf[info.extend_block_start :]
        rope_part = ext[:, 0, info.kv_lora_rank :]
        assert rope_part.allclose(torch.full_like(rope_part, 3.0))

    def test_local_block_untouched(self):
        """fill_extend_block_for_layer must not touch the local_block."""
        info, _, k_nope, k_pe = _standard_batch()
        info.combined_kv_buf[: info.local_block_size] = 42.0
        fill_extend_block_for_layer(info, k_nope, k_pe)

        lb = info.combined_kv_buf[: info.local_block_size]
        assert lb.min().item() == pytest.approx(42.0)
        assert lb.max().item() == pytest.approx(42.0)

    def test_transfer_block_untouched(self):
        """fill_extend_block_for_layer must not touch the transfer_block."""
        info, _, k_nope, k_pe = _standard_batch()
        info.combined_kv_buf[info.local_block_size : info.extend_block_start] = 33.0
        fill_extend_block_for_layer(info, k_nope, k_pe)

        tb = info.combined_kv_buf[info.local_block_size : info.extend_block_start]
        assert tb.min().item() == pytest.approx(33.0)
        assert tb.max().item() == pytest.approx(33.0)

    def test_split_equals_combined(self):
        """
        fill_local_block + fill_extend_block together should produce the same
        result as fill_local_and_extend_for_layer.
        """
        clear_persistent_share_prefix_bufs()
        info_split, kv_buf, k_nope, k_pe = _standard_batch()
        fill_local_block_for_layer(info_split, kv_buf)
        fill_extend_block_for_layer(info_split, k_nope, k_pe)

        clear_persistent_share_prefix_bufs()
        info_combined, kv_buf2, k_nope2, k_pe2 = _standard_batch()
        # pre-fill transfer block the same way
        info_combined.combined_kv_buf[
            info_combined.local_block_size : info_combined.extend_block_start
        ] = 9.0
        fill_local_and_extend_for_layer(info_combined, k_nope2, k_pe2, kv_buf2)

        assert info_split.combined_kv_buf.allclose(info_combined.combined_kv_buf), \
            "split fill should produce the same result as the combined call"


# ---------------------------------------------------------------------------
# 3. TestLaunchPhaseA (CUDA only)
# ---------------------------------------------------------------------------

class _MockTPGroup:
    """Minimal stub so _launch_phase_a signature is satisfied without NCCL."""
    class device_group:
        pass


@cuda_only
class TestLaunchPhaseA:
    """
    Run Phase A on comm_stream and verify combined_kv_buf contents match the
    result of the synchronous path after torch.cuda.synchronize().
    """

    def setup_method(self):
        clear_persistent_share_prefix_bufs()
        torch.cuda.empty_cache()

    def _cuda_batch(self):
        clear_persistent_share_prefix_bufs()
        info, kv_buf, k_nope, k_pe = _standard_batch(device=_TEST_CUDA_DEVICE)
        # Remove the pre-filled transfer block (Phase A will reset it)
        info.combined_kv_buf.zero_()
        return info, kv_buf, k_nope, k_pe

    def test_fill_local_block_via_phase_a_no_transfer(self):
        """
        When total_transfer_tokens > 0, Phase A normally includes all_reduce.
        Here we patch info to total_transfer_tokens=0 to test fill_local only
        without requiring NCCL.
        """
        info, kv_buf, k_nope, k_pe = self._cuda_batch()

        # Force zero-transfer path
        info.total_transfer_tokens = 0

        comm_stream = torch.cuda.Stream(device=_TEST_CUDA_DEVICE)
        ltr_event = _launch_phase_a(info, layer_id=0, kv_buf=kv_buf,
                                     tp_group=None, comm_stream=comm_stream)
        # ltr_event should be a CUDA Event
        assert isinstance(ltr_event, torch.cuda.Event)

        # Synchronize to let comm_stream complete
        torch.cuda.synchronize()

        # Verify local_block was filled
        # all_local_src_indices = [0,1,10,11,12]
        # kv_buf[0,0,0]=0.0, kv_buf[1,0,0]=1.0, kv_buf[10,0,0]=10.0 ...
        buf = info.combined_kv_buf.cpu()
        assert buf[0, 0, 0].item() == pytest.approx(0.0)
        assert buf[1, 0, 0].item() == pytest.approx(1.0)
        assert buf[2, 0, 0].item() == pytest.approx(10.0)

    def test_ltr_event_is_cuda_event(self):
        info, kv_buf, _, _ = self._cuda_batch()
        info.total_transfer_tokens = 0
        comm_stream = torch.cuda.Stream(device=_TEST_CUDA_DEVICE)
        ltr_event = _launch_phase_a(info, layer_id=3, kv_buf=kv_buf,
                                     tp_group=None, comm_stream=comm_stream)
        assert isinstance(ltr_event, torch.cuda.Event)
        torch.cuda.synchronize()

    def test_wait_event_blocks_main_stream(self):
        """
        After main_stream.wait_event(ltr_event), fill_extend should see local_block
        already written (because CUDA enforces the event dependency).
        """
        info, kv_buf, k_nope, k_pe = self._cuda_batch()
        info.total_transfer_tokens = 0

        comm_stream = torch.cuda.Stream(device=_TEST_CUDA_DEVICE)
        ltr_event = _launch_phase_a(info, layer_id=0, kv_buf=kv_buf,
                                     tp_group=None, comm_stream=comm_stream)

        torch.cuda.current_stream().wait_event(ltr_event)
        fill_extend_block_for_layer(info, k_nope, k_pe)
        torch.cuda.synchronize()

        buf = info.combined_kv_buf.cpu()
        # local_block check
        assert buf[0, 0, 0].item() == pytest.approx(0.0)
        # extend_block nope check
        ext = buf[info.extend_block_start :]
        assert ext[:, 0, : info.kv_lora_rank].min().item() == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# 4. TestPipelineEventLifecycle (CUDA only)
# ---------------------------------------------------------------------------

@cuda_only
class TestPipelineEventLifecycle:
    """
    Simulate N sequential calls to _launch_phase_c_and_maybe_next_phase_a and
    verify that:
      - _pending_ltr_event is set after each non-last layer
      - _pending_ltr_event is None (returned None) for the last layer
    """

    def setup_method(self):
        clear_persistent_share_prefix_bufs()
        torch.cuda.empty_cache()

    def _make_minimal_info(self):
        """
        Single request, no transfer (total_transfer_tokens=0), to avoid NCCL.
        Allocates combined_kv_buf directly to avoid the persistent buffer's
        large minimum capacity which can cause OOM on memory-constrained GPUs.
        """
        clear_persistent_share_prefix_bufs()
        kv_cache_dim, kv_lora_rank = 4, 3
        info = _make_info(
            seq_lens=[6], local_prefix=[2], min_prefix=[2], max_prefix=[2],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
            device=_TEST_CUDA_DEVICE, kv_cache_dim=kv_cache_dim, kv_lora_rank=kv_lora_rank,
        )
        # Allocate directly to avoid the large min-capacity persistent buffer
        info.combined_kv_buf = torch.zeros(
            6, 1, kv_cache_dim, dtype=torch.float32, device=_TEST_CUDA_DEVICE
        )
        info.kv_lora_rank = kv_lora_rank
        info.kv_cache_dim = kv_cache_dim
        info.device = torch.device(_TEST_CUDA_DEVICE)
        return info, kv_cache_dim, kv_lora_rank

    def test_pending_event_lifecycle_3_layers(self):
        """
        For a 3-layer forward pass (layer_ids 0, 1, 2):
          - After layer 0: _pending_ltr_event is set (points to layer 1's LTR)
          - After layer 1: _pending_ltr_event is set (points to layer 2's LTR)
          - After layer 2 (last): returned event is None
        """
        info, kv_cache_dim, kv_lora_rank = self._make_minimal_info()
        info.set_pipeline_comm_stream(torch.cuda.Stream())
        info.pipeline_last_layer_id = 2  # 3-layer pass: layers 0, 1, 2

        total_slots = 20
        kv_buf = torch.zeros(total_slots, 1, kv_cache_dim, dtype=torch.float32,
                             device=_TEST_CUDA_DEVICE)
        out_cache_loc = torch.zeros(0, dtype=torch.int64, device=_TEST_CUDA_DEVICE)

        class _FakeLayer:
            layer_id = 0

        class _FakeKVPool:
            start_layer = 0
            kv_buffer = [kv_buf] * 3   # layers 0, 1, 2 – same tensor, for _kv_pool_has_layer
            def get_key_buffer(self, layer_id):
                return kv_buf
            def set_mla_kv_buffer(self, *args, **kwargs):
                pass

        kv_pool = _FakeKVPool()

        for layer_id in range(3):
            _FakeLayer.layer_id = layer_id
            is_last = layer_id == info.pipeline_last_layer_id
            next_layer_id = -1 if is_last else layer_id + 1

            attn_done = torch.cuda.Event()
            attn_done.record()

            next_ltr = _launch_phase_c_and_maybe_next_phase_a(
                info=info, layer_id=layer_id, layer=_FakeLayer,
                out_cache_loc=out_cache_loc, token_to_kv_pool=kv_pool,
                tp_group=None, attn_done_event=attn_done,
                next_layer_id=next_layer_id,
            )
            info._pending_ltr_event = next_ltr

            if not is_last:
                assert next_ltr is not None, \
                    f"layer {layer_id}: expected ltr_event, got None"
                assert isinstance(next_ltr, torch.cuda.Event)
            else:
                assert next_ltr is None, \
                    f"last layer {layer_id}: expected None ltr_event"

        torch.cuda.synchronize()

    def test_set_pipeline_comm_stream(self):
        """set_pipeline_comm_stream correctly sets and clears comm_stream."""
        info, _, _ = self._make_minimal_info()
        assert info.comm_stream is None

        stream = torch.cuda.Stream()
        info.set_pipeline_comm_stream(stream)
        assert info.comm_stream is stream

        info.set_pipeline_comm_stream(None)
        assert info.comm_stream is None

    def test_pipeline_last_layer_id_default(self):
        """pipeline_last_layer_id defaults to -1 (not active)."""
        info, _, _ = self._make_minimal_info()
        assert info.pipeline_last_layer_id == -1

    def test_pending_ltr_event_default(self):
        """_pending_ltr_event defaults to None."""
        info, _, _ = self._make_minimal_info()
        assert info._pending_ltr_event is None


# ---------------------------------------------------------------------------
# 5. TestPipelineMatchesSynchronous (CUDA only)
# ---------------------------------------------------------------------------

@cuda_only
class TestPipelineMatchesSynchronous:
    """
    Run fill_local_block + fill_extend_block through the async Phase A path
    (without NCCL, total_transfer_tokens=0) and verify the combined_kv_buf
    result matches the synchronous fill_local_and_extend_for_layer path.
    """

    def setup_method(self):
        clear_persistent_share_prefix_bufs()
        torch.cuda.empty_cache()

    def test_combined_kv_buf_matches_sync(self):
        """
        Pipeline path (fill_local on comm_stream, fill_extend on main_stream after
        wait_event) must produce the same local_block + extend_block as the sync path.

        We test with total_transfer_tokens=0 to avoid NCCL.  Both paths start from
        a zeroed combined_kv_buf so the comparison is well-defined.
        """
        device = _TEST_CUDA_DEVICE

        # --- Synchronous reference: start from zeroed buffer ---
        info_sync, kv_buf_sync, k_nope_sync, k_pe_sync = _standard_batch(device)
        info_sync.total_transfer_tokens = 0  # skip all_reduce for single-rank test
        info_sync.combined_kv_buf.zero_()   # same baseline as pipeline path
        fill_local_and_extend_for_layer(info_sync, k_nope_sync, k_pe_sync, kv_buf_sync)
        torch.cuda.synchronize()
        ref = info_sync.combined_kv_buf.clone().cpu()

        # --- Pipeline path ---
        info_pipe, kv_buf_pipe, k_nope_pipe, k_pe_pipe = _standard_batch(device)
        info_pipe.total_transfer_tokens = 0
        info_pipe.combined_kv_buf.zero_()

        comm_stream = torch.cuda.Stream(device=_TEST_CUDA_DEVICE)
        ltr_event = _launch_phase_a(info_pipe, layer_id=0, kv_buf=kv_buf_pipe,
                                     tp_group=None, comm_stream=comm_stream)

        torch.cuda.current_stream().wait_event(ltr_event)
        fill_extend_block_for_layer(info_pipe, k_nope_pipe, k_pe_pipe)
        torch.cuda.synchronize()
        got = info_pipe.combined_kv_buf.cpu()

        assert got.allclose(ref), (
            f"Pipeline path differs from sync path.\n"
            f"Max diff: {(got - ref).abs().max().item():.6f}"
        )

    def test_extend_block_independent_of_phase_a(self):
        """
        extend_block region should be fully determined by fill_extend_block_for_layer,
        independent of what Phase A does (Phase A does not write to extend_block).
        """
        device = _TEST_CUDA_DEVICE
        clear_persistent_share_prefix_bufs()
        info, kv_buf, k_nope, k_pe = _standard_batch(device)
        info.total_transfer_tokens = 0
        info.combined_kv_buf.zero_()

        comm_stream = torch.cuda.Stream(device=_TEST_CUDA_DEVICE)
        ltr_event = _launch_phase_a(info, layer_id=0, kv_buf=kv_buf,
                                     tp_group=None, comm_stream=comm_stream)
        torch.cuda.current_stream().wait_event(ltr_event)
        fill_extend_block_for_layer(info, k_nope, k_pe)
        torch.cuda.synchronize()

        ext = info.combined_kv_buf[info.extend_block_start :].cpu()
        nope_part = ext[:, 0, : info.kv_lora_rank]
        rope_part = ext[:, 0, info.kv_lora_rank :]
        assert nope_part.allclose(torch.full_like(nope_part, 7.0)), \
            "extend_block nope should be 7.0"
        assert rope_part.allclose(torch.full_like(rope_part, 3.0)), \
            "extend_block rope should be 3.0"

    def test_local_block_filled_before_flash_attn_placeholder(self):
        """
        Verify that after waiting for ltr_event, local_block values are accessible
        (simulates flash_attn reading from combined_kv_buf after the dependency).
        """
        device = _TEST_CUDA_DEVICE
        clear_persistent_share_prefix_bufs()
        info, kv_buf, k_nope, k_pe = _standard_batch(device)
        info.total_transfer_tokens = 0
        info.combined_kv_buf.zero_()

        comm_stream = torch.cuda.Stream(device=_TEST_CUDA_DEVICE)
        ltr_event = _launch_phase_a(info, layer_id=0, kv_buf=kv_buf,
                                     tp_group=None, comm_stream=comm_stream)
        # main_stream waits (CUDA dependency)
        torch.cuda.current_stream().wait_event(ltr_event)
        # Synchronize to verify from CPU
        torch.cuda.synchronize()

        buf = info.combined_kv_buf.cpu()
        # Slot 0: kv_buf[all_local_src_indices[0]] = kv_buf[0] → 0.0
        assert buf[0, 0, 0].item() == pytest.approx(0.0)
        # Slot 2: kv_buf[all_local_src_indices[2]] = kv_buf[10] → 10.0
        assert buf[2, 0, 0].item() == pytest.approx(10.0)

# ---------------------------------------------------------------------------
# 6. TestPipelineInitForSplitPrefill
# ---------------------------------------------------------------------------

class TestPipelineInitForSplitPrefill:
    """
    Unit tests for SharePrefixBatchInfo.init_pipeline_state_for_forward() and
    the _kv_pool_has_layer() guard, covering:
      - first / subsequent sub-forward init behaviour
      - force_reset semantics
      - _pending_ltr_event preservation across sub-forwards
      - KV pool boundary guard (PP safety)
    """

    def setup_method(self):
        clear_persistent_share_prefix_bufs()

    # ------------------------------------------------------------------
    # 6a. init_pipeline_state_for_forward – CPU-only (no CUDA needed)
    # ------------------------------------------------------------------

    def test_first_call_resets_pending_event(self):
        """First sub-forward must clear any stale _pending_ltr_event."""
        info, _, _, _ = _standard_batch()
        # Simulate a stale event left over from a prior run
        info._pending_ltr_event = object()  # any non-None sentinel
        info.pipeline_initialized = False   # pretend new batch

        info.init_pipeline_state_for_forward(pp_last_layer_id=59)

        assert info._pending_ltr_event is None
        assert info.pipeline_last_layer_id == 59
        assert info.pipeline_initialized is True

    def test_subsequent_call_preserves_pending_event(self):
        """Subsequent sub-forward must NOT overwrite _pending_ltr_event."""
        info, _, _, _ = _standard_batch()
        sentinel = object()

        # First sub-forward
        info.init_pipeline_state_for_forward(pp_last_layer_id=59)
        # Simulate Phase A having produced an event during sub-forward 1
        info._pending_ltr_event = sentinel

        # Second sub-forward (pipeline_initialized already True)
        info.init_pipeline_state_for_forward(pp_last_layer_id=59)

        assert info._pending_ltr_event is sentinel, \
            "Subsequent sub-forward must not clear _pending_ltr_event"

    def test_force_reset_clears_pending_event(self):
        """force_reset=True always clears _pending_ltr_event."""
        info, _, _, _ = _standard_batch()
        sentinel = object()
        info._pending_ltr_event = sentinel
        info.pipeline_initialized = True   # already initialised

        info.init_pipeline_state_for_forward(pp_last_layer_id=59, force_reset=True)

        assert info._pending_ltr_event is None
        assert info.pipeline_last_layer_id == 59

    def test_pipeline_last_layer_id_set_to_pp_boundary(self):
        """pipeline_last_layer_id must equal pp_last_layer_id, not sub-forward end."""
        info, _, _, _ = _standard_batch()
        info.pipeline_initialized = False

        # Simulates: 60-layer model, PP=1, sub-forward processes layers 0-14
        info.init_pipeline_state_for_forward(pp_last_layer_id=59)

        # Must be 59 (whole PP rank last layer), NOT 14 (sub-forward end)
        assert info.pipeline_last_layer_id == 59

    def test_pipeline_initialized_gates_resets(self):
        """pipeline_initialized correctly gates the reset across multiple calls."""
        info, _, _, _ = _standard_batch()

        assert info.pipeline_initialized is False

        # First call → initialises
        info.init_pipeline_state_for_forward(pp_last_layer_id=29)
        assert info.pipeline_initialized is True
        assert info.pipeline_last_layer_id == 29

        # Manufacture a pending event
        sentinel = object()
        info._pending_ltr_event = sentinel

        # Second call (same batch, next sub-forward) → must not reinit
        info.init_pipeline_state_for_forward(pp_last_layer_id=29)
        assert info._pending_ltr_event is sentinel

    # ------------------------------------------------------------------
    # 6b. _kv_pool_has_layer – CPU-only
    # ------------------------------------------------------------------

    def test_kv_pool_has_layer_in_range(self):
        """Layers within [start_layer, start_layer+len) must return True."""
        from sglang.srt.multiplex.share_prefix_helper import _kv_pool_has_layer

        class _FakePool:
            start_layer = 10
            kv_buffer = [None] * 5   # layers 10..14

        pool = _FakePool()
        for lid in [10, 11, 12, 13, 14]:
            assert _kv_pool_has_layer(pool, lid), f"layer {lid} should be in pool"

    def test_kv_pool_has_layer_out_of_range(self):
        """Layers outside the pool range must return False."""
        from sglang.srt.multiplex.share_prefix_helper import _kv_pool_has_layer

        class _FakePool:
            start_layer = 10
            kv_buffer = [None] * 5   # layers 10..14

        pool = _FakePool()
        for lid in [9, 15, 20]:
            assert not _kv_pool_has_layer(pool, lid), f"layer {lid} should NOT be in pool"

    # ------------------------------------------------------------------
    # 6c. Cross-sub-forward CUDA event preservation (CUDA only)
    # ------------------------------------------------------------------

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_pending_ltr_event_survives_second_init_call(self):
        """
        Simulate two sub-forward init calls:
          1. First call  → resets _pending_ltr_event, launches Phase A → ltr_event
          2. Second call → must NOT clear ltr_event
          3. main_stream.wait_event(ltr_event) must succeed
        """
        device = _TEST_CUDA_DEVICE
        info, kv_buf, k_nope, k_pe = _standard_batch(device)
        info.total_transfer_tokens = 0
        info.combined_kv_buf.zero_()

        comm_stream = torch.cuda.Stream(device=device)
        info.set_pipeline_comm_stream(comm_stream)

        # --- Sub-forward 1 init ---
        info.init_pipeline_state_for_forward(pp_last_layer_id=59)
        assert info._pending_ltr_event is None

        # Simulate Phase A launched during sub-forward 1
        ltr_event = _launch_phase_a(info, layer_id=0, kv_buf=kv_buf,
                                     tp_group=None, comm_stream=comm_stream)
        info._pending_ltr_event = ltr_event

        # --- Sub-forward 2 init (should NOT clear ltr_event) ---
        info.init_pipeline_state_for_forward(pp_last_layer_id=59)
        assert info._pending_ltr_event is ltr_event, \
            "Second sub-forward init must not clear _pending_ltr_event"

        # Verify the event is usable: main_stream waits on it
        torch.cuda.current_stream().wait_event(info._pending_ltr_event)
        fill_extend_block_for_layer(info, k_nope, k_pe)
        torch.cuda.synchronize()

        # Extend block should be filled correctly
        ext = info.combined_kv_buf[info.extend_block_start :].cpu()
        assert ext[:, 0, : info.kv_lora_rank].min().item() == pytest.approx(7.0)

    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_kv_pool_guard_demotes_next_layer_to_last(self):
        """
        When next_layer_id is outside the KV pool, _launch_phase_c_and_maybe_next_phase_a
        must return None (treating the layer as last) without crashing.
        """
        device = _TEST_CUDA_DEVICE
        info, kv_buf, k_nope, k_pe = _standard_batch(device)
        info.total_transfer_tokens = 0
        info.combined_kv_buf.zero_()

        comm_stream = torch.cuda.Stream(device=device)
        info.set_pipeline_comm_stream(comm_stream)
        info.pipeline_last_layer_id = 13  # PP rank has layers 0..13

        class _FakeLayer:
            layer_id = 13

        class _FakePool:
            start_layer = 0
            kv_buffer = [kv_buf] * 14   # layers 0..13 only

            def get_key_buffer(self, lid):
                from sglang.srt.multiplex.share_prefix_helper import _kv_pool_has_layer
                if not _kv_pool_has_layer(self, lid):
                    raise IndexError(f"layer {lid} out of range")
                return self.kv_buffer[lid - self.start_layer]

            def set_mla_kv_buffer(self, *a, **kw):
                pass

        out_cache_loc = torch.zeros(0, dtype=torch.int64, device=device)
        attn_done = torch.cuda.Event()
        attn_done.record()

        # next_layer_id=14 is outside pool[0..13] → should return None safely
        result = _launch_phase_c_and_maybe_next_phase_a(
            info=info, layer_id=13, layer=_FakeLayer,
            out_cache_loc=out_cache_loc, token_to_kv_pool=_FakePool(),
            tp_group=None, attn_done_event=attn_done,
            next_layer_id=14,   # out of range
        )
        torch.cuda.synchronize()
        assert result is None, \
            "Should return None when next_layer_id is outside KV pool range"
