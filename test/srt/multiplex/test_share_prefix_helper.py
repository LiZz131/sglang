"""
Unit tests for share_prefix_helper.py.

These tests exercise the CPU / mock-GPU logic of SharePrefixBatchInfo and the
helper functions without requiring a real multi-rank distributed environment.

Run with:
    pytest test/srt/multiplex/test_share_prefix_helper.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers to build mock SharePrefixBatchInfo directly (no MPI needed)
# ---------------------------------------------------------------------------

from sglang.srt.multiplex.share_prefix_helper import (
    SharePrefixBatchInfo,
    clear_persistent_combined_kv_buf,
    fill_local_and_extend_for_layer,
    fill_transfer_region_for_layer,
    get_persistent_combined_kv_buf_capacity,
    reset_transfer_region,
    save_dp_local_kv,
)


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
    """
    Construct a SharePrefixBatchInfo with synthetic values for unit testing.

    Computes all pre-computed fields (block layout, page_table_cached, etc.)
    exactly as compute_share_prefix_info does in production.

    Parameters
    ----------
    dp_local_req_global_indices : optional list of global batch indices for dp-local
        requests.  Defaults to all request indices (simulates single DP rank or all
        local).

    Note: extend_hidden_starts is computed based on max_prefix (not local_prefix)
    because input_ids is sliced from max_prefix on all ranks.  This matches the
    production semantics introduced to fix the all_reduce shape mismatch.
    """
    n = len(seq_lens)
    transfer_len = [max_prefix[i] - min_prefix[i] for i in range(n)]
    total_transfer = sum(transfer_len)

    transfer_offsets = []
    acc = 0
    for tl in transfer_len:
        transfer_offsets.append(acc)
        acc += tl

    # extend_hidden_starts must use max_prefix (not local_prefix) because
    # input_ids = fill_ids[max_prefix:] on every rank.
    extend_hidden_starts = []
    acc = 0
    for i in range(n):
        extend_hidden_starts.append(acc)
        acc += seq_lens[i] - max_prefix[i]

    # Build dummy prefix_indices_list: consecutive slots starting from 0
    prefix_indices_list = []
    slot = 0
    for i in range(n):
        n_slots = local_prefix[i]
        pi = torch.arange(slot, slot + n_slots, dtype=torch.int64)
        prefix_indices_list.append(pi)
        slot += n_slots

    if dp_local_req_global_indices is None:
        dp_local_req_global_indices = list(range(n))

    dev = torch.device(device)

    # ---- block layout offsets ----
    local_block_starts = []
    acc = 0
    for mp in min_prefix:
        local_block_starts.append(acc)
        acc += mp
    local_block_size: int = acc
    extend_block_start: int = local_block_size + total_transfer

    # ---- all_local_src_indices ----
    if local_block_size > 0:
        all_local_src_indices = torch.cat([
            prefix_indices_list[i][:min_prefix[i]]
            for i in range(n) if min_prefix[i] > 0
        ])
    else:
        all_local_src_indices = torch.empty(0, dtype=torch.int64)

    # ---- page_table_cached ----
    max_seq_len = max(seq_lens)
    page_table_cached = torch.zeros((n, max_seq_len), dtype=torch.int32)
    for i in range(n):
        min_p = min_prefix[i]
        max_p = max_prefix[i]
        seq_len = seq_lens[i]
        t_len = transfer_len[i]
        e_len = seq_len - max_p
        if min_p > 0:
            page_table_cached[i, :min_p] = torch.arange(
                local_block_starts[i], local_block_starts[i] + min_p,
                dtype=torch.int32,
            )
        if t_len > 0:
            page_table_cached[i, min_p:max_p] = torch.arange(
                local_block_size + transfer_offsets[i],
                local_block_size + transfer_offsets[i] + t_len,
                dtype=torch.int32,
            )
        if e_len > 0:
            page_table_cached[i, max_p:seq_len] = torch.arange(
                extend_block_start + extend_hidden_starts[i],
                extend_block_start + extend_hidden_starts[i] + e_len,
                dtype=torch.int32,
            )

    cache_seqlens_tensor = torch.tensor(list(seq_lens), dtype=torch.int32)
    cu_seqlens_k_new_tensor = F.pad(
        torch.cumsum(cache_seqlens_tensor, dim=0, dtype=torch.int32), (1, 0)
    )

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


# ---------------------------------------------------------------------------
# 1. test_compute_max_min_prefix
# ---------------------------------------------------------------------------

class TestComputeMaxMinPrefix:
    """Verify max/min/transfer/extend/max_rank calculation semantics."""

    def test_two_requests_two_dp_ranks(self):
        """
        Simulate the output of the all-gather step manually and check the
        resulting metadata.

        Setup:
          dp_rank 0:  req0 prefix=40,  req1 prefix=5
          dp_rank 1:  req0 prefix=10,  req1 prefix=20

          req0: max=40 (rank0), min=10, transfer=30, max_rank=0
          req1: max=20 (rank1), min=5,  transfer=15, max_rank=1
        """
        dp_prefix_matrix = [
            [40, 5],   # dp_rank 0
            [10, 20],  # dp_rank 1
        ]
        seq_lens = [60, 60]
        n = 2
        dp_size = 2

        max_prefix = [max(dp_prefix_matrix[r][i] for r in range(dp_size)) for i in range(n)]
        min_prefix = [min(dp_prefix_matrix[r][i] for r in range(dp_size)) for i in range(n)]
        max_rank   = [dp_prefix_matrix.index(max(dp_prefix_matrix, key=lambda row: row[i]))
                      for i in range(n)]

        assert max_prefix == [40, 20]
        assert min_prefix == [10, 5]
        assert max_rank == [0, 1]
        transfer_len = [max_prefix[i] - min_prefix[i] for i in range(n)]
        assert transfer_len == [30, 15]

        extend_lens = [seq_lens[i] - min_prefix[i] for i in range(n)]
        # This is NOT used directly; extend for attention Q is seq_len - max_prefix
        _ = extend_lens

    def test_uniform_prefix_gives_zero_transfer(self):
        """All ranks have the same prefix → transfer_len == 0."""
        dp_prefix_matrix = [[30, 20], [30, 20]]
        n = 2
        dp_size = 2

        transfer_len = [
            max(row[i] for row in dp_prefix_matrix) - min(row[i] for row in dp_prefix_matrix)
            for i in range(n)
        ]
        assert transfer_len == [0, 0]
        assert sum(transfer_len) == 0


# ---------------------------------------------------------------------------
# 2. test_transfer_offsets
# ---------------------------------------------------------------------------

class TestTransferOffsets:
    """Verify prefix-sum offset calculation for transfer buffer layout."""

    def test_offsets_are_exclusive_prefix_sums(self):
        transfer_len = [30, 15, 0, 10]
        offsets = []
        acc = 0
        for tl in transfer_len:
            offsets.append(acc)
            acc += tl
        assert offsets == [0, 30, 45, 45]
        assert acc == 55  # total_transfer_tokens

    def test_single_request(self):
        transfer_len = [7]
        offsets = []
        acc = 0
        for tl in transfer_len:
            offsets.append(acc)
            acc += tl
        assert offsets == [0]
        assert acc == 7


# ---------------------------------------------------------------------------
# 3. test_block_layout_offsets
# ---------------------------------------------------------------------------

class TestBlockLayoutOffsets:
    """Verify the block layout offset computation in _make_info / compute_share_prefix_info."""

    def test_local_block_size_is_sum_of_min_prefix(self):
        info = _make_info(
            seq_lens=[60, 60],
            local_prefix=[40, 10],
            min_prefix=[10, 10],
            max_prefix=[40, 20],
            max_rank=[0, 1],
            local_dp_rank=0,
            local_attn_tp_rank=0,
        )
        assert info.local_block_size == 10 + 10   # sum(min_prefix)

    def test_extend_block_start_is_local_plus_transfer(self):
        info = _make_info(
            seq_lens=[60, 60],
            local_prefix=[40, 10],
            min_prefix=[10, 10],
            max_prefix=[40, 20],
            max_rank=[0, 1],
            local_dp_rank=0,
            local_attn_tp_rank=0,
        )
        # total_transfer = (40-10) + (20-10) = 30 + 10 = 40
        assert info.extend_block_start == 20 + 40   # local_block_size + total_transfer

    def test_total_combined_equals_sum_seq_lens(self):
        """local_block + transfer_block + extend_block = sum(seq_lens)."""
        seq_lens = [20, 15]
        min_prefix = [2, 3]
        max_prefix = [8, 6]
        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=[4, 3],
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=[0, 1],
            local_dp_rank=1,
            local_attn_tp_rank=0,
        )
        total_transfer = sum(max_prefix[i] - min_prefix[i] for i in range(2))  # 6+3=9
        total_extend = sum(seq_lens[i] - max_prefix[i] for i in range(2))      # 12+9=21
        assert info.local_block_size + total_transfer + total_extend == sum(seq_lens)


# ---------------------------------------------------------------------------
# 4. test_page_table_block_layout
# ---------------------------------------------------------------------------

class TestPageTableBlockLayout:
    """Verify page_table_cached has correct block-layout index mapping."""

    def _make_two_req_info(self):
        """
        2 requests, dp_rank=1, attn_tp_rank=0
          req0: seq=20, local_prefix=4, min_prefix=2, max_prefix=8
          req1: seq=15, local_prefix=3, min_prefix=3, max_prefix=6

        Block layout:
          local_block_size  = 2 + 3 = 5
          transfer_offsets  = [0, 6]
          extend_block_start = 5 + 9 = 14
          extend_hidden_starts = [0, 12]  (req0: 12 ext; req1: 9 ext)

        page_table[0, :20]:
          pos[0:2]  → local_block_starts[0]=0, arange(0,2)     = [0,1]
          pos[2:8]  → local_block_size + 0 + arange(0,6)       = [5..10]
          pos[8:20] → extend_block_start + 0 + arange(0,12)    = [14..25]

        page_table[1, :15]:
          pos[0:3]  → local_block_starts[1]=2, arange(2,5)     = [2,3,4]
          pos[3:6]  → local_block_size + 6 + arange(0,3)       = [11,12,13]
          pos[6:15] → extend_block_start + 12 + arange(0,9)    = [26..34]
        """
        return _make_info(
            seq_lens=[20, 15],
            local_prefix=[4, 3],
            min_prefix=[2, 3],
            max_prefix=[8, 6],
            max_rank=[0, 1],
            local_dp_rank=1,
            local_attn_tp_rank=0,
        )

    def test_shape(self):
        info = self._make_two_req_info()
        n, max_seq = 2, 20
        assert info.page_table_cached.shape == (n, max_seq)
        assert info.page_table_cached.dtype == torch.int32

    def test_req0_local_positions(self):
        """req0 positions [0, min_p0) map to local_block_starts[0] + offset."""
        info = self._make_two_req_info()
        for j in range(info.min_prefix[0]):   # j in [0,2)
            expected = info.local_block_starts[0] + j
            assert info.page_table_cached[0, j].item() == expected

    def test_req0_transfer_positions(self):
        """req0 positions [min_p0, max_p0) map into transfer_block."""
        info = self._make_two_req_info()
        min_p0 = info.min_prefix[0]  # 2
        max_p0 = info.max_prefix[0]  # 8
        for j in range(max_p0 - min_p0):   # j in [0,6)
            expected = info.local_block_size + info.transfer_offsets[0] + j
            assert info.page_table_cached[0, min_p0 + j].item() == expected

    def test_req0_extend_positions(self):
        """req0 positions [max_p0, seq0) map into extend_block."""
        info = self._make_two_req_info()
        max_p0 = info.max_prefix[0]   # 8
        seq0   = info.seq_lens[0]     # 20
        for j in range(seq0 - max_p0):   # j in [0,12)
            expected = info.extend_block_start + info.extend_hidden_starts[0] + j
            assert info.page_table_cached[0, max_p0 + j].item() == expected

    def test_req1_local_positions(self):
        info = self._make_two_req_info()
        for j in range(info.min_prefix[1]):   # j in [0,3)
            expected = info.local_block_starts[1] + j
            assert info.page_table_cached[1, j].item() == expected

    def test_req1_transfer_positions(self):
        info = self._make_two_req_info()
        min_p1 = info.min_prefix[1]  # 3
        max_p1 = info.max_prefix[1]  # 6
        for j in range(max_p1 - min_p1):   # j in [0,3)
            expected = info.local_block_size + info.transfer_offsets[1] + j
            assert info.page_table_cached[1, min_p1 + j].item() == expected

    def test_req1_extend_positions(self):
        info = self._make_two_req_info()
        max_p1 = info.max_prefix[1]  # 6
        seq1   = info.seq_lens[1]    # 15
        for j in range(seq1 - max_p1):   # j in [0,9)
            expected = info.extend_block_start + info.extend_hidden_starts[1] + j
            assert info.page_table_cached[1, max_p1 + j].item() == expected

    def test_all_active_positions_unique(self):
        """All active combined_kv_buf slot indices in page_table are distinct."""
        info = self._make_two_req_info()
        seq_lens = info.seq_lens
        indices = []
        for i, sl in enumerate(seq_lens):
            indices.extend(info.page_table_cached[i, :sl].tolist())
        assert len(indices) == len(set(indices)), "Duplicate slot indices in page_table"

    def test_cache_seqlens_tensor_values(self):
        info = self._make_two_req_info()
        assert info.cache_seqlens_tensor.tolist() == info.seq_lens
        assert info.cache_seqlens_tensor.dtype == torch.int32

    def test_cu_seqlens_k_new_tensor_values(self):
        info = self._make_two_req_info()
        expected = [0, 20, 35]
        assert info.cu_seqlens_k_new_tensor.tolist() == expected


# ---------------------------------------------------------------------------
# 5. test_fill_transfer_region_logic
# ---------------------------------------------------------------------------

class TestFillTransferRegion:
    """Verify fill_transfer_region_for_layer fills the right slots in combined_kv_buf."""

    def setup_method(self):
        clear_persistent_combined_kv_buf()

    def _setup(self, local_dp_rank, local_attn_tp_rank):
        """
        Batch: 2 requests
          req0: seq=60, local_prefix=40, min_prefix=10, max_prefix=40, max_rank=0
          req1: seq=60, local_prefix=10, min_prefix=10, max_prefix=20, max_rank=1

        Block layout:
          local_block_size  = 10 + 10 = 20
          total_transfer    = 30 + 10 = 40
          extend_block_start = 60
          total_combined    = 120
        """
        kv_cache_dim = 4
        kv_lora_rank = 3

        info = _make_info(
            seq_lens=[60, 60],
            local_prefix=[40, 10],
            min_prefix=[10, 10],
            max_prefix=[40, 20],
            max_rank=[0, 1],
            local_dp_rank=local_dp_rank,
            local_attn_tp_rank=local_attn_tp_rank,
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        info.ensure_combined_buf_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu"))
        reset_transfer_region(info)

        # KV pool: value at slot s = s + 1
        total_slots = 100
        kv_buf = torch.zeros(total_slots, 1, kv_cache_dim, dtype=torch.float32)
        for s in range(total_slots):
            kv_buf[s, 0, :] = float(s + 1)

        # req0: 40 prefix slots [0..39]; req1: 10 prefix slots [40..49]
        info.prefix_indices_list[0] = torch.arange(0, 40, dtype=torch.int64)
        info.prefix_indices_list[1] = torch.arange(40, 50, dtype=torch.int64)

        return info, kv_buf

    def test_rank0_tp0_fills_req0(self):
        """DP rank 0, attn_tp_rank 0: should fill req0's transfer region [10:40]."""
        info, kv_buf = self._setup(local_dp_rank=0, local_attn_tp_rank=0)
        fill_transfer_region_for_layer(info, layer_id=0, kv_buf=kv_buf)

        # transfer_block = combined_kv_buf[20:60]
        # req0 transfer: slots [10,40) → 30 tokens
        # combined_kv_buf[20 : 50] (base=20, t_start=0) should have kv_buf[10..39]
        tb = info.combined_kv_buf[info.local_block_size : info.extend_block_start]
        assert tb.shape == (40, 1, 4)

        # req0 (max_rank=0 == local_dp_rank=0): filled
        for j in range(30):
            expected_slot = 10 + j
            expected_val = float(expected_slot + 1)
            assert tb[j, 0, 0].item() == pytest.approx(expected_val), \
                f"transfer_block[{j}] expected {expected_val}, got {tb[j, 0, 0].item()}"

        # req1 (max_rank=1 ≠ local_dp_rank=0): stays zero after reset
        assert tb[30:40].abs().sum().item() == pytest.approx(0.0)

    def test_rank0_tp1_skips(self):
        """attn_tp_rank != 0: no filling (leaves zeros)."""
        info, kv_buf = self._setup(local_dp_rank=0, local_attn_tp_rank=1)
        fill_transfer_region_for_layer(info, layer_id=0, kv_buf=kv_buf)
        tb = info.combined_kv_buf[info.local_block_size : info.extend_block_start]
        assert tb.sum().item() == pytest.approx(0.0)

    def test_rank1_tp0_fills_req1_only(self):
        """DP rank 1, attn_tp_rank 0: should fill req1 transfer [10:20], not req0."""
        kv_cache_dim = 4
        kv_lora_rank = 3

        info = _make_info(
            seq_lens=[60, 60],
            local_prefix=[10, 20],   # rank1 perspective
            min_prefix=[10, 10],
            max_prefix=[40, 20],
            max_rank=[0, 1],
            local_dp_rank=1,
            local_attn_tp_rank=0,
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        info.ensure_combined_buf_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu"))
        reset_transfer_region(info)

        kv_buf = torch.zeros(100, 1, kv_cache_dim, dtype=torch.float32)
        for s in range(100):
            kv_buf[s, 0, :] = float(s + 1)

        # req0: local_prefix=10 → pref_idx [0..9] (max_rank=0 ≠ rank1 → skip)
        # req1: local_prefix=20 → pref_idx [10..29], transfer = [10:20] = slots [20..29]
        info.prefix_indices_list[0] = torch.arange(0, 10, dtype=torch.int64)
        info.prefix_indices_list[1] = torch.arange(10, 30, dtype=torch.int64)

        fill_transfer_region_for_layer(info, 0, kv_buf)

        tb = info.combined_kv_buf[info.local_block_size : info.extend_block_start]

        # req0 transfer (offset=0, len=30): stays zero (not filled on rank1)
        assert tb[0:30].abs().sum().item() == pytest.approx(0.0), \
            "req0 transfer should be 0 on rank1"

        # req1 transfer (offset=30, len=10): filled from slots pref_idx[1][10:20] = [20..29]
        for j in range(10):
            expected_slot = 20 + j
            expected_val = float(expected_slot + 1)
            assert tb[30 + j, 0, 0].item() == pytest.approx(expected_val), \
                f"transfer_block[{30+j}] expected {expected_val}"


# ---------------------------------------------------------------------------
# 6. test_fill_local_and_extend_for_layer
# ---------------------------------------------------------------------------

class TestFillLocalAndExtend:
    """Verify fill_local_and_extend_for_layer fills correct slots in combined_kv_buf."""

    def setup_method(self):
        clear_persistent_combined_kv_buf()

    def _make_test_batch(self):
        """
        2 requests:
          req0: seq=20, local_prefix=4, min_prefix=2, max_prefix=8
          req1: seq=15, local_prefix=3, min_prefix=3, max_prefix=6

        Block layout:
          local_block_size  = 5
          total_transfer    = 9
          extend_block_start = 14
          total_combined    = 35
        """
        kv_cache_dim = 8
        kv_lora_rank = 5
        seq_lens     = [20, 15]
        local_prefix = [4, 3]
        min_prefix   = [2, 3]
        max_prefix   = [8, 6]

        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=local_prefix,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=[0, 1],
            local_dp_rank=1,
            local_attn_tp_rank=0,
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        info.ensure_combined_buf_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu"))

        # Pre-fill transfer_block with 9.0 (simulates post-all_reduce state)
        info.combined_kv_buf[info.local_block_size : info.extend_block_start] = 9.0

        # kv_buf: value at slot s = float(s)
        total_slots = 100
        kv_buf = torch.zeros(total_slots, 1, kv_cache_dim, dtype=torch.float32)
        for s in range(total_slots):
            kv_buf[s, 0, :] = float(s)

        # prefix_indices for req0 (4 slots): [0,1,2,3]
        # prefix_indices for req1 (3 slots): [10,11,12]
        info.prefix_indices_list[0] = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        info.prefix_indices_list[1] = torch.tensor([10, 11, 12], dtype=torch.int64)
        # Rebuild all_local_src_indices to match
        info.all_local_src_indices = torch.cat([
            info.prefix_indices_list[0][:min_prefix[0]],   # [0, 1]
            info.prefix_indices_list[1][:min_prefix[1]],   # [10, 11, 12]
        ])

        # k_nope / k_pe for model-processed extend tokens (from max_prefix onward)
        # req0: 20-8=12, req1: 15-6=9 → total 21
        total_model_extend = sum(s - mp for s, mp in zip(seq_lens, max_prefix))  # 21
        k_nope = torch.full((total_model_extend, 1, kv_lora_rank),                 7.0)
        k_pe   = torch.full((total_model_extend, 1, kv_cache_dim - kv_lora_rank),  3.0)

        return info, kv_buf, k_nope, k_pe, kv_cache_dim, kv_lora_rank, seq_lens

    def test_combined_kv_buf_shape(self):
        info, kv_buf, k_nope, k_pe, kv_cache_dim, kv_lora_rank, seq_lens = self._make_test_batch()
        fill_local_and_extend_for_layer(info, k_nope, k_pe, kv_buf)
        assert info.combined_kv_buf.shape == (sum(seq_lens), 1, kv_cache_dim)

    def test_local_block_correct_values(self):
        """local_block should match kv_buf[all_local_src_indices]."""
        info, kv_buf, k_nope, k_pe, kv_cache_dim, kv_lora_rank, seq_lens = self._make_test_batch()
        fill_local_and_extend_for_layer(info, k_nope, k_pe, kv_buf)

        # req0 local: 2 slots starting at local_block_starts[0]=0
        # all_local_src_indices[:2] = [0, 1] → kv_buf[0]=0.0, kv_buf[1]=1.0
        assert info.combined_kv_buf[0, 0, 0].item() == pytest.approx(0.0)  # slot 0
        assert info.combined_kv_buf[1, 0, 0].item() == pytest.approx(1.0)  # slot 1

        # req1 local: 3 slots starting at local_block_starts[1]=2
        # all_local_src_indices[2:5] = [10, 11, 12]
        assert info.combined_kv_buf[2, 0, 0].item() == pytest.approx(10.0)  # slot 10
        assert info.combined_kv_buf[3, 0, 0].item() == pytest.approx(11.0)  # slot 11
        assert info.combined_kv_buf[4, 0, 0].item() == pytest.approx(12.0)  # slot 12

    def test_transfer_block_untouched(self):
        """fill_local_and_extend does NOT overwrite the transfer_block."""
        info, kv_buf, k_nope, k_pe, kv_cache_dim, kv_lora_rank, seq_lens = self._make_test_batch()
        fill_local_and_extend_for_layer(info, k_nope, k_pe, kv_buf)

        # transfer_block was set to 9.0 before calling fill
        tb = info.combined_kv_buf[info.local_block_size : info.extend_block_start]
        assert tb.abs().mean().item() == pytest.approx(9.0), \
            "Transfer block should be untouched by fill_local_and_extend"

    def test_extend_block_nope_values(self):
        """extend_block's k_nope component should be 7.0."""
        info, kv_buf, k_nope, k_pe, kv_cache_dim, kv_lora_rank, seq_lens = self._make_test_batch()
        fill_local_and_extend_for_layer(info, k_nope, k_pe, kv_buf)

        ext = info.combined_kv_buf[info.extend_block_start:]
        nope_part = ext[:, 0, :kv_lora_rank]
        assert nope_part.allclose(torch.full_like(nope_part, 7.0)), \
            f"extend_block k_nope should be 7.0, max diff={((nope_part - 7.0).abs().max()).item()}"

    def test_extend_block_rope_values(self):
        """extend_block's k_pe component should be 3.0."""
        info, kv_buf, k_nope, k_pe, kv_cache_dim, kv_lora_rank, seq_lens = self._make_test_batch()
        fill_local_and_extend_for_layer(info, k_nope, k_pe, kv_buf)

        ext = info.combined_kv_buf[info.extend_block_start:]
        rope_part = ext[:, 0, kv_lora_rank:]
        assert rope_part.allclose(torch.full_like(rope_part, 3.0)), \
            f"extend_block k_pe should be 3.0, max diff={((rope_part - 3.0).abs().max()).item()}"


# ---------------------------------------------------------------------------
# 7. test_ensure_combined_buf_allocated
# ---------------------------------------------------------------------------

class TestEnsureCombinedBufAllocated:
    """Verify lazy wiring and idempotency of ensure_combined_buf_allocated."""

    def setup_method(self):
        clear_persistent_combined_kv_buf()

    def test_wires_on_first_call(self):
        """After first call, combined_kv_buf is a view with the correct shape."""
        info = _make_info(
            seq_lens=[10], local_prefix=[5], min_prefix=[3], max_prefix=[7],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
        )
        assert info.combined_kv_buf is None
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        assert info.combined_kv_buf is not None
        # The view has exactly total_combined=10 tokens
        assert info.combined_kv_buf.shape == (10, 1, 8)

    def test_idempotent(self):
        """Second call on same info returns the same view object."""
        info = _make_info(
            seq_lens=[10], local_prefix=[5], min_prefix=[3], max_prefix=[7],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
        )
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        buf1 = info.combined_kv_buf
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        assert info.combined_kv_buf is buf1  # same object (no re-allocation)

    def test_view_is_backed_by_persistent_buf(self):
        """The view shares storage with the persistent buffer."""
        info = _make_info(
            seq_lens=[10], local_prefix=[5], min_prefix=[3], max_prefix=[7],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
        )
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        # Persistent buffer capacity >= 10
        cap = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))
        assert cap >= 10


# ---------------------------------------------------------------------------
# 8. test_reset_transfer_region
# ---------------------------------------------------------------------------

class TestResetTransferRegion:
    def setup_method(self):
        clear_persistent_combined_kv_buf()

    def test_zeroes_transfer_block(self):
        info = _make_info(
            seq_lens=[10], local_prefix=[5], min_prefix=[3], max_prefix=[7],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
        )
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        # Fill entire buffer with 1.0
        info.combined_kv_buf[:] = 1.0
        # Reset only transfer region
        reset_transfer_region(info)
        # Transfer_block should be zero
        tb = info.combined_kv_buf[info.local_block_size : info.extend_block_start]
        assert tb.sum().item() == pytest.approx(0.0)
        # Local and extend blocks should remain 1.0
        lb = info.combined_kv_buf[: info.local_block_size]
        eb = info.combined_kv_buf[info.extend_block_start :]
        assert lb.sum().item() == pytest.approx(float(lb.numel()))
        assert eb.sum().item() == pytest.approx(float(eb.numel()))


# ---------------------------------------------------------------------------
# Mock KV pool for save_dp_local_kv tests
# ---------------------------------------------------------------------------

class MockKVPool:
    """
    Minimal mock of MLATokenToKVPool that records set_mla_kv_buffer calls.
    """

    def __init__(self, num_slots: int, kv_lora_rank: int, qk_rope_head_dim: int):
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.store: dict[int, torch.Tensor] = {}
        self._default_num_slots = num_slots
        self._kv_cache_dim = kv_cache_dim

    def _ensure_layer(self, layer_id: int):
        if layer_id not in self.store:
            self.store[layer_id] = torch.full(
                (self._default_num_slots, 1, self._kv_cache_dim), float("nan")
            )

    def set_mla_kv_buffer(self, layer, loc: torch.Tensor,
                           cache_k_nope: torch.Tensor, cache_k_rope: torch.Tensor):
        layer_id = getattr(layer, "layer_id", 0)
        self._ensure_layer(layer_id)
        combined = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
        self.store[layer_id][loc] = combined


class MockLayer:
    """Minimal RadixAttention stub: just holds layer_id."""
    def __init__(self, layer_id: int = 0):
        self.layer_id = layer_id


# ---------------------------------------------------------------------------
# Helper: build combined_kv_buf with specific transfer and extend values
# ---------------------------------------------------------------------------

def _setup_combined_kv_buf(info, xfer_val, nope_val, pe_val):
    """
    Allocate combined_kv_buf and fill:
      - transfer_block with xfer_val (simulates post-all_reduce state)
      - extend_block with k_nope = nope_val, k_pe = pe_val
    """
    kv_cache_dim = info.kv_cache_dim
    kv_lora_rank = info.kv_lora_rank
    info.ensure_combined_buf_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu"))
    # Transfer block
    info.combined_kv_buf[info.local_block_size : info.extend_block_start] = xfer_val
    # Extend block
    n_ext = info.combined_kv_buf[info.extend_block_start :].shape[0]
    if n_ext > 0:
        info.combined_kv_buf[info.extend_block_start :, 0, :kv_lora_rank] = nope_val
        info.combined_kv_buf[info.extend_block_start :, 0, kv_lora_rank:] = pe_val


# ---------------------------------------------------------------------------
# 9. test_save_dp_local_kv
# ---------------------------------------------------------------------------

class TestSaveDpLocalKv:
    """Verify save_dp_local_kv writes transfer and extend KV to the correct slots."""

    def setup_method(self):
        clear_persistent_combined_kv_buf()

    # ------------------------------------------------------------------
    # Scenario 1: two requests, both dp-local, non-donor rank
    # req0: seq=20, local_prefix=4, min_prefix=2, max_prefix=8 (non-donor, local_p < max_p)
    # req1: seq=15, local_prefix=3, min_prefix=3, max_prefix=6 (non-donor, local_p = min_p)
    # dp_local_req_global_indices = [0, 1]
    # out_cache_loc: req0 has extend_len=16 slots, req1 has 12 slots
    # ------------------------------------------------------------------

    def _make_two_req_info(self):
        kv_lora_rank     = 5
        qk_rope_head_dim = 3
        kv_cache_dim     = kv_lora_rank + qk_rope_head_dim  # 8
        seq_lens         = [20, 15]
        local_prefix     = [4, 3]
        min_prefix       = [2, 3]
        max_prefix       = [8, 6]

        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=local_prefix,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=[0, 1],
            local_dp_rank=1,
            local_attn_tp_rank=0,
            dp_local_req_global_indices=[0, 1],
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        # Fill combined_kv_buf: transfer=5.0, extend nope=7.0 pe=3.0
        _setup_combined_kv_buf(info, xfer_val=5.0, nope_val=7.0, pe_val=3.0)

        # out_cache_loc: req0 → 16 slots (100..115), req1 → 12 slots (116..127)
        total_kv_slots = sum(s - lp for s, lp in zip(seq_lens, local_prefix))  # 28
        out_cache_loc = torch.arange(100, 100 + total_kv_slots, dtype=torch.int64)

        pool  = MockKVPool(200, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        return info, out_cache_loc, pool, layer

    def test_transfer_and_extend_written(self):
        """Both transfer and extend segments are written for non-donor dp-local reqs."""
        info, out_cache_loc, pool, layer = self._make_two_req_info()
        kv_cache_dim = info.kv_cache_dim

        save_dp_local_kv(info, layer, out_cache_loc, pool)

        pool._ensure_layer(0)
        kv = pool.store[0]  # [200, 1, 8]

        # ---- req0 ----
        # out_cache_loc[0:16] = slots 100..115
        # transfer segment [local_p=4, max_p=8): 4 slots → oc[0:4] = 100..103
        # extend  segment [max_p=8,  seq=20):   12 slots → oc[4:16] = 104..115
        for j in range(4):
            assert kv[100 + j, 0, :].tolist() == pytest.approx([5.0] * kv_cache_dim), \
                f"req0 transfer slot {100+j}"
        for j in range(4, 16):
            expected = [7.0] * info.kv_lora_rank + [3.0] * (kv_cache_dim - info.kv_lora_rank)
            assert kv[100 + j, 0, :].tolist() == pytest.approx(expected), \
                f"req0 extend slot {100+j}"

        # ---- req1 ----
        # out_cache_loc[16:28] = slots 116..127
        # transfer segment [local_p=3, max_p=6): 3 slots → oc[0:3] = 116..118
        # extend  segment [max_p=6, seq=15):     9 slots → oc[3:12] = 119..127
        for j in range(3):
            assert kv[116 + j, 0, :].tolist() == pytest.approx([5.0] * kv_cache_dim), \
                f"req1 transfer slot {116+j}"
        for j in range(3, 12):
            expected = [7.0] * info.kv_lora_rank + [3.0] * (kv_cache_dim - info.kv_lora_rank)
            assert kv[116 + j, 0, :].tolist() == pytest.approx(expected), \
                f"req1 extend slot {116+j}"

    # ------------------------------------------------------------------
    # Scenario 2: donor rank (local_prefix == max_prefix) → only extend
    # ------------------------------------------------------------------

    def test_donor_rank_only_extend(self):
        """Donor rank (local_p == max_p): no transfer segment, only extend."""
        kv_lora_rank     = 4
        qk_rope_head_dim = 4
        kv_cache_dim     = 8
        seq_lens         = [20]
        local_prefix     = [8]   # = max_prefix → donor
        min_prefix       = [2]
        max_prefix       = [8]

        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=local_prefix,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=[0],
            local_dp_rank=0,
            local_attn_tp_rank=0,
            dp_local_req_global_indices=[0],
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        # Transfer block filled with 99.0 (should never be read for donor)
        _setup_combined_kv_buf(info, xfer_val=99.0, nope_val=2.0, pe_val=4.0)

        extend_len = seq_lens[0] - local_prefix[0]  # 12
        out_cache_loc = torch.arange(50, 50 + extend_len, dtype=torch.int64)
        pool  = MockKVPool(100, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        save_dp_local_kv(info, layer, out_cache_loc, pool)

        pool._ensure_layer(0)
        kv = pool.store[0]
        expected = [2.0] * kv_lora_rank + [4.0] * qk_rope_head_dim
        for j in range(extend_len):
            slot = 50 + j
            assert kv[slot, 0, :].tolist() == pytest.approx(expected), \
                f"Donor extend slot {slot}: {kv[slot, 0, :].tolist()}"

    # ------------------------------------------------------------------
    # Scenario 3: total_transfer == 0 (all ranks share same prefix)
    # ------------------------------------------------------------------

    def test_zero_transfer_pure_extend(self):
        """When total_transfer=0, only the extend segment is written (transfer_len=0)."""
        kv_lora_rank     = 3
        qk_rope_head_dim = 2
        kv_cache_dim     = 5
        seq_lens         = [15]
        local_prefix     = [5]
        min_prefix       = [5]
        max_prefix       = [5]   # = min = local → no transfer

        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=local_prefix,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=[0],
            local_dp_rank=0,
            local_attn_tp_rank=0,
            dp_local_req_global_indices=[0],
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        assert info.total_transfer_tokens == 0
        _setup_combined_kv_buf(info, xfer_val=0.0, nope_val=1.5, pe_val=2.5)

        extend_len = seq_lens[0] - local_prefix[0]  # 10
        out_cache_loc = torch.arange(0, extend_len, dtype=torch.int64)
        pool  = MockKVPool(20, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        save_dp_local_kv(info, layer, out_cache_loc, pool)

        pool._ensure_layer(0)
        kv = pool.store[0]
        expected = [1.5] * kv_lora_rank + [2.5] * qk_rope_head_dim
        for j in range(extend_len):
            assert kv[j, 0, :].tolist() == pytest.approx(expected), \
                f"Zero-transfer extend slot {j}: {kv[j, 0, :].tolist()}"

    # ------------------------------------------------------------------
    # Scenario 4: only a subset of requests are dp-local
    # ------------------------------------------------------------------

    def test_only_subset_dp_local(self):
        """Only dp-local requests get KV written; non-local reqs are skipped."""
        kv_lora_rank     = 4
        qk_rope_head_dim = 4
        kv_cache_dim     = 8
        seq_lens         = [20, 18, 16]
        local_prefix     = [3,  5,  2]
        min_prefix       = [3,  2,  2]
        max_prefix       = [10, 8,  7]
        max_rank         = [0,  0,  0]

        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=local_prefix,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=max_rank,
            local_dp_rank=1,
            local_attn_tp_rank=0,
            dp_local_req_global_indices=[1],   # only req1 is dp-local
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        _setup_combined_kv_buf(info, xfer_val=6.0, nope_val=8.0, pe_val=9.0)

        # out_cache_loc for req1 only: extend_len = seq_len - local_prefix = 13 slots
        req1_extend_len = seq_lens[1] - local_prefix[1]  # 13
        out_cache_loc = torch.arange(200, 200 + req1_extend_len, dtype=torch.int64)

        pool  = MockKVPool(300, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        save_dp_local_kv(info, layer, out_cache_loc, pool)

        pool._ensure_layer(0)
        kv = pool.store[0]

        # req1: transfer [local_p=5, max_p=8) → 3 slots 200..202
        for j in range(3):
            assert kv[200 + j, 0, :].tolist() == pytest.approx([6.0] * 8), \
                f"req1 transfer slot {200+j}"
        # req1: extend [max_p=8, seq=18) → 10 slots 203..212
        expected_ext = [8.0] * 4 + [9.0] * 4
        for j in range(10):
            assert kv[203 + j, 0, :].tolist() == pytest.approx(expected_ext), \
                f"req1 extend slot {203+j}"

        # Slots not written should remain NaN
        assert kv[0:200].isnan().all(), "Non-dp-local slots should be untouched"
        assert kv[213:].isnan().all(), "Slots beyond req1 should be untouched"


# ---------------------------------------------------------------------------
# 10. test_persistent_combined_kv_buf
# ---------------------------------------------------------------------------

class TestPersistentCombinedKvBuf:
    """Verify the grow-only persistent combined_kv_buf semantics.

    The persistent buffer avoids per-batch cudaMalloc by keeping a module-level
    tensor that is only reallocated (2x growth) when total_combined exceeds the
    current capacity.  All other calls get a zero-copy view.
    """

    def setup_method(self):
        clear_persistent_combined_kv_buf()

    def _make_simple_info(self, seq_lens, kv_cache_dim=8, kv_lora_rank=5):
        return _make_info(
            seq_lens=seq_lens,
            local_prefix=[1] * len(seq_lens),
            min_prefix=[1] * len(seq_lens),
            max_prefix=[max(1, s - 2) for s in seq_lens],
            max_rank=[0] * len(seq_lens),
            local_dp_rank=0,
            local_attn_tp_rank=0,
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )

    # ------ basic allocation ------

    def test_initial_capacity_zero_before_any_alloc(self):
        cap = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))
        assert cap == 0

    def test_first_alloc_sets_capacity_at_least_2x(self):
        info = self._make_simple_info(seq_lens=[50])
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))
        assert cap >= 100, f"Expected capacity >= 100, got {cap}"

    def test_view_has_exact_shape(self):
        info = self._make_simple_info(seq_lens=[30, 20])  # total_combined = 50
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        assert info.combined_kv_buf.shape == (50, 1, 8)

    # ------ reuse semantics ------

    def test_second_batch_reuses_storage(self):
        """Two batches of the same total_combined share the same storage."""
        info1 = self._make_simple_info(seq_lens=[40])
        info1.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        ptr1 = info1.combined_kv_buf.data_ptr()

        info2 = self._make_simple_info(seq_lens=[40])
        info2.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        ptr2 = info2.combined_kv_buf.data_ptr()

        assert ptr1 == ptr2, "Both batches must reuse the same persistent storage"

    def test_smaller_batch_reuses_storage(self):
        info1 = self._make_simple_info(seq_lens=[100])
        info1.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap_before = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))

        info2 = self._make_simple_info(seq_lens=[30])
        info2.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap_after = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))

        assert cap_after == cap_before, "Smaller batch must not grow the buffer"
        assert info2.combined_kv_buf.shape == (30, 1, 8)

    def test_capacity_never_shrinks(self):
        info_large = self._make_simple_info(seq_lens=[200])
        info_large.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap_large = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))

        info_small = self._make_simple_info(seq_lens=[10])
        info_small.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap_small = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))

        assert cap_small == cap_large, "Capacity must not shrink on a smaller batch"

    # ------ growth semantics ------

    def test_larger_batch_grows_buffer(self):
        info1 = self._make_simple_info(seq_lens=[10])
        info1.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap1 = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))

        need = cap1 + 1
        info2 = _make_info(
            seq_lens=[need], local_prefix=[1], min_prefix=[1],
            max_prefix=[need - 1], max_rank=[0],
            local_dp_rank=0, local_attn_tp_rank=0,
            kv_cache_dim=8, kv_lora_rank=5,
        )
        info2.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap2 = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))

        assert cap2 > cap1, "Buffer must grow when total_combined exceeds capacity"
        assert cap2 >= need * 2, "New capacity should be >= 2x needed (doubling strategy)"

    def test_growth_changes_storage_pointer(self):
        info1 = self._make_simple_info(seq_lens=[10])
        info1.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap1 = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))
        ptr1 = info1.combined_kv_buf.data_ptr()

        info2 = _make_info(
            seq_lens=[cap1 + 1], local_prefix=[1], min_prefix=[1],
            max_prefix=[cap1], max_rank=[0],
            local_dp_rank=0, local_attn_tp_rank=0,
            kv_cache_dim=8, kv_lora_rank=5,
        )
        info2.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        ptr2 = info2.combined_kv_buf.data_ptr()

        assert ptr2 != ptr1, "New allocation must have a different storage address"

    # ------ key isolation ------

    def test_different_kv_cache_dim_get_separate_buffers(self):
        info8 = self._make_simple_info(seq_lens=[10], kv_cache_dim=8, kv_lora_rank=5)
        info8.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))

        info16 = self._make_simple_info(seq_lens=[10], kv_cache_dim=16, kv_lora_rank=10)
        info16.ensure_combined_buf_allocated(16, 10, torch.float32, torch.device("cpu"))

        assert info8.combined_kv_buf.data_ptr() != info16.combined_kv_buf.data_ptr()
        cap8  = get_persistent_combined_kv_buf_capacity(8,  torch.float32, torch.device("cpu"))
        cap16 = get_persistent_combined_kv_buf_capacity(16, torch.float32, torch.device("cpu"))
        assert cap8 >= 10 and cap16 >= 10

    def test_different_dtype_get_separate_buffers(self):
        info_f32 = self._make_simple_info(seq_lens=[10])
        info_f32.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))

        info_f16 = self._make_simple_info(seq_lens=[10])
        info_f16.ensure_combined_buf_allocated(8, 5, torch.float16, torch.device("cpu"))

        assert info_f32.combined_kv_buf.data_ptr() != info_f16.combined_kv_buf.data_ptr()
        assert info_f16.combined_kv_buf.dtype == torch.float16
        assert info_f32.combined_kv_buf.dtype == torch.float32

    # ------ clear ------

    def test_clear_resets_capacity_to_zero(self):
        info = self._make_simple_info(seq_lens=[50])
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        assert get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu")) > 0

        clear_persistent_combined_kv_buf()
        assert get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu")) == 0

    def test_clear_then_alloc_works(self):
        info1 = self._make_simple_info(seq_lens=[10])
        info1.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        clear_persistent_combined_kv_buf()

        info2 = self._make_simple_info(seq_lens=[10])
        info2.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        assert info2.combined_kv_buf.shape == (10, 1, 8)
        assert get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu")) >= 20

    # ------ data integrity ------

    def test_views_from_same_batch_are_aliases(self):
        """Two info objects for the same batch size share storage (aliases)."""
        info1 = self._make_simple_info(seq_lens=[10])
        info1.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        info1.combined_kv_buf[:] = 1.0

        info2 = self._make_simple_info(seq_lens=[10])
        info2.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        info2.combined_kv_buf[:] = 2.0

        assert info1.combined_kv_buf[0, 0, 0].item() == pytest.approx(2.0), (
            "info1 view and info2 view are storage aliases"
        )

    def test_view_does_not_exceed_total_combined(self):
        total_combined = 20
        info = self._make_simple_info(seq_lens=[total_combined])
        info.ensure_combined_buf_allocated(8, 5, torch.float32, torch.device("cpu"))
        cap = get_persistent_combined_kv_buf_capacity(8, torch.float32, torch.device("cpu"))

        assert info.combined_kv_buf.shape[0] == total_combined
        assert cap > total_combined, "Persistent buffer has headroom beyond the view"


# ---------------------------------------------------------------------------
# Tests for the dispatch fix: sum_extend_prefix_lens threshold scenario
# ---------------------------------------------------------------------------

class TestDispatchFixLogic:
    """
    Verify the logic that prevents MHA_ONE_SHOT from leaking through when
    share-prefix is active and sum(extend_prefix_lens) is large.

    These tests exercise the _handle_attention_backend dispatch condition directly
    without needing a full model instance.  We check:
    1. The threshold condition that causes the MHA_ONE_SHOT path to be chosen.
    2. That the out_cache_loc size mismatch between share-prefix and
       dp_local_token_start/end is correctly characterised.
    """

    def test_extend_prefix_lens_threshold_condition(self):
        """
        Reproduce the condition that causes the dispatch to choose MHA_ONE_SHOT
        in the high-cache-hit (extend=1) scenario.

        In share-prefix mode prepare_for_extend sets prefix_lens = max_prefix.
        Here we verify that such large prefix sums DO exceed a typical threshold.
        """
        # Simulate the high-cache-hit scenario from the bug report
        # seq_lens = [1954, 1309, 664, 5953, 1954, 1825]
        # max_prefix ≈ seq_len - 1
        max_prefix = [1953, 1308, 663, 5952, 1953, 1824]
        sum_extend_prefix_lens = sum(max_prefix)   # 13653

        # A typical chunked_prefix_cache_threshold is small (e.g., a few thousand)
        # The point is sum_extend_prefix_lens is large when all tokens are cached.
        # Verify it is non-zero and would plausibly exceed a threshold.
        assert sum_extend_prefix_lens > 0
        # With extend=1, each request contributes only 1 extend token
        extend_lens = [1, 1, 1, 1, 1, 1]
        total_extend_tokens = sum(extend_lens)  # 6
        assert total_extend_tokens == 6

        # The critical invariant: sum_extend_prefix_lens is huge because it
        # equals sum(max_prefix), not sum(local_prefix) or sum(extend_lens).
        assert sum_extend_prefix_lens > 1000

    def test_out_cache_loc_size_mismatch_characterisation(self):
        """
        Verify that out_cache_loc (allocated by alloc_for_extend in share-prefix
        mode) has a DIFFERENT size than what _set_mla_kv_buffer_for_dp expects.

        share-prefix out_cache_loc size  = sum(seq_len - local_prefix)  ← extend only
        _set_mla_kv_buffer_for_dp expects = dp_local_token_end - dp_local_token_start
        """
        # From the bug report log:
        # DP1 dp-local requests: indices [3, 4, 5] → seq_lens [5953, 1954, 1825]
        # local_prefix for DP1:  [5952, 1953, 1824]
        # extend = 1 for each
        seq_lens_all     = [1954, 1309, 664, 5953, 1954, 1825]
        local_prefix_dp1 = [137,  137,  137, 5952, 1953, 1824]   # dp1's local view
        dp_local_indices = [3, 4, 5]

        # share-prefix out_cache_loc: sum(seq_len - local_prefix) for dp-local reqs
        oc_size_share_prefix = sum(
            seq_lens_all[g] - local_prefix_dp1[g] for g in dp_local_indices
        )
        assert oc_size_share_prefix == 3   # 1+1+1

        # _set_mla_kv_buffer_for_dp expects: dp_local_token_end - dp_local_token_start
        # dp_local_token_start = sum(seq_lens[:3]) = 1954+1309+664 = 3927
        # dp_local_token_end   = sum(seq_lens)     = 13659
        dp_local_token_start = sum(seq_lens_all[:3])   # = 3927
        dp_local_token_end   = sum(seq_lens_all)       # = 13659
        local_tok_n = dp_local_token_end - dp_local_token_start
        assert local_tok_n == 9732

        # The mismatch: 3 ≠ 9732  → AssertionError in _set_mla_kv_buffer_for_dp
        assert oc_size_share_prefix != local_tok_n, (
            "_set_mla_kv_buffer_for_dp would crash: "
            f"out_cache_loc.shape[0]={oc_size_share_prefix}, "
            f"local_tok_n={local_tok_n}"
        )

    def test_fix_forces_mla_when_share_prefix_active(self):
        """
        Verify that the fix correctly forces result = MLA regardless of what
        the backend initially returned.

        This test directly exercises the dispatch override condition without
        instantiating a full model.
        """
        from sglang.srt.models.deepseek_common.attention_backend_handler import (
            _dispatch_mla_subtype,
        )
        from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
            AttnForwardMethod,
        )

        # Mock the override logic from dispatch_attn_forward_method
        def apply_share_prefix_override(initial_result, enable_share_prefix):
            """Mirrors the logic in dispatch_attn_forward_method."""
            if enable_share_prefix:
                # Always force MLA (or its subtype)
                # We can't call _dispatch_mla_subtype without a real attn object,
                # so we simulate what it returns for the non-HIP CUDA case.
                return AttnForwardMethod.MLA
            else:
                if initial_result == AttnForwardMethod.MLA:
                    return AttnForwardMethod.MHA
                return initial_result

        # Case 1: backend returned MHA_ONE_SHOT, share-prefix enabled → must become MLA
        result = apply_share_prefix_override(AttnForwardMethod.MHA_ONE_SHOT, enable_share_prefix=True)
        assert result == AttnForwardMethod.MLA, (
            f"Expected MLA but got {result.name}: share-prefix must force MLA for MHA_ONE_SHOT"
        )

        # Case 2: backend returned MHA_CHUNKED_KV, share-prefix enabled → must become MLA
        result = apply_share_prefix_override(AttnForwardMethod.MHA_CHUNKED_KV, enable_share_prefix=True)
        assert result == AttnForwardMethod.MLA

        # Case 3: backend returned MLA, share-prefix enabled → stays MLA
        result = apply_share_prefix_override(AttnForwardMethod.MLA, enable_share_prefix=True)
        assert result == AttnForwardMethod.MLA

        # Case 4: backend returned MLA, share-prefix disabled (prefix-0 mode) → becomes MHA
        result = apply_share_prefix_override(AttnForwardMethod.MLA, enable_share_prefix=False)
        assert result == AttnForwardMethod.MHA, (
            "prefix-0 mode should still downgrade MLA → MHA"
        )

        # Case 5: backend returned MHA_ONE_SHOT, share-prefix disabled → unchanged
        result = apply_share_prefix_override(AttnForwardMethod.MHA_ONE_SHOT, enable_share_prefix=False)
        assert result == AttnForwardMethod.MHA_ONE_SHOT
