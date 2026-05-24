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

# ---------------------------------------------------------------------------
# Helpers to build mock SharePrefixBatchInfo directly (no MPI needed)
# ---------------------------------------------------------------------------

from sglang.srt.multiplex.share_prefix_helper import (
    SharePrefixBatchInfo,
    build_combined_kv_for_layer,
    fill_transfer_buffer_for_layer,
    reset_transfer_buffer,
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
    # In reality these would be KV pool slot indices
    prefix_indices_list = []
    slot = 0
    for i in range(n):
        n_slots = local_prefix[i]
        pi = torch.arange(slot, slot + n_slots, dtype=torch.int64)
        prefix_indices_list.append(pi)
        slot += n_slots

    if dp_local_req_global_indices is None:
        dp_local_req_global_indices = list(range(n))

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
        extend_hidden_starts=list(extend_hidden_starts),
        prefix_indices_list=prefix_indices_list,
        dp_local_req_global_indices=list(dp_local_req_global_indices),
        transfer_buffer=None,
        kv_cache_dim=kv_cache_dim,
        kv_lora_rank=kv_lora_rank,
        device=torch.device(device),
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
        # This is NOT used directly; extend for attention Q is seq_len - local_prefix
        _ = extend_lens

    def test_uniform_prefix_gives_zero_transfer(self):
        """All ranks have the same prefix → transfer_len == 0 → return None."""
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
# 3. test_fill_transfer_buffer_logic
# ---------------------------------------------------------------------------

class TestFillTransferBuffer:
    """Verify fill_transfer_buffer_for_layer fills the right slots."""

    def _setup(self, local_dp_rank, local_attn_tp_rank):
        """
        Batch: 2 requests
          req0: seq=60, local_prefix=40, min_prefix=10, max_prefix=40, max_rank=0
          req1: seq=60, local_prefix=10, min_prefix=10, max_prefix=20, max_rank=1
        local_dp_rank=0, local_attn_tp_rank=0  → should fill req0 only
        """
        kv_cache_dim = 4
        kv_lora_rank = 3
        device = "cpu"

        info = _make_info(
            seq_lens=[60, 60],
            local_prefix=[40, 10],
            min_prefix=[10, 10],
            max_prefix=[40, 20],
            max_rank=[0, 1],
            local_dp_rank=local_dp_rank,
            local_attn_tp_rank=local_attn_tp_rank,
            device=device,
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        info.ensure_transfer_buffer_allocated(
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
            dtype=torch.float32,
            device=torch.device(device),
        )
        reset_transfer_buffer(info)

        # KV pool: 100 slots, each [1, kv_cache_dim]; fill with slot_idx+1 for easy checking
        total_slots = 100
        kv_buf = torch.zeros(total_slots, 1, kv_cache_dim, dtype=torch.float32)
        for slot in range(total_slots):
            kv_buf[slot, 0, :] = float(slot + 1)

        # Override prefix_indices_list so slot[j] = j for both reqs
        # req0 has 40 prefix slots: [0, 1, ..., 39]
        # req1 has 10 prefix slots: [40, 41, ..., 49]
        info.prefix_indices_list[0] = torch.arange(0, 40, dtype=torch.int64)
        info.prefix_indices_list[1] = torch.arange(40, 50, dtype=torch.int64)

        return info, kv_buf

    def test_rank0_tp0_fills_req0(self):
        """DP rank 0, attn_tp_rank 0: should fill req0's transfer region [10:40]."""
        info, kv_buf = self._setup(local_dp_rank=0, local_attn_tp_rank=0)
        layer_id = 0
        fill_transfer_buffer_for_layer(info, layer_id, kv_buf)

        # req0 transfer: slots [10, 40) → 30 tokens
        # transfer_buffer[0:30] should be filled
        # slot 10 → value 11, slot 11 → 12, ..., slot 39 → 40
        tb = info.transfer_buffer
        # total_transfer = req0_transfer(30) + req1_transfer(10) = 40
        assert tb.shape == (40, 1, 4)
        # We only fill req0 (max_rank=0 == local_dp_rank=0); req1 (max_rank=1) stays 0
        for j in range(30):
            expected_slot = 10 + j  # slot index in kv_buf
            expected_val = float(expected_slot + 1)
            assert tb[j, 0, 0].item() == pytest.approx(expected_val), \
                f"transfer_buffer[{j}] expected {expected_val}, got {tb[j,0,0].item()}"
        # req1's transfer portion (offset=30, len=10) must remain zero
        assert tb[30:40].abs().sum().item() == pytest.approx(0.0)

    def test_rank0_tp1_skips(self):
        """attn_tp_rank != 0: no filling (leaves zeros)."""
        info, kv_buf = self._setup(local_dp_rank=0, local_attn_tp_rank=1)
        layer_id = 0
        fill_transfer_buffer_for_layer(info, layer_id, kv_buf)
        assert info.transfer_buffer.sum().item() == pytest.approx(0.0)

    def test_rank1_tp0_fills_req1_only(self):
        """DP rank 1, attn_tp_rank 0: should fill req1 transfer [10:20], not req0."""
        kv_cache_dim = 4
        kv_lora_rank = 3
        device = "cpu"

        info = _make_info(
            seq_lens=[60, 60],
            local_prefix=[10, 20],   # rank1 perspective
            min_prefix=[10, 10],
            max_prefix=[40, 20],
            max_rank=[0, 1],
            local_dp_rank=1,
            local_attn_tp_rank=0,
            device=device,
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        info.ensure_transfer_buffer_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device(device))
        reset_transfer_buffer(info)

        kv_buf = torch.zeros(100, 1, kv_cache_dim, dtype=torch.float32)
        for s in range(100):
            kv_buf[s, 0, :] = float(s + 1)

        # req0: local_prefix=10 → pref_idx has 10 slots [0..9] but max_rank=0 ≠ local_dp_rank=1 → skip
        # req1: local_prefix=20 → pref_idx [10..29], transfer region = slots [10:20] (min_p=10, max_p=20)
        info.prefix_indices_list[0] = torch.arange(0, 10, dtype=torch.int64)
        info.prefix_indices_list[1] = torch.arange(10, 30, dtype=torch.int64)

        fill_transfer_buffer_for_layer(info, 0, kv_buf)

        # req0 transfer (offset=0, len=30) should be zeros (not filled on rank1)
        tb = info.transfer_buffer
        assert tb[0:30].abs().sum().item() == pytest.approx(0.0), "req0 transfer should be 0 on rank1"

        # req1 transfer (offset=30, len=10) should be filled
        # slots info.prefix_indices_list[1][10:20] = [20, ..., 29]
        for j in range(10):
            expected_slot = 20 + j  # index in prefix_indices_list[1][10+j]
            expected_val = float(expected_slot + 1)
            assert tb[30 + j, 0, 0].item() == pytest.approx(expected_val), \
                f"transfer_buffer[{30+j}] expected {expected_val}"


# ---------------------------------------------------------------------------
# 4. test_build_combined_kv_shape
# ---------------------------------------------------------------------------

class TestBuildCombinedKv:
    """Verify combined KV shape and cu_seqlens_k correctness."""

    def _make_test_batch(self):
        """
        2 requests, dp_rank=1, attn_tp_rank=0
          req0: seq=20, local_prefix=4, min_prefix=2, max_prefix=8
          req1: seq=15, local_prefix=3, min_prefix=3, max_prefix=6
        """
        kv_cache_dim = 8
        kv_lora_rank = 5
        device = "cpu"
        n = 2
        seq_lens    = [20, 15]
        local_prefix = [4, 3]
        min_prefix   = [2, 3]
        max_prefix   = [8, 6]
        max_rank     = [0, 1]

        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=local_prefix,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=max_rank,
            local_dp_rank=1,
            local_attn_tp_rank=0,
            device=device,
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        info.ensure_transfer_buffer_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device(device))
        # Assign some test transfer data
        info.transfer_buffer[:] = 9.0

        # kv_buf: many slots, each [1, kv_cache_dim], value = slot_index
        total_slots = 100
        kv_buf = torch.zeros(total_slots, 1, kv_cache_dim, dtype=torch.float32)
        for s in range(total_slots):
            kv_buf[s, 0, :] = float(s)

        # prefix_indices for req0 (4 slots): [0,1,2,3]
        # prefix_indices for req1 (3 slots): [10,11,12]
        info.prefix_indices_list[0] = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        info.prefix_indices_list[1] = torch.tensor([10, 11, 12], dtype=torch.int64)

        # k_nope / k_pe for model-processed extend tokens (input_ids from max_prefix onward)
        # req0 extend: max_prefix=8,  seq=20 → 12 extend tokens (positions 8..19)
        # req1 extend: max_prefix=6,  seq=15 →  9 extend tokens (positions 6..14)
        # total = 21 (NOT sum(seq-local_prefix)=28; that larger count was the old bug)
        total_model_extend = sum(s - mp for s, mp in zip(seq_lens, max_prefix))  # 12 + 9 = 21
        k_nope = torch.ones(total_model_extend, 1, kv_lora_rank, dtype=torch.float32) * 7.0
        k_pe   = torch.ones(total_model_extend, 1, kv_cache_dim - kv_lora_rank, dtype=torch.float32) * 3.0

        return info, kv_buf, k_nope, k_pe, kv_cache_dim, seq_lens

    def test_combined_kv_shape(self):
        info, kv_buf, k_nope, k_pe, kv_cache_dim, seq_lens = self._make_test_batch()
        combined_kv, page_table, cache_seqlens = build_combined_kv_for_layer(
            info, layer_id=0, k_nope=k_nope, k_pe=k_pe, kv_buf=kv_buf
        )
        total_expected = sum(seq_lens)  # 20 + 15 = 35
        assert combined_kv.shape == (total_expected, 1, kv_cache_dim), \
            f"Expected ({total_expected}, 1, {kv_cache_dim}), got {combined_kv.shape}"

    def test_page_table_shape(self):
        info, kv_buf, k_nope, k_pe, _, seq_lens = self._make_test_batch()
        _, page_table, _ = build_combined_kv_for_layer(
            info, layer_id=0, k_nope=k_nope, k_pe=k_pe, kv_buf=kv_buf
        )
        n = len(seq_lens)
        max_seq = max(seq_lens)
        assert page_table.shape == (n, max_seq), \
            f"Expected ({n}, {max_seq}), got {page_table.shape}"
        assert page_table.dtype == torch.int32

    def test_cache_seqlens_values(self):
        info, kv_buf, k_nope, k_pe, _, seq_lens = self._make_test_batch()
        _, _, cache_seqlens = build_combined_kv_for_layer(
            info, layer_id=0, k_nope=k_nope, k_pe=k_pe, kv_buf=kv_buf
        )
        assert cache_seqlens.tolist() == seq_lens, \
            f"cache_seqlens should equal seq_lens, got {cache_seqlens.tolist()}"
        assert cache_seqlens.dtype == torch.int32

    def test_page_table_contiguous_slots(self):
        """Each request's slots in page_table should be consecutive."""
        info, kv_buf, k_nope, k_pe, _, seq_lens = self._make_test_batch()
        _, page_table, _ = build_combined_kv_for_layer(
            info, layer_id=0, k_nope=k_nope, k_pe=k_pe, kv_buf=kv_buf
        )
        # req0 starts at 0
        for j in range(seq_lens[0]):
            assert page_table[0, j].item() == j
        # req1 starts at seq_lens[0]
        start1 = seq_lens[0]
        for j in range(seq_lens[1]):
            assert page_table[1, j].item() == start1 + j

    def test_combined_kv_part_a_local_prefix(self):
        """Part A (local prefix) of combined_kv should match kv_buf[prefix_indices[0:min_prefix]]."""
        info, kv_buf, k_nope, k_pe, kv_cache_dim, seq_lens = self._make_test_batch()
        combined_kv, _, _ = build_combined_kv_for_layer(
            info, layer_id=0, k_nope=k_nope, k_pe=k_pe, kv_buf=kv_buf
        )
        # req0: Part A = combined_kv[0:2] (min_prefix=2)
        # prefix_indices[0] = [0,1,2,3]; slots [0:2] = [0,1]
        # kv_buf[0] = [0.0]*8,  kv_buf[1] = [1.0]*8
        for j in range(info.min_prefix[0]):
            slot = info.prefix_indices_list[0][j].item()
            expected = kv_buf[slot, 0, :].tolist()
            actual = combined_kv[j, 0, :].tolist()
            assert actual == pytest.approx(expected), \
                f"Part A req0 slot {j}: expected {expected}, got {actual}"

    def test_combined_kv_part_b_transfer(self):
        """Part B (transfer) should match transfer_buffer data."""
        info, kv_buf, k_nope, k_pe, kv_cache_dim, seq_lens = self._make_test_batch()
        # transfer_buffer was set to 9.0
        combined_kv, _, _ = build_combined_kv_for_layer(
            info, layer_id=0, k_nope=k_nope, k_pe=k_pe, kv_buf=kv_buf
        )
        min_p0 = info.min_prefix[0]  # 2
        max_p0 = info.max_prefix[0]  # 8
        for j in range(max_p0 - min_p0):
            actual = combined_kv[min_p0 + j, 0, :].tolist()
            assert actual == pytest.approx([9.0] * kv_cache_dim), \
                f"Part B req0 token {j}: expected 9.0, got {actual}"

    def test_combined_kv_part_c_extend(self):
        """Part C (extend) should match k_nope / k_pe for true extend tokens."""
        info, kv_buf, k_nope, k_pe, kv_cache_dim, seq_lens = self._make_test_batch()
        kv_lora_rank = info.kv_lora_rank
        combined_kv, _, _ = build_combined_kv_for_layer(
            info, layer_id=0, k_nope=k_nope, k_pe=k_pe, kv_buf=kv_buf
        )
        # req0: Part C = combined_kv[8:20]  (max_prefix=8, seq=20)
        # k_nope was 7.0, k_pe was 3.0
        max_p0 = info.max_prefix[0]
        seq0   = info.seq_lens[0]
        for j in range(seq0 - max_p0):
            nope_part = combined_kv[max_p0 + j, 0, :kv_lora_rank].tolist()
            rope_part = combined_kv[max_p0 + j, 0, kv_lora_rank:].tolist()
            assert nope_part == pytest.approx([7.0] * kv_lora_rank), \
                f"Part C req0 extend[{j}] nope: expected 7.0, got {nope_part}"
            assert rope_part == pytest.approx([3.0] * (kv_cache_dim - kv_lora_rank)), \
                f"Part C req0 extend[{j}] rope: expected 3.0, got {rope_part}"


# ---------------------------------------------------------------------------
# 5. test_ensure_transfer_buffer_allocated
# ---------------------------------------------------------------------------

class TestEnsureTransferBufferAllocated:
    """Verify lazy allocation and idempotency."""

    def test_allocates_on_first_call(self):
        info = _make_info(
            seq_lens=[10], local_prefix=[5], min_prefix=[3], max_prefix=[7],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
        )
        assert info.transfer_buffer is None
        info.ensure_transfer_buffer_allocated(8, 5, torch.float32, torch.device("cpu"))
        assert info.transfer_buffer is not None
        assert info.transfer_buffer.shape == (4, 1, 8)  # total_transfer = 4

    def test_idempotent(self):
        info = _make_info(
            seq_lens=[10], local_prefix=[5], min_prefix=[3], max_prefix=[7],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
        )
        info.ensure_transfer_buffer_allocated(8, 5, torch.float32, torch.device("cpu"))
        buf1 = info.transfer_buffer
        info.ensure_transfer_buffer_allocated(8, 5, torch.float32, torch.device("cpu"))
        assert info.transfer_buffer is buf1  # same object


# ---------------------------------------------------------------------------
# 6. test_reset_transfer_buffer
# ---------------------------------------------------------------------------

class TestResetTransferBuffer:
    def test_zeroes_buffer(self):
        info = _make_info(
            seq_lens=[10], local_prefix=[5], min_prefix=[3], max_prefix=[7],
            max_rank=[0], local_dp_rank=0, local_attn_tp_rank=0,
        )
        info.ensure_transfer_buffer_allocated(8, 5, torch.float32, torch.device("cpu"))
        info.transfer_buffer[:] = 1.0
        reset_transfer_buffer(info)
        assert info.transfer_buffer.sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Mock KV pool for save_dp_local_kv tests
# ---------------------------------------------------------------------------

class MockKVPool:
    """
    Minimal mock of MLATokenToKVPool that records set_mla_kv_buffer calls.

    kv_store[layer_id][slot_idx] = [kv_lora_rank + qk_rope_head_dim] tensor.
    """

    def __init__(self, num_slots: int, kv_lora_rank: int, qk_rope_head_dim: int):
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        # layer_id → [num_slots, 1, kv_cache_dim] tensor
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
        combined = torch.cat([cache_k_nope, cache_k_rope], dim=-1)  # [n, 1, kv_cache_dim]
        self.store[layer_id][loc] = combined


class MockLayer:
    """Minimal RadixAttention stub: just holds layer_id."""
    def __init__(self, layer_id: int = 0):
        self.layer_id = layer_id


# ---------------------------------------------------------------------------
# 7. test_save_dp_local_kv
# ---------------------------------------------------------------------------

class TestSaveDpLocalKv:
    """Verify save_dp_local_kv writes transfer and extend KV to the correct slots."""

    # ------------------------------------------------------------------
    # Scenario 1: two requests, both dp-local, non-donor rank
    # req0: seq=20, local_prefix=4, min_prefix=2, max_prefix=8 (non-donor, local_p < max_p)
    # req1: seq=15, local_prefix=3, min_prefix=3, max_prefix=6 (non-donor, local_p = min_p)
    # dp_local_req_global_indices = [0, 1]
    # out_cache_loc: req0 has extend_len=16 slots, req1 has 12 slots
    # ------------------------------------------------------------------

    def _make_two_req_info(self) -> tuple:
        kv_lora_rank    = 5
        qk_rope_head_dim = 3
        kv_cache_dim    = kv_lora_rank + qk_rope_head_dim  # 8
        seq_lens        = [20, 15]
        local_prefix    = [4, 3]
        min_prefix      = [2, 3]
        max_prefix      = [8, 6]

        info = _make_info(
            seq_lens=seq_lens,
            local_prefix=local_prefix,
            min_prefix=min_prefix,
            max_prefix=max_prefix,
            max_rank=[0, 1],
            local_dp_rank=1,
            local_attn_tp_rank=0,
            dp_local_req_global_indices=[0, 1],  # both are dp-local
            kv_cache_dim=kv_cache_dim,
            kv_lora_rank=kv_lora_rank,
        )
        info.ensure_transfer_buffer_allocated(
            kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu")
        )
        # Fill transfer buffer with a distinct constant (5.0) so we can verify
        info.transfer_buffer[:] = 5.0

        # k_nope / k_pe: model processes only (seq_len - max_prefix) tokens per request
        # req0: max_prefix=8,  seq=20 → 12 model tokens
        # req1: max_prefix=6,  seq=15 →  9 model tokens  → total 21
        total_model_extend = sum(s - mp for s, mp in zip(seq_lens, max_prefix))  # 21
        k_nope_val = 7.0
        k_pe_val   = 3.0
        k_nope = torch.full((total_model_extend, 1, kv_lora_rank),    k_nope_val)
        k_pe   = torch.full((total_model_extend, 1, qk_rope_head_dim), k_pe_val)

        # out_cache_loc: KV-pool slots = seq_len - local_prefix (covers transfer + extend regions)
        # req0: 16 kv slots, req1: 12 kv slots → total 28 (numbered 100..127)
        total_kv_slots = sum(s - lp for s, lp in zip(seq_lens, local_prefix))  # 28
        out_cache_loc = torch.arange(100, 100 + total_kv_slots, dtype=torch.int64)

        pool = MockKVPool(200, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        return info, k_nope, k_pe, out_cache_loc, pool, layer

    def test_transfer_and_extend_written(self):
        """Both transfer and extend segments are written for non-donor dp-local reqs."""
        info, k_nope, k_pe, out_cache_loc, pool, layer = self._make_two_req_info()

        save_dp_local_kv(info, layer, k_nope, k_pe, out_cache_loc, pool)

        pool._ensure_layer(0)
        kv = pool.store[0]  # [200, 1, 8]

        # ---- req0 ----
        # out_cache_loc for req0: slots 100..115 (extend_len=16)
        # transfer segment [local_p=4, max_p=8): 4 slots → oc[0:4] = slots 100..103
        # extend  segment [max_p=8,  seq=20):   12 slots → oc[4:16] = slots 104..115
        for j in range(4):
            slot = 100 + j
            # transfer KV = 5.0 for all dims
            assert kv[slot, 0, :].tolist() == pytest.approx([5.0] * 8), \
                f"req0 transfer slot {slot}: {kv[slot,0,:].tolist()}"
        for j in range(4, 16):
            slot = 100 + j
            expected = [7.0] * 5 + [3.0] * 3
            assert kv[slot, 0, :].tolist() == pytest.approx(expected), \
                f"req0 extend slot {slot}: {kv[slot,0,:].tolist()}"

        # ---- req1 ----
        # out_cache_loc for req1: slots 116..127 (extend_len=12)
        # transfer segment [local_p=3, max_p=6): 3 slots → oc[0:3] = slots 116..118
        # extend  segment [max_p=6, seq=15):     9 slots → oc[3:12] = slots 119..127
        for j in range(3):
            slot = 116 + j
            assert kv[slot, 0, :].tolist() == pytest.approx([5.0] * 8), \
                f"req1 transfer slot {slot}: {kv[slot,0,:].tolist()}"
        for j in range(3, 12):
            slot = 116 + j
            expected = [7.0] * 5 + [3.0] * 3
            assert kv[slot, 0, :].tolist() == pytest.approx(expected), \
                f"req1 extend slot {slot}: {kv[slot,0,:].tolist()}"

    # ------------------------------------------------------------------
    # Scenario 2: donor rank (local_prefix == max_prefix) → only extend
    # req0: seq=20, local_prefix=8, min_prefix=2, max_prefix=8 (donor)
    # extend_len = 12 (positions [8, 20))
    # ------------------------------------------------------------------

    def test_donor_rank_only_extend(self):
        """Donor rank (local_p == max_p): no transfer segment, only extend."""
        kv_lora_rank    = 4
        qk_rope_head_dim = 4
        kv_cache_dim    = 8
        seq_lens        = [20]
        local_prefix    = [8]   # = max_prefix → donor
        min_prefix      = [2]
        max_prefix      = [8]

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
        info.ensure_transfer_buffer_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu"))
        info.transfer_buffer[:] = 99.0  # should never be read for the donor

        extend_len = seq_lens[0] - local_prefix[0]  # 12
        k_nope = torch.full((extend_len, 1, kv_lora_rank),    2.0)
        k_pe   = torch.full((extend_len, 1, qk_rope_head_dim), 4.0)
        out_cache_loc = torch.arange(50, 50 + extend_len, dtype=torch.int64)

        pool  = MockKVPool(100, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        save_dp_local_kv(info, layer, k_nope, k_pe, out_cache_loc, pool)

        pool._ensure_layer(0)
        kv = pool.store[0]
        expected = [2.0] * kv_lora_rank + [4.0] * qk_rope_head_dim
        for j in range(extend_len):
            slot = 50 + j
            assert kv[slot, 0, :].tolist() == pytest.approx(expected), \
                f"Donor extend slot {slot}: {kv[slot,0,:].tolist()}"

    # ------------------------------------------------------------------
    # Scenario 3: total_transfer == 0 (all ranks share same prefix)
    # Equivalent to normal extend-only write.
    # req0: seq=15, local_prefix=5, min_prefix=5, max_prefix=5
    # extend_len = 10
    # ------------------------------------------------------------------

    def test_zero_transfer_pure_extend(self):
        """When total_transfer=0, only the extend segment is written (transfer_len=0)."""
        kv_lora_rank    = 3
        qk_rope_head_dim = 2
        kv_cache_dim    = 5
        seq_lens        = [15]
        local_prefix    = [5]
        min_prefix      = [5]
        max_prefix      = [5]   # = min = local → no transfer

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
        # total_transfer_tokens == 0 → transfer_buffer has shape (0, 1, 5)
        info.ensure_transfer_buffer_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu"))
        assert info.total_transfer_tokens == 0
        assert info.transfer_buffer.shape[0] == 0

        extend_len = seq_lens[0] - local_prefix[0]  # 10
        k_nope = torch.full((extend_len, 1, kv_lora_rank),    1.5)
        k_pe   = torch.full((extend_len, 1, qk_rope_head_dim), 2.5)
        out_cache_loc = torch.arange(0, extend_len, dtype=torch.int64)

        pool  = MockKVPool(20, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        save_dp_local_kv(info, layer, k_nope, k_pe, out_cache_loc, pool)

        pool._ensure_layer(0)
        kv = pool.store[0]
        expected = [1.5] * kv_lora_rank + [2.5] * qk_rope_head_dim
        for j in range(extend_len):
            assert kv[j, 0, :].tolist() == pytest.approx(expected), \
                f"Zero-transfer extend slot {j}: {kv[j,0,:].tolist()}"

    # ------------------------------------------------------------------
    # Scenario 4: only a subset of requests are dp-local
    # Batch has 3 reqs; only req1 is dp-local.
    # ------------------------------------------------------------------

    def test_only_subset_dp_local(self):
        """Only dp-local requests get KV written; non-local reqs are skipped."""
        kv_lora_rank    = 4
        qk_rope_head_dim = 4
        kv_cache_dim    = 8
        seq_lens        = [20, 18, 16]
        local_prefix    = [3,  5,  2]
        min_prefix      = [3,  2,  2]
        max_prefix      = [10, 8,  7]
        max_rank        = [0,  0,  0]

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
        info.ensure_transfer_buffer_allocated(kv_cache_dim, kv_lora_rank, torch.float32, torch.device("cpu"))
        info.transfer_buffer[:] = 6.0

        # k_nope / k_pe: model processes (seq_len - max_prefix) tokens per request
        # req0: 20-10=10, req1: 18-8=10, req2: 16-7=9 → total 29
        total_model_extend = sum(s - mp for s, mp in zip(seq_lens, max_prefix))  # 10+10+9=29
        k_nope = torch.full((total_model_extend, 1, kv_lora_rank),    8.0)
        k_pe   = torch.full((total_model_extend, 1, qk_rope_head_dim), 9.0)

        # out_cache_loc for req1 only: KV-pool slots = seq_len - local_prefix = 13 slots
        req1_extend_len = seq_lens[1] - local_prefix[1]   # 13 (covers transfer + extend)
        out_cache_loc = torch.arange(200, 200 + req1_extend_len, dtype=torch.int64)

        pool  = MockKVPool(300, kv_lora_rank, qk_rope_head_dim)
        layer = MockLayer(layer_id=0)

        save_dp_local_kv(info, layer, k_nope, k_pe, out_cache_loc, pool)

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

        # Slots 0..199 and 213..299 should not have been written (remain NaN)
        assert kv[0:200].isnan().all(), "Non-dp-local slots should be untouched"
        assert kv[213:].isnan().all(), "Slots beyond req1 should be untouched"
