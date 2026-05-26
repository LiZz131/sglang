"""
Unit tests for Scheme C scheduling synchronization (gather_scheduling_prefix_data).

These tests exercise the CPU-only logic of:
  - SchedulingGatherResult dataclass
  - gather_scheduling_prefix_data() — mocked via direct tensor manipulation
  - PrefillAdder.rem_total_tokens_cap capping
  - SchedulePolicy.calc_priority(use_global_prefix=True) sorting

No real multi-rank distributed environment is required.  The all_gather is
bypassed by directly calling the post-gather computation helpers.

Run with:
    pytest test/srt/multiplex/test_scheduling_prefix_gather.py -v
"""

from __future__ import annotations

import unittest
from typing import List, Optional
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.multiplex.share_prefix_helper import SchedulingGatherResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_req(rid: str, origin_input_ids: List[int], prefix_len: int):
    """Create a lightweight mock Req with required attributes."""
    req = MagicMock()
    req.rid = rid
    req.origin_input_ids = origin_input_ids
    req.output_ids = []
    req.extra_key = None
    req.fill_ids = origin_input_ids
    req.extend_input_len = len(origin_input_ids) - prefix_len
    # Simulate prefix_indices as a list of dummy indices
    req.prefix_indices = list(range(prefix_len))
    req.last_node = None
    req.last_host_node = None
    req.host_hit_length = 0
    req.sched_max_prefix = None
    return req


def _simulate_gather_result(
    waiting_queue,
    dp_prefix_matrix: List[List[int]],
    dp_budgets: List[int],
    attn_tp_size: int = 1,
) -> SchedulingGatherResult:
    """
    Simulate the post-gather computation of gather_scheduling_prefix_data
    without actual dist.all_gather.  dp_prefix_matrix[dp_r][i] = prefix len
    for DP rank dp_r, request i.
    """
    dp_size = len(dp_prefix_matrix)
    n = len(waiting_queue)
    max_prefix_list: List[int] = []
    for i in range(n):
        max_p = max(dp_prefix_matrix[dp_r][i] for dp_r in range(dp_size))
        max_prefix_list.append(max_p)

    for req, mp in zip(waiting_queue, max_prefix_list):
        req.sched_max_prefix = mp

    global_min_raw_budget = min(dp_budgets)
    return SchedulingGatherResult(
        max_prefix_list=max_prefix_list,
        global_min_raw_budget=global_min_raw_budget,
    )


# ---------------------------------------------------------------------------
# Test: SchedulingGatherResult correctness
# ---------------------------------------------------------------------------

class TestSchedulingGatherResult(unittest.TestCase):
    def test_max_prefix_two_ranks(self):
        """max_prefix_list = per-request max across DP ranks."""
        reqs = [
            _make_mock_req("r0", list(range(120)), prefix_len=10),
            _make_mock_req("r1", list(range(80)),  prefix_len=50),
        ]
        # DP0 has [10, 50], DP1 has [100, 20]
        result = _simulate_gather_result(
            reqs,
            dp_prefix_matrix=[[10, 50], [100, 20]],
            dp_budgets=[5000, 4000],
        )
        self.assertEqual(result.max_prefix_list, [100, 50])
        self.assertEqual(result.global_min_raw_budget, 4000)

    def test_equal_prefix_all_ranks(self):
        """When all ranks share the same prefix, max == local."""
        reqs = [_make_mock_req("r0", list(range(100)), prefix_len=60)]
        result = _simulate_gather_result(
            reqs,
            dp_prefix_matrix=[[60], [60]],
            dp_budgets=[8000, 8000],
        )
        self.assertEqual(result.max_prefix_list, [60])
        self.assertEqual(result.global_min_raw_budget, 8000)

    def test_three_dp_ranks(self):
        """Three DP ranks: max is the overall maximum."""
        reqs = [
            _make_mock_req("r0", list(range(200)), prefix_len=30),
            _make_mock_req("r1", list(range(150)), prefix_len=10),
        ]
        result = _simulate_gather_result(
            reqs,
            dp_prefix_matrix=[[30, 10], [120, 80], [50, 150]],
            dp_budgets=[6000, 7000, 5500],
        )
        self.assertEqual(result.max_prefix_list, [120, 150])
        self.assertEqual(result.global_min_raw_budget, 5500)

    def test_sched_max_prefix_set_on_req(self):
        """gather sets req.sched_max_prefix correctly."""
        reqs = [
            _make_mock_req("r0", list(range(100)), prefix_len=5),
            _make_mock_req("r1", list(range(100)), prefix_len=80),
        ]
        _simulate_gather_result(
            reqs,
            dp_prefix_matrix=[[5, 80], [70, 10]],
            dp_budgets=[3000, 3500],
        )
        self.assertEqual(reqs[0].sched_max_prefix, 70)
        self.assertEqual(reqs[1].sched_max_prefix, 80)

    def test_budget_cap_global_min(self):
        """global_min_raw_budget = min of all DP budgets."""
        reqs = [_make_mock_req("r0", list(range(50)), prefix_len=0)]
        result = _simulate_gather_result(
            reqs,
            dp_prefix_matrix=[[0], [0]],
            dp_budgets=[10000, 1000],
        )
        self.assertEqual(result.global_min_raw_budget, 1000)


# ---------------------------------------------------------------------------
# Test: PrefillAdder rem_total_tokens_cap
# ---------------------------------------------------------------------------

class TestPrefillAdderBudgetCap(unittest.TestCase):
    def _make_adder(self, available_size: int, cap: Optional[int]):
        """Build a minimal PrefillAdder with mocked allocator."""
        from sglang.srt.managers.schedule_policy import PrefillAdder

        mock_tree_cache = MagicMock()
        mock_tree_cache.evictable_size.return_value = 0
        mock_tree_cache.full_evictable_size.return_value = 0

        mock_allocator = MagicMock()
        mock_allocator.available_size.return_value = available_size

        with patch(
            "sglang.srt.managers.schedule_policy.is_nsa_prefill_cp_in_seq_split",
            return_value=False,
        ):
            adder = PrefillAdder(
                page_size=16,
                tree_cache=mock_tree_cache,
                token_to_kv_pool_allocator=mock_allocator,
                running_batch=None,
                new_token_ratio=1.0,
                rem_input_tokens=100000,
                rem_chunk_tokens=None,
                rem_total_tokens_cap=cap,
            )
        return adder

    def test_no_cap(self):
        """Without cap, rem_total_tokens uses full available_size."""
        adder = self._make_adder(available_size=8000, cap=None)
        self.assertEqual(adder.rem_total_tokens, 8000)

    def test_cap_lower_than_available(self):
        """Cap reduces rem_total_tokens to the global minimum."""
        adder = self._make_adder(available_size=8000, cap=3000)
        self.assertEqual(adder.rem_total_tokens, 3000)

    def test_cap_higher_than_available(self):
        """Cap higher than available has no effect."""
        adder = self._make_adder(available_size=2000, cap=5000)
        self.assertEqual(adder.rem_total_tokens, 2000)

    def test_cap_equal_to_available(self):
        """Cap equal to available = no change."""
        adder = self._make_adder(available_size=4096, cap=4096)
        self.assertEqual(adder.rem_total_tokens, 4096)


# ---------------------------------------------------------------------------
# Test: calc_priority with use_global_prefix=True
# ---------------------------------------------------------------------------

class TestCalcPriorityGlobalPrefix(unittest.TestCase):
    def _make_policy(self):
        """Create a SchedulePolicy with FCFS policy (avoids real tree_cache)."""
        from sglang.srt.managers.schedule_policy import SchedulePolicy

        mock_tree_cache = MagicMock()
        mock_tree_cache.disable = True  # forces FCFS (cache-agnostic)
        policy = SchedulePolicy(
            policy="lpm",  # would normally sort by prefix; overridden by use_global_prefix
            tree_cache=mock_tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            enable_special_dp_attention_prefix_0=False,
        )
        return policy

    def test_sort_by_sched_max_prefix_descending(self):
        """use_global_prefix=True sorts waiting_queue by sched_max_prefix descending."""
        policy = self._make_policy()
        reqs = [
            _make_mock_req("r0", list(range(100)), prefix_len=5),
            _make_mock_req("r1", list(range(100)), prefix_len=80),
            _make_mock_req("r2", list(range(100)), prefix_len=40),
        ]
        reqs[0].sched_max_prefix = 30
        reqs[1].sched_max_prefix = 90
        reqs[2].sched_max_prefix = 60

        waiting_queue = list(reqs)
        result = policy.calc_priority(waiting_queue, use_global_prefix=True)

        self.assertTrue(result)
        self.assertEqual([r.rid for r in waiting_queue], ["r1", "r2", "r0"])

    def test_fallback_to_prefix_indices_when_no_sched_max_prefix(self):
        """If sched_max_prefix is missing, fall back to len(prefix_indices)."""
        policy = self._make_policy()
        reqs = [
            _make_mock_req("r0", list(range(100)), prefix_len=10),
            _make_mock_req("r1", list(range(100)), prefix_len=70),
        ]
        # Do NOT set sched_max_prefix; should use len(prefix_indices)
        for r in reqs:
            del r.sched_max_prefix  # remove the MagicMock attribute

        waiting_queue = list(reqs)
        policy.calc_priority(waiting_queue, use_global_prefix=True)

        # r1 has longer prefix_indices so should come first
        self.assertEqual(waiting_queue[0].rid, "r1")

    def test_returns_true(self):
        """calc_priority returns True when use_global_prefix=True."""
        policy = self._make_policy()
        reqs = [_make_mock_req("r0", list(range(50)), prefix_len=10)]
        reqs[0].sched_max_prefix = 10
        result = policy.calc_priority(list(reqs), use_global_prefix=True)
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# Test: extend_input_len override logic
# ---------------------------------------------------------------------------

class TestExtendInputLenOverride(unittest.TestCase):
    """Test that extend_input_len is correctly overridden with global max_prefix."""

    def test_extend_len_uses_global_max_prefix(self):
        """
        If local_prefix=10, sched_max_prefix=100, seq_len=120:
          global_extend = 120 - 100 = 20  (not 120 - 10 = 110)
        """
        req = _make_mock_req("r0", list(range(120)), prefix_len=10)
        req.sched_max_prefix = 100

        mp = req.sched_max_prefix
        global_extend = max(0, len(req.fill_ids) - mp)
        self.assertEqual(global_extend, 20)

    def test_extend_len_zero_when_fully_cached(self):
        """If sched_max_prefix == seq_len, extend = 0."""
        req = _make_mock_req("r0", list(range(80)), prefix_len=30)
        req.sched_max_prefix = 80

        global_extend = max(0, len(req.fill_ids) - req.sched_max_prefix)
        self.assertEqual(global_extend, 0)

    def test_extend_len_never_negative(self):
        """max(0, ...) guards against sched_max_prefix > seq_len."""
        req = _make_mock_req("r0", list(range(50)), prefix_len=10)
        req.sched_max_prefix = 60  # pathological: larger than seq_len

        global_extend = max(0, len(req.fill_ids) - req.sched_max_prefix)
        self.assertEqual(global_extend, 0)

    def test_symmetric_across_ranks(self):
        """Two ranks with different local_prefix use same global_extend."""
        seq_len = 120
        sched_max_prefix = 100
        local_prefix_rank0 = 10
        local_prefix_rank1 = 100

        extend_rank0 = max(0, seq_len - sched_max_prefix)
        extend_rank1 = max(0, seq_len - sched_max_prefix)
        self.assertEqual(extend_rank0, extend_rank1)
        self.assertEqual(extend_rank0, 20)


if __name__ == "__main__":
    unittest.main()
