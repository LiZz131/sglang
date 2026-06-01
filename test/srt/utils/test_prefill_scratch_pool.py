"""
Unit tests for prefill_scratch_pool.py.

Run:
    pytest test/srt/utils/test_prefill_scratch_pool.py -v
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.prefill_scratch_pool import (
    PrefillScratchBufferPool,
    compute_scratch_requirements,
)


@dataclass
class _FakeForwardBatch:
    forward_mode: ForwardMode
    prefill_scratch_pool: object = None


def _mock_forward_batch(mode=ForwardMode.SPLIT_PREFILL):
    return _FakeForwardBatch(forward_mode=mode)


@pytest.fixture(autouse=True)
def _clear_pools():
    PrefillScratchBufferPool.clear_all()
    yield
    PrefillScratchBufferPool.clear_all()


@contextmanager
def _patch_enabled_deps(
    *,
    flag: bool = True,
    capture: bool = False,
    piecewise: bool = False,
):
    server_args = MagicMock()
    server_args.enable_prefill_scratch_pool = flag
    with patch(
        "sglang.srt.server_args.get_global_server_args",
        return_value=server_args,
    ), patch(
        "sglang.srt.model_executor.cuda_graph_runner.get_is_capture_mode",
        return_value=capture,
    ), patch(
        "sglang.srt.compilation.piecewise_context_manager.is_in_piecewise_cuda_graph",
        return_value=piecewise,
    ):
        yield


class TestEnabledGuards:
    def test_enabled_requires_split_prefill(self):
        with _patch_enabled_deps():
            assert PrefillScratchBufferPool.enabled(
                _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
            )
            assert not PrefillScratchBufferPool.enabled(
                _mock_forward_batch(ForwardMode.EXTEND)
            )

    def test_disabled_when_flag_off(self):
        with _patch_enabled_deps(flag=False):
            assert not PrefillScratchBufferPool.enabled(
                _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
            )

    def test_disabled_during_capture(self):
        with _patch_enabled_deps(capture=True):
            assert not PrefillScratchBufferPool.enabled(
                _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
            )

    def test_disabled_during_piecewise(self):
        with _patch_enabled_deps(piecewise=True):
            assert not PrefillScratchBufferPool.enabled(
                _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
            )

    def test_disabled_for_decode_mode(self):
        with _patch_enabled_deps():
            assert not PrefillScratchBufferPool.enabled(
                _mock_forward_batch(ForwardMode.DECODE)
            )


class TestGrowOnly:
    def test_grow_only_attn_moe(self):
        pool = PrefillScratchBufferPool.get(torch.device("cpu"), torch.float32)
        pool.ensure_capacity(attn_numel=100, moe_numel=200)
        attn_ptr = pool._attn_buf.data_ptr()
        moe_ptr = pool._moe_buf.data_ptr()

        pool.ensure_capacity(attn_numel=50, moe_numel=80)
        assert pool._attn_buf.data_ptr() == attn_ptr
        assert pool._moe_buf.data_ptr() == moe_ptr

        pool.ensure_capacity(attn_numel=500, moe_numel=600)
        assert pool._attn_buf.data_ptr() != attn_ptr
        assert pool._moe_buf.data_ptr() != moe_ptr
        assert pool._attn_cap >= 500
        assert pool._moe_cap >= 600


class TestAcquireRegions:
    def test_acquire_attn_two_regions_no_overlap(self):
        pool = PrefillScratchBufferPool.get(torch.device("cpu"), torch.float32)
        pool.ensure_capacity(attn_numel=200, moe_numel=0)

        bmm = pool.acquire_attn_bmm((2, 3, 4))
        flash = pool.acquire_flash_out((2, 3, 4))

        assert bmm.numel() == 24
        assert flash.numel() == 24
        assert bmm.data_ptr() == pool._attn_buf.data_ptr()
        assert flash.data_ptr() == pool._attn_buf.data_ptr() + 24 * bmm.element_size()
        bmm.fill_(1.0)
        flash.fill_(2.0)
        assert pool._attn_buf[0].item() == 1.0
        assert pool._attn_buf[24].item() == 2.0

    def test_acquire_moe_1d(self):
        pool = PrefillScratchBufferPool.get(torch.device("cpu"), torch.float32)
        pool.ensure_capacity(attn_numel=0, moe_numel=128)
        buf = pool.acquire_moe_1d(64)
        assert buf.shape == (64,)
        assert buf.data_ptr() == pool._moe_buf.data_ptr()


class TestBinding:
    def test_binding_active_for_moe(self):
        fb = _mock_forward_batch(ForwardMode.SPLIT_PREFILL)

        with _patch_enabled_deps():
            with PrefillScratchBufferPool.binding(
                fb,
                torch.device("cpu"),
                torch.float32,
                attn_numel=64,
                moe_numel=128,
            ) as pool:
                assert pool is not None
                assert PrefillScratchBufferPool.get_active() is pool
                assert fb.prefill_scratch_pool is pool

        assert PrefillScratchBufferPool.get_active() is None
        assert fb.prefill_scratch_pool is None

    def test_binding_yields_none_when_disabled(self):
        fb = _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
        with _patch_enabled_deps(flag=False):
            with PrefillScratchBufferPool.binding(
                fb,
                torch.device("cpu"),
                torch.float32,
                attn_numel=64,
                moe_numel=128,
            ) as pool:
                assert pool is None
                assert PrefillScratchBufferPool.get_active() is None


class TestAttnBmmShape:
    """qv layout: bmm yields [H,T,D]; one transpose -> [T,H,D] for flash_attn."""

    def test_pool_bmm_matches_direct_bmm_shape(self):
        H, T, K, D = 8, 32, 128, 512
        pool = PrefillScratchBufferPool.get(torch.device("cpu"), torch.float32)
        pool.ensure_capacity(attn_numel=2 * H * T * D, moe_numel=0)

        q_bmm = torch.randn(H, T, K)
        w = torch.randn(H, K, D)

        bmm_out = pool.acquire_attn_bmm((H, T, D))
        torch.bmm(q_bmm, w, out=bmm_out)
        q_nope_out = bmm_out.transpose(0, 1)

        ref = torch.bmm(q_bmm, w).transpose(0, 1)
        assert q_nope_out.shape == (T, H, D)
        assert q_nope_out.shape == ref.shape


class TestComputeRequirements:
    def test_compute_scratch_requirements_smoke(self):
        config = MagicMock()
        config.num_experts_per_tok = 8
        config.moe_intermediate_size = 2048
        config.hidden_size = 7168
        self_attn = MagicMock()
        self_attn.tp_num_heads = 16
        self_attn.kv_lora_rank = 512
        layer = MagicMock()
        layer.self_attn = self_attn
        model = MagicMock()
        model.layers = [layer]
        model.config = config

        attn, moe = compute_scratch_requirements(model, 0, num_tokens=32)
        assert attn == 2 * 32 * 16 * 512
        assert moe > 0


class TestClearAll:
    def test_clear_all(self):
        pool = PrefillScratchBufferPool.get(torch.device("cpu"), torch.float32)
        pool.ensure_capacity(attn_numel=16, moe_numel=16)
        PrefillScratchBufferPool.clear_all()
        assert PrefillScratchBufferPool.get_active() is None
        new_pool = PrefillScratchBufferPool.get(torch.device("cpu"), torch.float32)
        assert new_pool._attn_cap == 0
