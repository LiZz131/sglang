"""
Unit tests for prefill_mem_stream.py.

Run:
    pytest test/srt/utils/test_prefill_mem_stream.py -v
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.prefill_mem_stream import (
    PrefillMemStream,
    is_active,
    scratch_empty,
)


@dataclass
class _FakeForwardBatch:
    forward_mode: ForwardMode
    prefill_mem_stream_active: bool = False


def _mock_forward_batch(mode=ForwardMode.SPLIT_PREFILL):
    return _FakeForwardBatch(forward_mode=mode)


@pytest.fixture(autouse=True)
def _clear_state():
    PrefillMemStream.clear_all()
    yield
    PrefillMemStream.clear_all()


@contextmanager
def _patch_enabled_deps(
    *,
    flag: bool = True,
    capture: bool = False,
    piecewise: bool = False,
):
    server_args = MagicMock()
    server_args.enable_prefill_mem_stream = flag
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
            assert PrefillMemStream.enabled(
                _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
            )
            assert not PrefillMemStream.enabled(
                _mock_forward_batch(ForwardMode.EXTEND)
            )

    def test_disabled_when_flag_off(self):
        with _patch_enabled_deps(flag=False):
            assert not PrefillMemStream.enabled(
                _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
            )

    def test_disabled_during_capture(self):
        with _patch_enabled_deps(capture=True):
            assert not PrefillMemStream.enabled(
                _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
            )


class TestBinding:
    def test_binding_sets_active(self):
        fb = _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
        with _patch_enabled_deps():
            with PrefillMemStream.binding(fb) as active:
                assert active is True
                assert is_active()
                assert fb.prefill_mem_stream_active is True
        assert not is_active()
        assert fb.prefill_mem_stream_active is False

    def test_binding_inactive_when_disabled(self):
        fb = _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
        with _patch_enabled_deps(flag=False):
            with PrefillMemStream.binding(fb) as active:
                assert active is False
                assert not is_active()


class TestScratchEmpty:
    def test_inactive_uses_plain_empty(self):
        t = scratch_empty((4, 8), dtype=torch.float32, device=torch.device("cpu"))
        assert t.shape == (4, 8)

    def test_active_cpu_fallback(self):
        fb = _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
        with _patch_enabled_deps():
            with PrefillMemStream.binding(fb):
                t = scratch_empty((2, 3), dtype=torch.float32, device=torch.device("cpu"))
                assert t.shape == (2, 3)

    def test_sequential_allocs_are_distinct_tensors(self):
        fb = _mock_forward_batch(ForwardMode.SPLIT_PREFILL)
        with _patch_enabled_deps():
            with PrefillMemStream.binding(fb):
                a = scratch_empty((8,), dtype=torch.float32, device=torch.device("cpu"))
                b = scratch_empty((8,), dtype=torch.float32, device=torch.device("cpu"))
                assert a.data_ptr() != b.data_ptr() or a is not b
