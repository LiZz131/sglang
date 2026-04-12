"""
Green context stream tests. NVTX ranges are named `pytest::<test_name>` for nsys / Nsight Systems.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import pytest
import torch

from sgl_kernel import (
    create_greenctx_stream_by_value,
    create_greenctx_streams_by_value_enhanced,
    get_sm_available,
)


@contextlib.contextmanager
def nvtx_test(name: str) -> Iterator[None]:
    """One NVTX range per test (visible in nsys timeline when CUDA profiling is enabled)."""
    if not torch.cuda.is_available():
        yield
        return
    nv = torch.cuda.nvtx
    span = getattr(nv, "range", None)
    if span is not None:
        with span(name):
            yield
    else:
        nv.range_push(name)
        try:
            yield
        finally:
            nv.range_pop()


def test_green_ctx():
    with nvtx_test("pytest::test_green_ctx"):
        A = torch.randn(5120, 5120).cuda()
        B = torch.randn(5120, 5120).cuda()
        C = torch.matmul(A, B)
        sm_counts = get_sm_available(0)
        stream_group = create_greenctx_stream_by_value(sm_counts // 2, sm_counts // 2, 0)
        with torch.cuda.stream(stream_group[0]):
            for _ in range(100):
                result_0 = torch.matmul(A, B)
        with torch.cuda.stream(stream_group[1]):
            for _ in range(100):
                result_1 = torch.matmul(A, B)
        torch.cuda.synchronize()
        assert torch.allclose(result_0, C)
        assert torch.allclose(result_1, C)


def test_green_ctx_multi_stream_enhanced():
    with nvtx_test("pytest::test_green_ctx_multi_stream_enhanced"):
        A = torch.randn(5120, 5120).cuda()
        B = torch.randn(5120, 5120).cuda()
        C = torch.matmul(A, B)
        sm_counts = get_sm_available(0)
        streams_a, streams_b, sm_a, sm_b = create_greenctx_streams_by_value_enhanced(
            sm_counts // 2, sm_counts // 2, 1, 2, 0
        )
        assert sm_a > 0 and sm_b > 0
        assert len(streams_a) == 1 and len(streams_b) == 2
        with torch.cuda.stream(streams_a[0]):
            result_a = torch.matmul(A, B)
        with torch.cuda.stream(streams_b[0]):
            result_b0 = torch.matmul(A, B)
        with torch.cuda.stream(streams_b[1]):
            result_b1 = torch.matmul(A, B)
        torch.cuda.synchronize()
        assert torch.allclose(result_a, C)
        assert torch.allclose(result_b0, C)
        assert torch.allclose(result_b1, C)


def test_green_ctx_same_partition_event_wait():
    """
    Two streams on the same green SM partition must honor CUDA event ordering:
    s1 must not read T until s0 has finished writing it (recorded on the event).
    """
    with nvtx_test("pytest::test_green_ctx_same_partition_event_wait"):
        device = 0
        torch.cuda.synchronize()

        sm_counts = get_sm_available(device)
        streams_a, _streams_b, sm_a, sm_b = create_greenctx_streams_by_value_enhanced(
            sm_counts // 2, sm_counts // 2, 2, 1, device
        )
        assert sm_a > 0 and sm_b > 0
        assert len(streams_a) == 2

        s0, s1 = streams_a[0], streams_a[1]
        A = torch.randn(2048, 2048, device=f"cuda:{device}")
        B = torch.randn(2048, 2048, device=f"cuda:{device}")
        T = torch.empty(2048, 2048, device=f"cuda:{device}")
        ref = torch.matmul(A, B)
        expected = ref + 1.0

        evt = torch.cuda.Event()

        with torch.cuda.stream(s0):
            torch.matmul(A, B, out=T)
            evt.record()

        with torch.cuda.stream(s1):
            s1.wait_event(evt)
            Out = T + 1.0

        torch.cuda.synchronize()
        assert evt.query(), "event must be complete after device-wide sync"
        assert torch.allclose(Out, expected), "s1 must observe s0's write to T after wait_event"

        # Explicit numeric spot-check (not only allclose tolerance path)
        assert Out.shape == expected.shape
        mid = Out.shape[0] // 2
        assert float(Out[mid, mid].item()) == float(expected[mid, mid].item())


def test_green_ctx_same_partition_wait_stream():
    """
    Stream-ordered dependency via wait_stream: s1 waits until s0 completes all prior work,
    then consumes T produced on s0.
    """
    with nvtx_test("pytest::test_green_ctx_same_partition_wait_stream"):
        device = 0
        torch.cuda.synchronize()

        sm_counts = get_sm_available(device)
        streams_a, _streams_b, _, _ = create_greenctx_streams_by_value_enhanced(
            sm_counts // 2, sm_counts // 2, 2, 1, device
        )
        s0, s1 = streams_a[0], streams_a[1]

        A = torch.randn(1536, 1536, device=f"cuda:{device}")
        B = torch.randn(1536, 1536, device=f"cuda:{device}")
        T = torch.empty(1536, 1536, device=f"cuda:{device}")
        ref = torch.matmul(A, B)
        expected = ref * 2.0

        with torch.cuda.stream(s0):
            torch.matmul(A, B, out=T)

        with torch.cuda.stream(s1):
            s1.wait_stream(s0)
            Out = T * 2.0

        torch.cuda.synchronize()
        assert torch.allclose(Out, expected), "s1 must observe s0's write to T after wait_stream"
        assert float(Out[0, 0].item()) == float(expected[0, 0].item())


def test_green_ctx_event_chain_cross_streams():
    """
    Event recorded on stream A partition, waited on stream B partition (different green ctx):
    still a valid device-wide ordering tool for nsys visibility.
    """
    with nvtx_test("pytest::test_green_ctx_event_chain_cross_streams"):
        device = 0
        torch.cuda.synchronize()

        sm_counts = get_sm_available(device)
        streams_a, streams_b, _, _ = create_greenctx_streams_by_value_enhanced(
            sm_counts // 2, sm_counts // 2, 1, 1, device
        )
        sa, sb = streams_a[0], streams_b[0]

        token = torch.zeros(1, dtype=torch.int32, device=f"cuda:{device}")
        evt = torch.cuda.Event()

        with torch.cuda.stream(sa):
            token.fill_(1)
            evt.record()

        with torch.cuda.stream(sb):
            sb.wait_event(evt)
            snap = token.clone()
            token.fill_(2)

        torch.cuda.synchronize()
        assert evt.query()
        assert int(snap.item()) == 1, "after wait_event, sb must observe sa's write (value 1)"
        assert int(token.item()) == 2, "sb then overwrites token to 2"


if __name__ == "__main__":
    pytest.main([__file__])
