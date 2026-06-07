"""
mem_stream scratch allocation for split-prefill (PD-MUX prefill) big tensors.

Each allocation is an independent ``torch.empty`` on a dedicated CUDA stream,
followed by ``record_event`` / ``wait_event`` so the current compute stream sees
completed allocation.  No pre-sized pool; PyTorch caching allocator + refcount
handle reuse.

Disabled during CUDA graph capture, decode, and piecewise CUDA graph.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

_mem_streams: dict[str, torch.cuda.Stream] = {}
_active_local = threading.local()


def _device_key(device: torch.device) -> str:
    return str(device)


def get_mem_stream(device: torch.device) -> Optional[torch.cuda.Stream]:
    if device.type != "cuda":
        return None
    key = _device_key(device)
    stream = _mem_streams.get(key)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _mem_streams[key] = stream
    return stream


def is_active() -> bool:
    return bool(getattr(_active_local, "active", False))


class PrefillMemStream:
    @staticmethod
    def enabled(forward_batch) -> bool:
        if forward_batch is None:
            return False
        try:
            from sglang.srt.compilation.piecewise_context_manager import (
                is_in_piecewise_cuda_graph,
            )
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
            from sglang.srt.server_args import get_global_server_args

            if not get_global_server_args().enable_prefill_mem_stream:
                return False
            fm = getattr(forward_batch, "forward_mode", None)
            if fm is None or not fm.is_split_prefill():
                return False
            if get_is_capture_mode():
                return False
            if fm.is_decode():
                return False
            if is_in_piecewise_cuda_graph():
                return False
            return True
        except Exception:
            return False

    @classmethod
    @contextmanager
    def binding(cls, forward_batch) -> Iterator[bool]:
        if not cls.enabled(forward_batch):
            forward_batch.prefill_mem_stream_active = False
            yield False
            return

        forward_batch.prefill_mem_stream_active = True
        prev = getattr(_active_local, "active", False)
        _active_local.active = True
        try:
            logger.debug("[prefill_mem_stream] binding enabled=1")
            yield True
        finally:
            _active_local.active = prev
            forward_batch.prefill_mem_stream_active = False

    @classmethod
    def clear_all(cls) -> None:
        _mem_streams.clear()
        _active_local.active = False


def _normalize_shape(shape: Union[Tuple[int, ...], torch.Size]) -> Tuple[int, ...]:
    return tuple(int(d) for d in shape)


def scratch_empty(
    shape: Union[Tuple[int, ...], torch.Size],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocate on mem_stream when active; otherwise plain ``torch.empty``."""
    shape = _normalize_shape(shape)
    if not is_active() or device.type != "cuda":
        return torch.empty(shape, dtype=dtype, device=device)

    mem_stream = get_mem_stream(device)
    if mem_stream is None:
        return torch.empty(shape, dtype=dtype, device=device)

    with torch.cuda.stream(mem_stream):
        tensor = torch.empty(shape, dtype=dtype, device=device)
    alloc_event = mem_stream.record_event()
    torch.cuda.current_stream(device).wait_event(alloc_event)
    logger.debug(
        "[prefill_mem_stream] alloc shape=%s dtype=%s device=%s",
        shape,
        dtype,
        device,
    )
    return tensor


def scratch_empty_like(
    like: torch.Tensor,
    shape: Optional[Union[Tuple[int, ...], torch.Size]] = None,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    return scratch_empty(
        _normalize_shape(shape if shape is not None else like.shape),
        dtype=dtype if dtype is not None else like.dtype,
        device=like.device,
    )


def scratch_empty_1d(
    numel: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return scratch_empty((int(numel),), dtype=dtype, device=device)
