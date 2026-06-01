"""
Grow-only scratch buffer pool for split-prefill (PD-MUX prefill) big tensors.

Avoids repeated torch.empty / PyTorch CUDACachingAllocator cudaMalloc on hot
paths (q_nope bmm, flash_attn out, fused_moe cache).  Disabled during CUDA
graph capture, decode, and piecewise CUDA graph.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# fused_moe.py CHUNK_SIZE — keep in sync for moe_numel upper bound
_MOE_CHUNK_SIZE = 64 * 1024

_INSTANCES: dict[tuple[str, str], "PrefillScratchBufferPool"] = {}
_active_local = threading.local()


def _pool_key(device: torch.device, dtype: torch.dtype) -> tuple[str, str]:
    return (str(device), str(dtype))


def _grow_capacity(current: int, needed: int) -> int:
    if current >= needed:
        return current
    cap = max(needed, 1)
    while cap < needed:
        cap *= 2
    return cap


class PrefillScratchBufferPool:
    """Per-(device, dtype) grow-only scratch buffers for split-prefill."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self._attn_buf: Optional[torch.Tensor] = None
        self._moe_buf: Optional[torch.Tensor] = None
        self._attn_cap: int = 0
        self._moe_cap: int = 0
        # Per-layer sub-region sizes (set by acquire_* each layer)
        self._attn_region_numel: int = 0

    @classmethod
    def get(cls, device: torch.device, dtype: torch.dtype) -> "PrefillScratchBufferPool":
        key = _pool_key(device, dtype)
        pool = _INSTANCES.get(key)
        if pool is None:
            pool = cls(device, dtype)
            _INSTANCES[key] = pool
        return pool

    @staticmethod
    def get_active() -> Optional["PrefillScratchBufferPool"]:
        return getattr(_active_local, "pool", None)

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

            if not get_global_server_args().enable_prefill_scratch_pool:
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

    def ensure_capacity(self, *, attn_numel: int, moe_numel: int) -> None:
        attn_numel = max(int(attn_numel), 0)
        moe_numel = max(int(moe_numel), 0)

        if attn_numel > self._attn_cap:
            old = self._attn_cap
            new_cap = _grow_capacity(self._attn_cap, attn_numel)
            self._attn_buf = torch.empty(new_cap, device=self.device, dtype=self.dtype)
            self._attn_cap = new_cap
            logger.info(
                "[prefill_scratch] GROW attn: old_cap=%d new_cap=%d device=%s dtype=%s",
                old,
                new_cap,
                self.device,
                self.dtype,
            )

        if moe_numel > self._moe_cap:
            old = self._moe_cap
            new_cap = _grow_capacity(self._moe_cap, moe_numel)
            self._moe_buf = torch.empty(new_cap, device=self.device, dtype=self.dtype)
            self._moe_cap = new_cap
            logger.info(
                "[prefill_scratch] GROW moe: old_cap=%d new_cap=%d device=%s dtype=%s",
                old,
                new_cap,
                self.device,
                self.dtype,
            )

        if attn_numel > 0 or moe_numel > 0:
            logger.debug(
                "[prefill_scratch] REUSE attn_cap=%d moe_cap=%d need_attn=%d need_moe=%d",
                self._attn_cap,
                self._moe_cap,
                attn_numel,
                moe_numel,
            )

    def set_attn_region_numel(self, region_numel: int) -> None:
        """Size of one attn sub-region (q_nope / flash out each use one region)."""
        self._attn_region_numel = max(int(region_numel), 0)

    def acquire_attn_bmm(self, shape: Tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for d in shape:
            numel *= int(d)
        self.set_attn_region_numel(numel)
        assert self._attn_buf is not None and numel <= self._attn_cap
        return self._attn_buf[:numel].view(*shape)

    def acquire_flash_out(self, shape: Tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for d in shape:
            numel *= int(d)
        region = self._attn_region_numel or numel
        offset = region
        end = offset + numel
        assert self._attn_buf is not None and end <= self._attn_cap
        return self._attn_buf[offset:end].view(*shape)

    def acquire_moe_1d(self, numel: int) -> torch.Tensor:
        numel = int(numel)
        assert self._moe_buf is not None and numel <= self._moe_cap
        return self._moe_buf[:numel]

    @classmethod
    @contextmanager
    def binding(
        cls,
        forward_batch,
        device: torch.device,
        dtype: torch.dtype,
        attn_numel: int,
        moe_numel: int,
    ) -> Iterator[Optional["PrefillScratchBufferPool"]]:
        if not cls.enabled(forward_batch):
            forward_batch.prefill_scratch_pool = None
            yield None
            return

        pool = cls.get(device, dtype)
        pool.ensure_capacity(attn_numel=attn_numel, moe_numel=moe_numel)
        forward_batch.prefill_scratch_pool = pool

        prev = getattr(_active_local, "pool", None)
        _active_local.pool = pool
        try:
            logger.debug(
                "[prefill_scratch] binding enabled=1 attn_numel=%d moe_numel=%d device=%s",
                attn_numel,
                moe_numel,
                device,
            )
            yield pool
        finally:
            _active_local.pool = prev
            forward_batch.prefill_scratch_pool = None

    @classmethod
    def clear_all(cls) -> None:
        _INSTANCES.clear()
        _active_local.pool = None


def compute_scratch_requirements(
    model,
    layer_start_idx: int,
    num_tokens: int,
) -> Tuple[int, int]:
    """Return (attn_numel, moe_numel) upper bounds for a split-prefill chunk."""
    num_tokens = max(int(num_tokens), 0)
    layers = getattr(model, "layers", None)
    attn_heads = 128
    kv_lora_rank = 512
    if layers is not None and len(layers) > layer_start_idx:
        layer = layers[layer_start_idx]
        sa = getattr(layer, "self_attn", None)
        if sa is not None:
            attn_heads = int(getattr(sa, "tp_num_heads", attn_heads))
            kv_lora_rank = int(getattr(sa, "kv_lora_rank", kv_lora_rank))

    attn_numel = 2 * num_tokens * attn_heads * kv_lora_rank

    config = getattr(model, "config", None)
    topk = int(getattr(config, "num_experts_per_tok", 8) or 8)
    intermediate = int(getattr(config, "moe_intermediate_size", 2048) or 2048)
    hidden = int(getattr(config, "hidden_size", 7168) or 7168)
    max_dim = max(intermediate, hidden)

    m = min(num_tokens, _MOE_CHUNK_SIZE)
    # Conservative: token*topk plus TMA padding headroom (same order as fused_moe)
    moe_numel = (m * topk + m * topk) * max_dim
    return attn_numel, moe_numel
