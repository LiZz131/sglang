"""CUDA memory allocation snapshot helpers for PyTorch memory_viz."""

import logging

logger = logging.getLogger(__name__)

_MEMORY_HISTORY_ACTIVE = False


def enable_memory_history(max_entries: int = 500_000) -> bool:
    """Start recording allocation history. Returns False if already active."""
    global _MEMORY_HISTORY_ACTIVE
    if _MEMORY_HISTORY_ACTIVE:
        return False

    import torch

    if not torch.cuda.is_available():
        return False

    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="all",
        max_entries=max_entries,
    )
    _MEMORY_HISTORY_ACTIVE = True
    logger.info("CUDA memory history enabled (max_entries=%d)", max_entries)
    return True


def disable_memory_history() -> None:
    global _MEMORY_HISTORY_ACTIVE
    if not _MEMORY_HISTORY_ACTIVE:
        return

    import torch

    torch.cuda.memory._record_memory_history(enabled=None)
    _MEMORY_HISTORY_ACTIVE = False


def is_memory_history_active() -> bool:
    return _MEMORY_HISTORY_ACTIVE


def dump_memory_snapshot(path: str) -> None:
    import torch

    torch.cuda.memory._dump_snapshot(path)
    logger.info("CUDA memory snapshot saved to %s", path)


def maybe_enable_from_server_args(server_args) -> None:
    if not server_args.enable_cuda_memory_snapshot_from_start:
        return
    enable_memory_history(max_entries=server_args.cuda_memory_snapshot_max_entries)
