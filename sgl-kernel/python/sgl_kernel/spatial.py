import torch
from torch.cuda.streams import ExternalStream

try:
    from . import spatial_ops  # triggers TORCH extension registration
except Exception as _e:
    _spatial_import_error = _e
else:
    _spatial_import_error = None

_IMPORT_ERROR = ImportError(
    "Failed to load sgl_kernel.spatial_ops extension. Ensure CUDA Driver >= 12.4"
)


def create_greenctx_stream_by_value(
    SM_a: int, SM_b: int, device_id: int = None
) -> tuple[ExternalStream, ExternalStream]:
    """
    Create two streams for greenctx.
    Args:
        sm_A (int): The SM of stream A.
        sm_B (int): The weight of stream B.
        device_id (int): The device id.
    Returns:
        tuple[ExternalStream, ExternalStream]: The two streams.
    """
    if _spatial_import_error is not None:
        raise _IMPORT_ERROR from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    res = torch.ops.sgl_kernel.create_greenctx_stream_by_value(SM_a, SM_b, device_id)

    stream_a = ExternalStream(
        stream_ptr=res[0], device=torch.device(f"cuda:{device_id}")
    )
    stream_b = ExternalStream(
        stream_ptr=res[1], device=torch.device(f"cuda:{device_id}")
    )

    return stream_a, stream_b


def create_greenctx_streams_by_value_enhanced(
    SM_a: int,
    SM_b: int,
    n_streams_a: int,
    n_streams_b: int,
    device_id: int = None,
) -> tuple[tuple[ExternalStream, ...], tuple[ExternalStream, ...], int, int]:
    """
    Create two SM partitions (same split semantics as create_greenctx_stream_by_value) and attach
    multiple CUDA streams per partition on the corresponding green contexts.

    Args:
        SM_a: Requested SM count for partition A (same meaning as create_greenctx_stream_by_value).
        SM_b: Requested SM count for partition B.
        n_streams_a: Number of streams to create on partition A (>= 1).
        n_streams_b: Number of streams to create on partition B (>= 1).
        device_id: CUDA device ordinal.

    Returns:
        streams_a: Tuple of ExternalStream for partition A.
        streams_b: Tuple of ExternalStream for partition B.
        actual_sm_a: SM count provisioned for partition A after driver alignment/split.
        actual_sm_b: SM count provisioned for partition B.
    """
    if _spatial_import_error is not None:
        raise _IMPORT_ERROR from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    if n_streams_a < 1 or n_streams_b < 1:
        raise ValueError("n_streams_a and n_streams_b must be >= 1")

    res = torch.ops.sgl_kernel.create_greenctx_streams_by_value_enhanced(
        SM_a, SM_b, n_streams_a, n_streams_b, device_id
    )
    dev = torch.device(f"cuda:{device_id}")
    na = int(n_streams_a)
    nb = int(n_streams_b)
    streams_a = tuple(
        ExternalStream(stream_ptr=res[i], device=dev) for i in range(na)
    )
    streams_b = tuple(
        ExternalStream(stream_ptr=res[na + i], device=dev) for i in range(nb)
    )
    actual_sm_a = int(res[-2])
    actual_sm_b = int(res[-1])
    return streams_a, streams_b, actual_sm_a, actual_sm_b


def get_sm_available(device_id: int = None) -> int:
    """
    Get the SMs available on the device.
    Args:
        device_id (int): The device id.
    Returns:
        int: The SMs available.
    """
    if _spatial_import_error is not None:
        raise _IMPORT_ERROR from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    device_props = torch.cuda.get_device_properties(device_id)

    # Get the number of Streaming Multiprocessors (SMs)
    sm_count = device_props.multi_processor_count

    return sm_count
