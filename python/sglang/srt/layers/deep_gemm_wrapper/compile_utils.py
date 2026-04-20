import logging
import os
from contextlib import contextmanager
from enum import IntEnum, auto
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    disable_symmetric_memory_context,
    restore_symmetric_memory_context,
)
from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import ceil_div, get_available_gpu_memory

logger = logging.getLogger(__name__)

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm


_BUILTIN_M_LIST = list(range(1, 1024 * 16 + 1))
_ENABLE_JIT_DEEPGEMM_PRECOMPILE = envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.get()
_DO_COMPILE_ALL = True
_IS_FIRST_RANK_ON_NODE = envs.SGLANG_IS_FIRST_RANK_ON_NODE.get()
_IN_PRECOMPILE_STAGE = envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get()
# PDMux: JIT warmup over every distinct prefill SM (see pdmux_context.prefill_sm_counts_for_deepgemm_warmup)
_ENABLE_PDMUX_DEEGEMM_COMPILE_WARMUP = False
_DEEGEMM_WARMUP_GPU_ID = 0

# Force redirect deep_gemm cache_dir
os.environ["DG_JIT_CACHE_DIR"] = os.getenv(
    "SGLANG_DG_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "deep_gemm")
)

# Refer to https://github.com/deepseek-ai/DeepGEMM/commit/d75b218b7b8f4a5dd5406ac87905039ead3ae42f
# NVRTC may have performance loss with some cases.
# And NVCC JIT speed is also 9x faster in the ref commit
os.environ["DG_JIT_USE_NVRTC"] = os.getenv("SGL_DG_USE_NVRTC", "0")


def update_deep_gemm_config(gpu_id: int, server_args: ServerArgs):
    global _BUILTIN_M_LIST
    global _DO_COMPILE_ALL
    global _IS_FIRST_RANK_ON_NODE
    global _ENABLE_PDMUX_DEEGEMM_COMPILE_WARMUP
    global _DEEGEMM_WARMUP_GPU_ID

    # Generate m_max
    m_max = 1024 * 16
    if server_args.chunked_prefill_size < 1:
        m_max = 1024 * 64
    elif server_args.chunked_prefill_size > 8192:
        m_max = server_args.chunked_prefill_size * 2
    m_max = min(1024 * 128, m_max)
    _BUILTIN_M_LIST = list(range(1, m_max + 1))

    _IS_FIRST_RANK_ON_NODE = server_args.base_gpu_id == gpu_id

    # Check if is the first rank on node.
    # Default each rank will try compile all Ms to
    # load all symbols at the launch stages.
    # Avoid loading symbols at the serving stages.
    _DO_COMPILE_ALL = _IS_FIRST_RANK_ON_NODE

    _ENABLE_PDMUX_DEEGEMM_COMPILE_WARMUP = (
        getattr(server_args, "enable_pdmux_deepgemm_compile_warmup", False)
    )
    _DEEGEMM_WARMUP_GPU_ID = gpu_id


def _deepgemm_warmup_num_sms_list() -> List[int]:
    """Return DeepGEMM ``num_sms`` values to run compile-only warmup for."""
    if not ENABLE_JIT_DEEPGEMM:
        return []
    default_sms = int(deep_gemm.get_num_sms())
    if not _ENABLE_PDMUX_DEEGEMM_COMPILE_WARMUP:
        return [default_sms]
    try:
        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
    except Exception:
        return [default_sms]
    if not getattr(server_args, "enable_pdmux", False):
        return [default_sms]
    try:
        from sglang.srt.multiplex.pdmux_context import (
            load_pdmux_config,
            prefill_sm_counts_for_deepgemm_warmup,
        )

        cfg = load_pdmux_config(server_args.pdmux_config_path or "")
        sms = prefill_sm_counts_for_deepgemm_warmup(_DEEGEMM_WARMUP_GPU_ID, cfg)
    except Exception as e:
        logger.warning(
            "PDMux DeepGEMM warmup: failed to compute prefill SM list (%s); "
            "using default num_sms=%s",
            e,
            default_sms,
        )
        return [default_sms]
    if not sms:
        return [default_sms]
    return sms


class DeepGemmKernelType(IntEnum):
    GROUPED_GEMM_NT_F8F8BF16_MASKED = auto()
    GROUPED_GEMM_NT_F8F8BF16_CONTIG = auto()
    GEMM_NT_F8F8BF16 = auto()


_INITIALIZATION_DICT: Dict[Tuple[DeepGemmKernelType, int, int, int], bool] = dict()


def _log_deepgemm_jit_precompile_debug(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    """Log env + module state once when the JIT precompile branch runs (per query_key)."""
    parts = [
        "DeepGEMM JIT precompile debug snapshot",
        f"query=({kernel_type.name}, N={n}, K={k}, num_groups={num_groups})",
        f"ENABLE_JIT_DEEPGEMM={ENABLE_JIT_DEEPGEMM}",
        f"module: _ENABLE_JIT_DEEPGEMM_PRECOMPILE={_ENABLE_JIT_DEEPGEMM_PRECOMPILE} "
        f"_DO_COMPILE_ALL={_DO_COMPILE_ALL} _IS_FIRST_RANK_ON_NODE={_IS_FIRST_RANK_ON_NODE} "
        f"_IN_PRECOMPILE_STAGE={_IN_PRECOMPILE_STAGE}",
        f"module: _ENABLE_PDMUX_DEEGEMM_COMPILE_WARMUP={_ENABLE_PDMUX_DEEGEMM_COMPILE_WARMUP} "
        f"_DEEGEMM_WARMUP_GPU_ID={_DEEGEMM_WARMUP_GPU_ID}",
        f"m_list: len={len(_BUILTIN_M_LIST)} max_m={_BUILTIN_M_LIST[-1] if _BUILTIN_M_LIST else 0}",
    ]
    # Effective envs (may differ from os.environ if unset)
    for env_field in (
        envs.SGLANG_ENABLE_JIT_DEEPGEMM,
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE,
        envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE,
        envs.SGLANG_IS_FIRST_RANK_ON_NODE,
        envs.SGLANG_DG_CACHE_DIR,
        envs.SGLANG_DG_USE_NVRTC,
    ):
        parts.append(f"envs.{env_field.name}.get()={env_field.get()!r}")
    # Process env strings often used for DeepGEMM JIT
    for key in (
        "SGLANG_ENABLE_JIT_DEEPGEMM",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE",
        "SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE",
        "SGLANG_IS_FIRST_RANK_ON_NODE",
        "SGLANG_DG_CACHE_DIR",
        "SGL_DG_USE_NVRTC",
        "DG_JIT_DEBUG",
        "DG_PRINT_CONFIGS",
        "DG_JIT_PRINT_COMPILER_COMMAND",
        "DG_JIT_CACHE_DIR",
        "DG_JIT_USE_NVRTC",
    ):
        if key in os.environ:
            parts.append(f"os.environ[{key!r}]={os.environ[key]!r}")
    if torch.cuda.is_available():
        parts.append(
            f"cuda: device={torch.cuda.current_device()} "
            f"cap={torch.cuda.get_device_capability()} "
            f"name={torch.cuda.get_device_name()!r}"
        )
    try:
        from sglang.srt.server_args import get_global_server_args

        sa = get_global_server_args()
        parts.append(
            f"server_args: enable_pdmux={getattr(sa, 'enable_pdmux', None)} "
            f"pdmux_config_path={getattr(sa, 'pdmux_config_path', None)!r} "
            f"enable_pdmux_deepgemm_compile_warmup="
            f"{getattr(sa, 'enable_pdmux_deepgemm_compile_warmup', None)} "
            f"base_gpu_id={getattr(sa, 'base_gpu_id', None)}"
        )
    except Exception as e:
        parts.append(f"server_args: unavailable ({e})")
    if ENABLE_JIT_DEEPGEMM:
        try:
            parts.append(
                f"deep_gemm: get_num_sms={deep_gemm.get_num_sms()} "
                f"get_compile_mode={deep_gemm.get_compile_mode()}"
            )
            parts.append(f"warmup_num_sms_list(preflight)={_deepgemm_warmup_num_sms_list()}")
        except Exception as e:
            parts.append(f"deep_gemm introspection failed: {e}")
    parts.append(
        f"os.environ effective DG_JIT_CACHE_DIR={os.environ.get('DG_JIT_CACHE_DIR')!r}"
    )
    logger.info("%s", " | ".join(parts))


# TODO improve code
def _maybe_compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
) -> None:
    global _INITIALIZATION_DICT
    global _BUILTIN_M_LIST

    query_key = (kernel_type, n, k, num_groups)
    if (
        _ENABLE_JIT_DEEPGEMM_PRECOMPILE
        and _DO_COMPILE_ALL
        and _INITIALIZATION_DICT.get(query_key) is None
    ):
        _INITIALIZATION_DICT[query_key] = True

        _log_deepgemm_jit_precompile_debug(kernel_type, n, k, num_groups)

        # TODO maybe improve logs
        if not _IN_PRECOMPILE_STAGE and _IS_FIRST_RANK_ON_NODE:
            logger.warning(
                "Entering DeepGEMM JIT Pre-Compile session. "
                "It may take a long time (typically 10-20 mins) "
                "if you have not run `sglang.compile_deep_gemm`. "
                "It is recommended to run `sglang.compile_deep_gemm` with same args as `sglang.launch_server`"
                " for pre-compilation to reduce the overhead if you have not run it before. "
                "For example: "
                "`python3 -m sglang.compile_deep_gemm --model deepseek-ai/DeepSeek-V3 --tp 8 --trust-remote-code`"
            )

        logger.info(
            f"Try DeepGEMM JIT Compiling for "
            f"<{kernel_type.name}> N={n}, K={k}, num_groups={num_groups} with all Ms."
            f"{' It only takes a little time (typically 1 sec) if you have run `python3 -m sglang.compile_deep_gemm`. ' if not _IN_PRECOMPILE_STAGE else ''}"
        )

        _compile_deep_gemm_one_type_all(
            kernel_type=kernel_type,
            n=n,
            k=k,
            num_groups=num_groups,
            m_list=_BUILTIN_M_LIST,
        )


# NOTE(alcanderian): get_num_sms should be change when 2-batch-overlap is introduced
def _compile_deep_gemm_one_type_all(
    kernel_type: DeepGemmKernelType,
    n: int,
    k: int,
    num_groups: int,
    m_list: List[int],
) -> None:
    # Symmetric memory allocation performs a collective operation across all the GPUs.
    # Temporary disable symmetric memory during compilation since it only runs on the first rank.
    saved_context = disable_symmetric_memory_context()
    try:
        if kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            m_alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
            m_list = sorted(list(set(m for m in m_list if m % m_alignment == 0)))

        # Here the precompilation is only run on the first rank, so gpu_id should be 0
        memory_budget = get_available_gpu_memory(device="cuda", gpu_id=0)

        # If the memory budget is less memory requirement, we need to reduce max_m to avoid out of memory, which might further cause hanging during warmup
        max_m = max(m_list)
        required_memory = _BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )
        logger.info(
            f"Required memory for warmup: {required_memory}GB, Available memory: {memory_budget}GB"
        )
        if memory_budget < required_memory:
            # TODO: Maybe compute the max_m based on the memory budget
            while (
                _BaseWarmupExecutor.get_memory_requirement(
                    kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
                )
                > memory_budget
                and max_m > 4096
            ):
                max_m = max_m // 2
            logger.warning(
                f"Available memory {memory_budget}GB is less than required memory {required_memory}GB for warmup, reducing max_m to {max_m} to avoid out of memory"
            )
            m_list = [m for m in m_list if m <= max_m]

        # Need some methods to estimate needed memory for warmup
        executor = _BaseWarmupExecutor.create(
            kernel_type, max_m=max_m, n=n, k=k, num_groups=num_groups
        )

        num_sms_list = _deepgemm_warmup_num_sms_list()
        logger.info(f"DeepGEMM compile-only warmup: num_sms_list: {num_sms_list}")
        if not num_sms_list:
            num_sms_list = [int(deep_gemm.get_num_sms())]
        if len(num_sms_list) > 1:
            logger.info(
                "DeepGEMM compile-only warmup: iterating num_sms=%s "
                "(enable_pdmux_deepgemm_compile_warmup)",
                num_sms_list,
            )

        saved_num_sms = deep_gemm.get_num_sms()
        old_compile_mode = deep_gemm.get_compile_mode()
        try:
            deep_gemm.set_compile_mode(1)
            for num_sms in num_sms_list:
                deep_gemm.set_num_sms(num_sms)
                for m in tqdm(
                    m_list,
                    desc=f"DeepGEMM warmup num_sms={num_sms}",
                ):
                    executor.execute(m=m)
        finally:
            deep_gemm.set_compile_mode(old_compile_mode)
            deep_gemm.set_num_sms(saved_num_sms)

        # clean up input buffers
        torch.cuda.current_stream().synchronize()
        del executor
        torch.cuda.empty_cache()
    finally:
        # Restore symmetric memory context
        restore_symmetric_memory_context(saved_context)


class _BaseWarmupExecutor:
    @staticmethod
    def create(kernel_type: DeepGemmKernelType, **kwargs):
        return {
            DeepGemmKernelType.GEMM_NT_F8F8BF16: _NormalWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG: _GroupedContWarmupExecutor,
            DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED: _GroupedMaskedWarmupExecutor,
        }[kernel_type](**kwargs)

    @staticmethod
    def get_memory_requirement(
        kernel_type: DeepGemmKernelType, max_m: int, n: int, k: int, num_groups: int
    ) -> int:
        # Return the required memory space in GB for warmup executor
        _GB = 1 << 30
        if kernel_type == DeepGemmKernelType.GEMM_NT_F8F8BF16:
            return (max_m * k + n * k + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG:
            return (max_m * k + num_groups * n * k + max_m * 4 + max_m * n * 2) / _GB
        elif kernel_type == DeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED:
            return (
                num_groups * max_m * k
                + num_groups * n * k
                + num_groups * 4
                + num_groups * max_m * n * 2
            ) / _GB
        else:
            raise ValueError(f"Invalid kernel type: {kernel_type}")

    def execute(self, m):
        raise NotImplementedError


def _empty_token_fp8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(k, _BLOCK_SIZE)), device="cuda", dtype=torch.float32
        ),
    )


def _empty_block_fp8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, ceil_div(n, _BLOCK_SIZE), ceil_div(k, _BLOCK_SIZE)),
            device="cuda",
            dtype=torch.float32,
        ),
    )


_BLOCK_SIZE = 128


class _NormalWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.fp8_gemm_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expect_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


@contextmanager
def deep_gemm_execution_hook(
    m: int, n: int, k: int, num_groups: int, kernel_type: DeepGemmKernelType
):
    if m > 0:
        _maybe_compile_deep_gemm_one_type_all(kernel_type, n, k, num_groups)
    yield
