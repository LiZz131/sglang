"""
Benchmark PDMUX + special-DP-attention latency for prefill and decode.

Measures four quantities per iteration, using CUDA events for GPU timing
and time.perf_counter for CPU timing:
  - prefill launch  (CPU time to submit kernels)
  - prefill run     (GPU execution time)
  - decode launch   (CPU time to submit cudaGraphLaunch)
  - decode run      (GPU execution time)

Results are written as JSON for downstream modeling.

# Usage (single GPU, dummy weights):
CUDA_VISIBLE_DEVICES=1 \
python -m sglang.bench_one_batch_pdmux \
    --model-path /sgl-workspace/data/DeepSeek-V3.1-mini --load-format dummy \
    --cuda-graph-max-bs 128 \
    --mem-fraction-static 0.8 \
    --batch-size 1 4 --input-len 128 512 --output-len 8 \
    --prefill-sm 80 --decode-sm 52

# Usage (2-GPU TP=2, DeepSeek-V3):
CUDA_VISIBLE_DEVICES=1,2 python -m sglang.bench_one_batch_pdmux \
    --model-path /sgl-workspace/data/DeepSeek-V3.1-mid --tp 2 --dp 2 \
    --load-format dummy \
    --enable-dp-attention --enable-special-dp-attention \
    --enable-special-dp-attention-prefix-0 \
    --enable-save-kv-cache-for-dp \
    --cuda-graph-max-bs 128 \
    --mem-fraction-static 0.8 \
    --enable-pdmux \
    --stage prefill \
    --prefill-sm 80 --decode-sm 52 \
    --stream-group-idx 1 \
    --batch-size 1 8 16 --input-len 128 512 --output-len 16

CUDA_VISIBLE_DEVICES=1,2 python -m sglang.bench_one_batch_pdmux \
    --model-path /sgl-workspace/data/DeepSeek-V3.1-mid --tp 2 --dp 2 \
    --load-format dummy \
    --enable-dp-attention --enable-special-dp-attention \
    --enable-special-dp-attention-prefix-0 \
    --enable-save-kv-cache-for-dp \
    --cuda-graph-max-bs 128 \
    --mem-fraction-static 0.8 \
    --enable-pdmux \
    --stage decode \
    --prefill-sm 80 --decode-sm 52 \
    --stream-group-idx 1 \
    --batch-size 1 8 16 --input-len 128 512 --output-len 16
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from itertools import chain
import logging
import multiprocessing
import os
import re
import time
from types import SimpleNamespace
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.layers.dp_attention import get_attention_dp_rank, get_attention_dp_size
from sglang.srt.multiplex.pdmux_context import (
    PDMuxConfig,
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
    set_current_stream_idx,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

logger = logging.getLogger(__name__)

# bench_pdmux 入口根据 --debug-batch 置 True
_DEBUG_BATCH = False


def dp_rank_for_schedule_batch(model_runner: ModelRunner) -> Optional[int]:
    """与 Scheduler 一致：special-dp-attention 下 ScheduleBatch.dp_rank 用于筛选 dp_local req。"""
    if model_runner.server_args.enable_special_dp_attention:
        return get_attention_dp_rank()
    return None


def narrow_schedule_batch_to_dp_local_prefill(
    batch: ScheduleBatch, model_runner: ModelRunner
) -> None:
    """
    enable_save_kv_cache_for_dp + special_dp 时，alloc 只为本地 req 分配 KV，out_cache_loc 长度等于
    dp_local_extend_num_tokens，但 prepare_for_extend 仍保留全局 input_ids / extend_num_tokens。
    收窄 batch，使 extend_num_tokens 与 out_cache_loc 一致，避免 prepare_mlp_sync_batch 将 out_cache_loc
    pad 到全局 token 数导致 deepseek_v2 断言失败。

    为何不用 ScheduleBatch.filter_batch(keep_indices=dp_local_req_indices)：
    multiplexing_mixin 里是在 prefill forward 结束、process_batch_result 之后才 filter，此时会
    out_cache_loc=None 仍可合并进 running_batch。bench 在「同一次」extend 里就要 forward，必须保留
    out_cache_loc；filter_batch 会清空 out_cache_loc 且不重建与本地 token 对齐的 input_ids，故不能
    直接替代；本函数等价于在 forward 前只保留本地 req 的张量形态。
    """
    sa = model_runner.server_args
    if not sa.enable_save_kv_cache_for_dp or not sa.enable_special_dp_attention:
        return
    if batch.dp_local_req_indices is None:
        return
    if len(batch.dp_local_req_indices) == len(batch.reqs):
        return
    if batch.return_logprob:
        raise NotImplementedError(
            "narrow_schedule_batch_to_dp_local_prefill does not support return_logprob"
        )

    local_idx = list(batch.dp_local_req_indices)
    local_reqs = batch.dp_local_reqs
    assert len(local_reqs) > 0, "narrow_schedule_batch_to_dp_local_prefill: empty dp_local_reqs"

    batch.reqs = list(local_reqs)
    device = batch.device
    # torch.tensor 不支持 non_blocking；与 schedule_batch.prepare_for_extend 一致先建 CPU 再 .to
    batch.input_ids = torch.tensor(
        list(chain.from_iterable(batch.dp_local_input_ids)),
        dtype=torch.int64,
    ).to(device, non_blocking=True)
    batch.seq_lens = batch.dp_local_seq_lens
    batch.seq_lens_cpu = batch.dp_local_seq_lens_cpu
    batch.orig_seq_lens = batch.dp_local_orig_seq_lens
    batch.prefix_lens = list(batch.dp_local_prefix_lens)
    batch.extend_lens = list(batch.dp_local_extend_lens)
    batch.extend_num_tokens = batch.dp_local_extend_num_tokens
    batch.seq_lens_sum = int(batch.seq_lens.sum().item())

    batch.dp_local_token_start = 0
    batch.dp_local_token_end = batch.dp_local_extend_num_tokens
    batch.dp_local_req_indices = list(range(len(local_reqs)))

    token_type_ids_local = [
        r.token_type_ids for r in batch.reqs if r.token_type_ids is not None
    ]
    if len(token_type_ids_local) > 0:
        batch.token_type_ids = torch.tensor(
            sum(token_type_ids_local, []), dtype=torch.int64
        ).to(device, non_blocking=True)
    else:
        batch.token_type_ids = None

    dims = getattr(batch, "dimensions", None)
    if batch.model_config.is_matryoshka and dims is not None:
        batch.dimensions = [dims[i] for i in local_idx]

    mm_inputs = getattr(batch, "multimodal_inputs", None)
    if mm_inputs:
        batch.multimodal_inputs = [mm_inputs[i] for i in local_idx]

    batch.extend_logprob_start_lens = [r.extend_logprob_start_len for r in batch.reqs]

    if batch.mamba_track_indices is not None:
        batch.mamba_track_indices = batch.mamba_track_indices[local_idx]
        batch.mamba_track_mask = batch.mamba_track_mask[local_idx]
        batch.mamba_track_seqlens = batch.mamba_track_seqlens[local_idx]

    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch, batch.model_config.vocab_size
    )


def _log_special_dp_extend(
    batch: ScheduleBatch,
    forward_batch: Optional[ForwardBatch],
    tag: str,
) -> None:
    """排查 dp_local / out_cache_loc 不一致时打开（--debug-batch）。"""
    if not _DEBUG_BATCH:
        return
    logger.info(
        "[%s] dp_rank=%s attn_dp_rank=%s attn_dp_size=%s "
        "dp_local_req_indices=%s dp_local_token_start=%s dp_local_token_end=%s "
        "extend_num_tokens=%s out_cache_loc_len=%s enable_save_kv_for_dp=%s",
        tag,
        batch.dp_rank,
        get_attention_dp_rank(),
        get_attention_dp_size(),
        batch.dp_local_req_indices,
        batch.dp_local_token_start,
        batch.dp_local_token_end,
        batch.extend_num_tokens,
        None if batch.out_cache_loc is None else int(batch.out_cache_loc.shape[0]),
        batch.enable_save_kv_cache_for_dp,
    )
    if forward_batch is not None:
        logger.info(
            "[%s] ForwardBatch: dp_local_token_start=%s dp_local_token_end=%s "
            "out_cache_loc_len=%s forward_mode=%s batch_size=%s",
            tag,
            getattr(forward_batch, "dp_local_token_start", None),
            getattr(forward_batch, "dp_local_token_end", None),
            None
            if getattr(forward_batch, "out_cache_loc", None) is None
            else int(forward_batch.out_cache_loc.shape[0]),
            getattr(forward_batch, "forward_mode", None),
            getattr(forward_batch, "batch_size", None),
        )


# ---------------------------------------------------------------------------
# Bench args
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PdmuxBenchArgs:
    run_name: str = "pdmux_bench"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "pdmux_bench_result.json"
    warmup_steps: int = 5
    bench_steps: int = 20
    # SM split: if pdmux-config-path is NOT given, create a 3-group config
    prefill_sm: int = 0
    decode_sm: int = 0
    # which stream group to benchmark (0..N-1)
    stream_group_idx: int = 1
    # which stage to benchmark
    stage: str = "both"  # prefill / decode / both
    # 打印 special-dp / dp_local / out_cache_loc 等（便于对照 deepseek_v2 断言）
    debug_batch: bool = False
    # --- randomness ---
    random_seed: int = 0
    # intra-batch prompt length randomness (per-request origin_input_ids length)
    input_len_mode: str = "fixed"  # fixed / uniform / loguniform / zipf
    input_len_min: int = 0  # 0 => use base input_len
    input_len_max: int = 0  # 0 => use base input_len
    zipf_s: float = 1.2
    # cross-DP randomness via decode_dp_rank assignment (only meaningful for special DP attention)
    dp_assign_mode: str = "uniform"  # uniform / counts
    dp_counts: str = ""  # e.g. "5,9" for dp_size=2, sum must equal batch_size
    dp_len_multipliers: str = ""  # optional per-dp multiplier, e.g. "1.0,1.3"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=PdmuxBenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=PdmuxBenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=PdmuxBenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=PdmuxBenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=PdmuxBenchArgs.result_filename
        )
        parser.add_argument(
            "--warmup-steps", type=int, default=PdmuxBenchArgs.warmup_steps
        )
        parser.add_argument(
            "--bench-steps", type=int, default=PdmuxBenchArgs.bench_steps
        )
        parser.add_argument(
            "--prefill-sm",
            type=int,
            default=PdmuxBenchArgs.prefill_sm,
            help="prefill green context 的 SM 数量（推荐直接传；若为 0 则回退到 --pdmux-config-path）。",
        )
        parser.add_argument(
            "--decode-sm",
            type=int,
            default=PdmuxBenchArgs.decode_sm,
            help="decode green context 的 SM 数量（推荐直接传；若为 0 则回退到 --pdmux-config-path）。",
        )
        parser.add_argument(
            "--stream-group-idx",
            type=int,
            default=PdmuxBenchArgs.stream_group_idx,
            help="Stream group index to benchmark.",
        )
        parser.add_argument(
            "--stage",
            type=str,
            default=PdmuxBenchArgs.stage,
            choices=["prefill", "decode", "both", "p", "d"],
            help="Benchmark stage: prefill only / decode only / both. (p,d are aliases)",
        )
        parser.add_argument(
            "--debug-batch",
            action="store_true",
            help="Log ScheduleBatch/ForwardBatch dp_local 与 out_cache_loc 等字段（排查 special-dp）",
        )
        parser.add_argument(
            "--bench-random-seed",
            type=int,
            dest="random_seed",
            default=PdmuxBenchArgs.random_seed,
            help="Random seed for synthetic batch generation.",
        )
        parser.add_argument(
            "--bench-input-len-mode",
            type=str,
            dest="input_len_mode",
            default=PdmuxBenchArgs.input_len_mode,
            choices=["fixed", "uniform", "loguniform", "zipf"],
            help="Randomize per-request prompt length inside a batch.",
        )
        parser.add_argument(
            "--bench-input-len-min",
            type=int,
            dest="input_len_min",
            default=PdmuxBenchArgs.input_len_min,
            help="Min prompt length for randomization (0 => use base --input-len).",
        )
        parser.add_argument(
            "--bench-input-len-max",
            type=int,
            dest="input_len_max",
            default=PdmuxBenchArgs.input_len_max,
            help="Max prompt length for randomization (0 => use base --input-len).",
        )
        parser.add_argument(
            "--bench-zipf-s",
            type=float,
            dest="zipf_s",
            default=PdmuxBenchArgs.zipf_s,
            help="Zipf exponent (only used when --input-len-mode=zipf).",
        )
        parser.add_argument(
            "--bench-dp-assign-mode",
            type=str,
            dest="dp_assign_mode",
            default=PdmuxBenchArgs.dp_assign_mode,
            choices=["uniform", "counts"],
            help="How to assign decode_dp_rank for requests (special DP attention).",
        )
        parser.add_argument(
            "--bench-dp-counts",
            type=str,
            dest="dp_counts",
            default=PdmuxBenchArgs.dp_counts,
            help='Per-DP request counts, e.g. \"5,9\" (sum must equal batch_size). Used when --dp-assign-mode=counts.',
        )
        parser.add_argument(
            "--bench-dp-len-multipliers",
            type=str,
            dest="dp_len_multipliers",
            default=PdmuxBenchArgs.dp_len_multipliers,
            help='Optional per-DP prompt length multipliers, e.g. \"1.0,1.3\" (len==dp_size).',
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


# ---------------------------------------------------------------------------
# Green context / stream group helpers
# ---------------------------------------------------------------------------


def setup_pdmux_streams(
    gpu_id: int, server_args: ServerArgs, bench_args: PdmuxBenchArgs
):
    """Initialize green context stream groups before ModelRunner is created."""
    # bench 场景通常只需要固定一组 (prefill_sm, decode_sm)，无需 pdmux_config。
    # 若用户显式给了 --prefill-sm/--decode-sm，则优先使用它们；否则才回退到 pdmux_config_path。
    prefill_sm = bench_args.prefill_sm
    decode_sm = bench_args.decode_sm
    if prefill_sm > 0 and decode_sm > 0:
        config = PDMuxConfig(
            sm_group_num=3,
            manual_divisions=[[prefill_sm, decode_sm, 0]],
        )
    elif server_args.pdmux_config_path:
        config = load_pdmux_config(server_args.pdmux_config_path)
    else:
        raise ValueError("请提供 --prefill-sm/--decode-sm（推荐），或使用 --pdmux-config-path。")
    server_args.sm_group_num = config.sm_group_num
    initialize_stream_groups(gpu_id, config)
    return config


# ---------------------------------------------------------------------------
# Model loading (adapted from bench_one_batch)
# ---------------------------------------------------------------------------


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def _parse_int_list(s: str) -> list[int]:
    if not s:
        return []
    return [int(x) for x in re.split(r"\s*,\s*", s.strip()) if x != ""]


def _parse_float_list(s: str) -> list[float]:
    if not s:
        return []
    return [float(x) for x in re.split(r"\s*,\s*", s.strip()) if x != ""]


def _assign_decode_dp_ranks(
    rng: np.random.Generator,
    *,
    batch_size: int,
    dp_size: int,
    mode: str,
    dp_counts: str,
) -> list[int]:
    if dp_size <= 1:
        return [0 for _ in range(batch_size)]
    if mode == "uniform":
        ranks = rng.integers(0, dp_size, size=batch_size, dtype=np.int32).tolist()
        # Avoid empty dp_local on any rank. In special-dp + save-kv-cache, an empty dp_local
        # will break the prefill path (out_cache_loc is local-only). We still allow asymmetry.
        if batch_size >= dp_size:
            for r in range(dp_size):
                if r not in ranks:
                    # Overwrite an entry (deterministic but random index) to ensure coverage.
                    idx = int(rng.integers(0, batch_size))
                    ranks[idx] = r
        return ranks
    if mode == "counts":
        counts = _parse_int_list(dp_counts)
        if len(counts) != dp_size:
            raise ValueError(f"--dp-counts must have len==dp_size ({dp_size}), got {counts}")
        if sum(counts) != batch_size:
            raise ValueError(f"--dp-counts sum must equal batch_size ({batch_size}), got {counts}")
        if batch_size >= dp_size and any(c <= 0 for c in counts):
            raise ValueError(
                f"--dp-counts must be positive for each dp when batch_size>=dp_size; got {counts}"
            )
        ranks: list[int] = []
        for r, c in enumerate(counts):
            ranks.extend([r] * c)
        rng.shuffle(ranks)
        return ranks
    raise ValueError(f"Unknown --dp-assign-mode: {mode}")


def _sample_prompt_lens(
    rng: np.random.Generator,
    *,
    batch_size: int,
    base_input_len: int,
    mode: str,
    len_min: int,
    len_max: int,
    zipf_s: float,
    dp_ranks: list[int],
    dp_len_multipliers: list[float],
) -> list[int]:
    lo = len_min if len_min > 0 else base_input_len
    hi = len_max if len_max > 0 else base_input_len
    if lo <= 0 or hi <= 0:
        raise ValueError(f"Invalid input length bounds: lo={lo}, hi={hi}")
    if lo > hi:
        raise ValueError(f"Invalid input length bounds: lo={lo} > hi={hi}")

    if mode == "fixed":
        lens = [base_input_len for _ in range(batch_size)]
    elif mode == "uniform":
        lens = rng.integers(lo, hi + 1, size=batch_size, dtype=np.int32).tolist()
    elif mode == "loguniform":
        if lo == hi:
            lens = [lo for _ in range(batch_size)]
        else:
            u = rng.uniform(np.log(lo), np.log(hi + 1e-9), size=batch_size)
            lens = np.clip(np.exp(u).astype(np.int32), lo, hi).tolist()
    elif mode == "zipf":
        if lo == hi:
            lens = [lo for _ in range(batch_size)]
        else:
            span = hi - lo + 1
            z = rng.zipf(zipf_s, size=batch_size).astype(np.int64)
            lens = (lo + (z % span)).astype(np.int32).tolist()
    else:
        raise ValueError(f"Unknown --input-len-mode: {mode}")

    if dp_len_multipliers:
        for i in range(batch_size):
            r = int(dp_ranks[i])
            if 0 <= r < len(dp_len_multipliers):
                lens[i] = int(max(1, round(lens[i] * float(dp_len_multipliers[r]))))
    return lens


def prepare_synthetic_reqs(
    *,
    batch_size: int,
    base_input_len: int,
    bench_args: PdmuxBenchArgs,
    model_runner: ModelRunner,
    rng: np.random.Generator,
) -> tuple[list[Req], dict]:
    """Return (reqs, meta). Meta contains the realized randomness for modeling."""
    sampling_params = SamplingParams(temperature=0, max_new_tokens=1)

    dp_size = get_attention_dp_size()
    enable_special = bool(model_runner.server_args.enable_special_dp_attention)

    if enable_special and dp_size > 0:
        dp_ranks = _assign_decode_dp_ranks(
            rng,
            batch_size=batch_size,
            dp_size=dp_size,
            mode=bench_args.dp_assign_mode,
            dp_counts=bench_args.dp_counts,
        )
        # bs==1 时若固定为 0，则在 attn_dp_rank!=0 的进程上无 dp_local；因此 bs==1 强制落本 rank
        if batch_size == 1:
            dp_ranks = [get_attention_dp_rank()]
    else:
        dp_size = max(int(model_runner.server_args.dp_size), 1)
        dp_ranks = [0 for _ in range(batch_size)]

    dp_len_multipliers = _parse_float_list(bench_args.dp_len_multipliers)
    if dp_len_multipliers and len(dp_len_multipliers) != dp_size:
        raise ValueError(
            f"--dp-len-multipliers must have len==dp_size ({dp_size}), got {dp_len_multipliers}"
        )

    prompt_lens = _sample_prompt_lens(
        rng,
        batch_size=batch_size,
        base_input_len=base_input_len,
        mode=bench_args.input_len_mode,
        len_min=bench_args.input_len_min,
        len_max=bench_args.input_len_max,
        zipf_s=bench_args.zipf_s,
        dp_ranks=dp_ranks,
        dp_len_multipliers=dp_len_multipliers,
    )

    reqs: list[Req] = []
    for i in range(batch_size):
        ilen = int(prompt_lens[i])
        toks = rng.integers(0, 10000, size=ilen, dtype=np.int32).tolist()
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=toks,
            sampling_params=sampling_params,
            decode_dp_rank=int(dp_ranks[i]) if enable_special else None,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    sums = [0 for _ in range(max(dp_size, 1))]
    for l, r in zip(prompt_lens, dp_ranks):
        sums[int(r)] += int(l)

    meta = {
        "req_prompt_lens": prompt_lens,
        "decode_dp_ranks": dp_ranks,
        "prompt_len_sum_per_dp": sums,
        "max_prompt_len_sum_per_dp": max(sums) if sums else 0,
    }
    return reqs, meta


def _make_dummy_tree_cache(model_runner):
    return SimpleNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )


def _make_idle_batch(model_runner: ModelRunner) -> ScheduleBatch:
    """Mirror SchedulerDPAttnMixin.get_idle_batch() for bench usage."""
    tree_cache = _make_dummy_tree_cache(model_runner)
    idle_batch = ScheduleBatch.init_new(
        [],
        model_runner.req_to_token_pool,
        model_runner.token_to_kv_pool_allocator,
        tree_cache,
        model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        dp_rank=dp_rank_for_schedule_batch(model_runner),
    )
    idle_batch.prepare_for_idle()
    return idle_batch


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner: ModelRunner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=lambda: _make_idle_batch(model_runner),
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
            enable_special_dp_attention=model_runner.server_args.enable_special_dp_attention,
        )


# ---------------------------------------------------------------------------
# Low-level extend / decode  (stream-aware, event-instrumented)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _extend_once(reqs, model_runner, stream: Optional[torch.cuda.Stream] = None):
    """Run a single prefill (extend) and return (next_token_ids, batch)."""
    tree_cache = _make_dummy_tree_cache(model_runner)
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        dp_rank=dp_rank_for_schedule_batch(model_runner),
    )
    batch.prepare_for_extend()
    narrow_schedule_batch_to_dp_local_prefill(batch, model_runner)
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    _log_special_dp_extend(batch, forward_batch, "_extend_once")

    if stream is not None:
        with torch.cuda.stream(stream):
            logits_output = model_runner.forward(forward_batch).logits_output
            next_token_ids = model_runner.sample(logits_output, forward_batch)
    else:
        logits_output = model_runner.forward(forward_batch).logits_output
        next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, batch


@torch.no_grad()
def _decode_once(
    input_token_ids, batch, model_runner, stream: Optional[torch.cuda.Stream] = None
):
    """Run a single decode step and return next_token_ids."""
    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    _log_special_dp_extend(batch, forward_batch, "_decode_once")

    if stream is not None:
        with torch.cuda.stream(stream):
            logits_output = model_runner.forward(forward_batch).logits_output
            next_token_ids = model_runner.sample(logits_output, forward_batch)
    else:
        logits_output = model_runner.forward(forward_batch).logits_output
        next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids


# ---------------------------------------------------------------------------
# Timed helpers (CUDA events + CPU timer)
# ---------------------------------------------------------------------------


@torch.no_grad()
def timed_extend(
    reqs, model_runner, stream: torch.cuda.Stream
):
    """Prefill with timing.  Returns (next_token_ids, batch, timing_dict)."""
    tree_cache = _make_dummy_tree_cache(model_runner)
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        dp_rank=dp_rank_for_schedule_batch(model_runner),
    )
    batch.prepare_for_extend()
    sa = model_runner.server_args
    # special-dp + save-kv: this rank may have 0 dp_local reqs. In that case we must NOT run
    # the extend forward (no KV allocated); instead run an IDLE batch to participate in MLP sync.
    if (
        sa.enable_save_kv_cache_for_dp
        and sa.enable_special_dp_attention
        and (batch.dp_local_reqs is None or len(batch.dp_local_reqs) == 0)
    ):
        batch = _make_idle_batch(model_runner)
    else:
        narrow_schedule_batch_to_dp_local_prefill(batch, model_runner)
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    _log_special_dp_extend(batch, forward_batch, "timed_extend")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start_event.record(stream)
        cpu_start = time.perf_counter()

        logits_output = model_runner.forward(forward_batch).logits_output

        cpu_end = time.perf_counter()
        end_event.record(stream)

    stream.synchronize()

    gpu_time_ms = start_event.elapsed_time(end_event)
    cpu_launch_ms = (cpu_end - cpu_start) * 1000.0

    if batch.batch_size() > 0:
        next_token_ids = model_runner.sample(logits_output, forward_batch)
    else:
        next_token_ids = torch.empty(0, dtype=torch.int64, device=model_runner.device)

    timing = {
        "prefill_launch_ms": cpu_launch_ms,
        "prefill_run_ms": gpu_time_ms,
    }
    return next_token_ids, batch, timing


@torch.no_grad()
def timed_decode(
    input_token_ids, batch, model_runner, stream: torch.cuda.Stream
):
    """Decode with timing.

    batch_info 必须在 prepare_for_decode + prepare_mlp_sync 之后采集，与本次 forward 一致；
    若在循环里先于 timed_decode 调用 _batch_info_str_decode，第一步仍是 extend 态，
    seq_lens 为整段 prefill 长度，global_num_tokens 为 extend 填充（如 [128,128]），
    而非 decode 步的 [1,1]。
    """
    # special-dp: this rank may be idle (0 local req). In that case, run an IDLE batch to
    # participate in MLP sync rather than prepare_for_decode().
    if batch is None or batch.batch_size() == 0:
        idle = _make_idle_batch(model_runner)
        _maybe_prepare_mlp_sync_batch(idle, model_runner)
        decode_batch_info = _batch_info_str_decode(idle)
        model_worker_batch = idle.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
        _log_special_dp_extend(idle, forward_batch, "timed_decode_idle")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start_event.record(stream)
            cpu_start = time.perf_counter()
            _ = model_runner.forward(forward_batch).logits_output
            cpu_end = time.perf_counter()
            end_event.record(stream)
        stream.synchronize()
        timing = {
            "decode_launch_ms": (cpu_end - cpu_start) * 1000.0,
            "decode_run_ms": start_event.elapsed_time(end_event),
        }
        next_token_ids = torch.empty(0, dtype=torch.int64, device=model_runner.device)
        return next_token_ids, timing, decode_batch_info

    batch.output_ids = input_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    decode_batch_info = _batch_info_str_decode(batch)

    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    _log_special_dp_extend(batch, forward_batch, "timed_decode")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start_event.record(stream)
        cpu_start = time.perf_counter()

        logits_output = model_runner.forward(forward_batch).logits_output

        cpu_end = time.perf_counter()
        end_event.record(stream)

    stream.synchronize()

    gpu_time_ms = start_event.elapsed_time(end_event)
    cpu_launch_ms = (cpu_end - cpu_start) * 1000.0

    next_token_ids = model_runner.sample(logits_output, forward_batch)

    timing = {
        "decode_launch_ms": cpu_launch_ms,
        "decode_run_ms": gpu_time_ms,
    }
    return next_token_ids, timing, decode_batch_info


# ---------------------------------------------------------------------------
# Batch info string (matches NVTX format)
# ---------------------------------------------------------------------------


def _batch_info_str_prefill(reqs) -> str:
    lens = [len(req.fill_ids) - len(req.prefix_indices) for req in reqs]
    return "[" + ",".join(str(v) for v in lens) + "]"


def _batch_info_str_decode(batch: ScheduleBatch) -> str:
    slc = batch.seq_lens_cpu
    n = len(batch.reqs)
    if slc is not None and slc.numel() == n:
        vals = [int(x) for x in slc.tolist()]
    else:
        vals = [int(req.extend_input_len) for req in batch.reqs]
    local_str = "[" + ",".join(str(v) for v in vals) + "]"

    gnt = getattr(batch, "global_num_tokens", None)
    if gnt is not None and len(gnt) > 0:
        global_str = "[" + ",".join(str(int(x)) for x in gnt) + "]"
    else:
        global_str = "[]"

    gsspd = getattr(batch, "global_seq_lens_sum_per_dp", None)
    if gsspd is not None and len(gsspd) > 0:
        gsspd_str = "[" + ",".join(str(int(x)) for x in gsspd) + "]"
        return f"{local_str}|{global_str}|{gsspd_str}"
    return f"{local_str}|{global_str}"


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------


def bench_one_config(
    model_runner: ModelRunner,
    rank_print,
    batch_size: int,
    input_len: int,
    output_len: int,
    stream_group_idx: int,
    warmup_steps: int,
    bench_steps: int,
    bench_args: PdmuxBenchArgs,
    rng: np.random.Generator,
    tp_rank: int,
    stage: Literal["prefill", "decode", "both"],
):
    stream_groups = get_stream_groups()
    sm_counts = get_sm_counts()

    if stream_group_idx >= len(stream_groups):
        rank_print(
            f"stream_group_idx={stream_group_idx} >= len(stream_groups)="
            f"{len(stream_groups)}, skipping."
        )
        return None

    prefill_stream, decode_stream = stream_groups[stream_group_idx]
    prefill_sm, decode_sm = sm_counts[stream_group_idx]

    set_current_stream_idx(stream_group_idx)
    model_runner.update_decode_attn_backend(stream_group_idx)

    max_tokens = model_runner.max_total_num_tokens
    max_batch_size = max_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping (bs={batch_size}, il={input_len}, ol={output_len}) "
            f"due to max batch size limit ({max_batch_size})"
        )
        return None

    rank_print(
        f"\n{'='*60}\n"
        f"Config: bs={batch_size}, input_len={input_len}, output_len={output_len}\n"
        f"stream_group={stream_group_idx}, prefill_sm={prefill_sm}, decode_sm={decode_sm}\n"
        f"warmup={warmup_steps}, bench={bench_steps}\n"
        f"{'='*60}"
    )

    prefill_records = []
    decode_records = []
    total_steps = warmup_steps + bench_steps

    for step in range(total_steps):
        is_warmup = step < warmup_steps

        model_runner.req_to_token_pool.clear()
        model_runner.token_to_kv_pool_allocator.clear()

        # decode-only 也需要一个不计时的 prefill 来构建 KV cache / batch
        next_token_ids = None
        batch = None

        reqs, synth_meta = prepare_synthetic_reqs(
            batch_size=batch_size,
            base_input_len=input_len,
            bench_args=bench_args,
            model_runner=model_runner,
            rng=rng,
        )
        prefill_batch_info = _batch_info_str_prefill(reqs)

        if stage in ("prefill", "both"):
            next_token_ids, batch, prefill_timing = timed_extend(
                reqs, model_runner, prefill_stream
            )
            if not is_warmup:
                rec = {
                    "batch_info": prefill_batch_info,
                    "batch_size": batch_size,
                    "input_len": input_len,
                    **synth_meta,
                    **prefill_timing,
                }
                prefill_records.append(rec)
                if step == warmup_steps:
                    rank_print(
                        f"  prefill: launch={prefill_timing['prefill_launch_ms']:.3f} ms, "
                        f"run={prefill_timing['prefill_run_ms']:.3f} ms"
                    )
        else:
            next_token_ids, batch = _extend_once(reqs, model_runner, prefill_stream)
            torch.cuda.synchronize()

        if stage in ("decode", "both"):
            assert next_token_ids is not None and batch is not None
            for d_step in range(output_len):
                next_token_ids, decode_timing, decode_batch_info = timed_decode(
                    next_token_ids, batch, model_runner, decode_stream
                )

                if not is_warmup:
                    rec = {
                        "batch_info": decode_batch_info,
                        "batch_size": batch_size,
                        "decode_step": d_step,
                        **synth_meta,
                        **decode_timing,
                    }
                    decode_records.append(rec)

                    if step == warmup_steps and d_step < 3:
                        rank_print(
                            f"  decode[{d_step}]: launch={decode_timing['decode_launch_ms']:.3f} ms, "
                            f"run={decode_timing['decode_run_ms']:.3f} ms, "
                            f"info={decode_batch_info}"
                        )

    # ---- summary ----
    prefill_launch = [r["prefill_launch_ms"] for r in prefill_records]
    prefill_run = [r["prefill_run_ms"] for r in prefill_records]
    decode_launch = [r["decode_launch_ms"] for r in decode_records]
    decode_run = [r["decode_run_ms"] for r in decode_records]

    summary = {
        "run_name": "",
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "stream_group_idx": stream_group_idx,
        "prefill_sm": prefill_sm,
        "decode_sm": decode_sm,
        "prefill_launch_ms_median": float(np.median(prefill_launch)) if prefill_launch else 0,
        "prefill_run_ms_median": float(np.median(prefill_run)) if prefill_run else 0,
        "decode_launch_ms_median": float(np.median(decode_launch)) if decode_launch else 0,
        "decode_run_ms_median": float(np.median(decode_run)) if decode_run else 0,
        "prefill_launch_ms_mean": float(np.mean(prefill_launch)) if prefill_launch else 0,
        "prefill_run_ms_mean": float(np.mean(prefill_run)) if prefill_run else 0,
        "decode_launch_ms_mean": float(np.mean(decode_launch)) if decode_launch else 0,
        "decode_run_ms_mean": float(np.mean(decode_run)) if decode_run else 0,
    }
    rank_print(
        f"  SUMMARY: prefill_run={summary['prefill_run_ms_median']:.3f} ms (median), "
        f"decode_run={summary['decode_run_ms_median']:.3f} ms (median)"
    )

    return {
        "summary": summary,
        "prefill_records": prefill_records,
        "decode_records": decode_records,
    }


# ---------------------------------------------------------------------------
# Top-level driver per rank
# ---------------------------------------------------------------------------


def bench_pdmux(
    server_args: ServerArgs,
    port_args: PortArgs,
    bench_args: PdmuxBenchArgs,
    gpu_id: int,
    tp_rank: int,
):
    global _DEBUG_BATCH
    _DEBUG_BATCH = bench_args.debug_batch

    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # RNG: deterministic and rank-stable
    seed = int(bench_args.random_seed)
    rng = np.random.default_rng(seed + 1000 * tp_rank)

    # 1) Initialize green context streams BEFORE loading model
    pdmux_config = setup_pdmux_streams(gpu_id, server_args, bench_args)
    rank_print(
        f"Initialized stream groups: {len(get_stream_groups())} groups, "
        f"SM counts: {get_sm_counts()}"
    )

    # 2) Force pdmux flags so ModelRunner creates decode_attn_backend_group
    #    and CudaGraphRunner captures per-stream-group graphs.
    server_args.enable_pdmux = True

    # 3) Load model (this also captures CUDA graphs for all stream groups)
    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
    rank_print(f"Model loaded. graph_runner={'yes' if model_runner.graph_runner else 'no'}")

    # 4) Warm up once on the target stream group
    rank_print("Warming up ...")
    set_current_stream_idx(bench_args.stream_group_idx)
    model_runner.update_decode_attn_backend(bench_args.stream_group_idx)
    sg = get_stream_groups()[bench_args.stream_group_idx]
    warm_reqs, _ = prepare_synthetic_reqs(
        batch_size=bench_args.batch_size[0],
        base_input_len=bench_args.input_len[0],
        bench_args=bench_args,
        model_runner=model_runner,
        rng=rng,
    )
    next_ids, batch = _extend_once(warm_reqs, model_runner, sg[0])
    for _ in range(min(4, bench_args.output_len[0])):
        next_ids = _decode_once(next_ids, batch, model_runner, sg[1])
    torch.cuda.synchronize()
    rank_print("Warmup done.")

    # 5) Sweep
    rank_print("Benchmark ...")
    all_results = []
    stage = bench_args.stage
    if stage == "p":
        stage = "prefill"
    elif stage == "d":
        stage = "decode"
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        result = bench_one_config(
            model_runner=model_runner,
            rank_print=rank_print,
            batch_size=bs,
            input_len=il,
            output_len=ol,
            stream_group_idx=bench_args.stream_group_idx,
            warmup_steps=bench_args.warmup_steps,
            bench_steps=bench_args.bench_steps,
            bench_args=bench_args,
            rng=rng,
            tp_rank=tp_rank,
            stage=stage,  # type: ignore[arg-type]
        )
        if result is not None:
            result["summary"]["run_name"] = bench_args.run_name
            all_results.append(result)

    # 6) Write results (rank 0 only, but you can modify this to write results to a file)
    if tp_rank == 0 and bench_args.result_filename:
        output_payload = {
            "run_name": bench_args.run_name,
            "stream_group_idx": bench_args.stream_group_idx,
            "sm_counts": get_sm_counts(),
            "model_path": server_args.model_path,
            "tp_size": server_args.tp_size,
            "dp_size": server_args.dp_size,
            "configs": all_results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(bench_args.result_filename)), exist_ok=True)
        with open(bench_args.result_filename, "w") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        rank_print(f"Results written to {bench_args.result_filename}")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(server_args: ServerArgs, bench_args: PdmuxBenchArgs):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        bench_pdmux(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=bench_pdmux,
                    args=(server_args, port_args, bench_args, gpu_id, tp_rank),
                )
                proc.start()
                workers.append(proc)

        for proc in workers:
            proc.join()
        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark PDMUX + special-DP-attention latency"
    )
    ServerArgs.add_cli_args(parser)
    PdmuxBenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = PdmuxBenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
