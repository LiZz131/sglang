"""
Offline replay + GPU-time measurement for PDMUX (+ optional special-dp-attention).

This script consumes a request dump produced by:
  python -m sglang.bench_serving ... --dump-requests requests.jsonl

For each fixed stream_group_idx (no auto switching), records **three** metric families:

  - **GPU run (CUDA events)**: ``prefill_gpu_run_ms`` / ``decode_gpu_run_ms`` — elapsed GPU time of
    ``model_runner.forward`` on the prefill/decode stream (sync before reading events).
  - **CPU prepare+launch**: ``prefill_cpu_prepare_and_launch_ms`` / ``decode_cpu_prepare_and_launch_ms`` —
    prefill from ``build_schedule_batch`` start through ``forward`` return; decode from ``prepare_for_decode``
    through ``forward`` return (no sync in the interval).
  - **CPU launch-only**: ``prefill_cpu_launch_only_ms`` / ``decode_cpu_launch_only_ms`` — forward call only;
    decode additionally uses monkeypatched graph ``replay()`` timing when CUDA graph is used.

Legacy keys ``prefill_table`` / ``decode_table`` duplicate **CPU prepare+launch** tables for
``PDMuxOfflineTables`` / existing tooling.

Notes / scope:
  - arrival time is ignored by design
  - decode currently keys by bs only (as requested)
  - prefill key uses max(seq_lens) within the batch (as requested)
  - ``--prefill-batch-sizes`` and ``--decode-batch-sizes`` are independent; the script sweeps
    ``sorted(set(prefill) | set(decode))`` and records each table only for sizes listed in the
    corresponding set (so decode can use larger chunk sizes than prefill).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import multiprocessing
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
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

# Set by monkeypatched CudaGraphRunner.replay: last ``graph.replay()`` CPU duration (seconds).
_BENCH_LAST_CUDA_GRAPH_LAUNCH_S: Optional[float] = None
_BENCH_CUDA_GRAPH_REPLAY_PATCHED = False


def install_bench_cuda_graph_launch_timing_patch() -> None:
    """Patch ``CudaGraphRunner.replay`` to record CPU time of ``graphs[key].replay()`` only (bench-local)."""
    global _BENCH_CUDA_GRAPH_REPLAY_PATCHED
    if _BENCH_CUDA_GRAPH_REPLAY_PATCHED:
        return
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
    from sglang.srt.multiplex.pdmux_context import get_current_stream_idx

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        global _BENCH_LAST_CUDA_GRAPH_LAUNCH_S
        self.deepep_adapter.replay()

        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)

        if self.enable_pdmux:
            graph_key = f"{get_current_stream_idx()}_{self.bs}"
        else:
            graph_key = self.bs
        t0 = time.perf_counter()
        self.graphs[graph_key].replay()
        _BENCH_LAST_CUDA_GRAPH_LAUNCH_S = time.perf_counter() - t0
        output = self.output_buffers[graph_key]

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = output.full_logits[: self.raw_num_token]
            else:
                full_logits = None
                next_token_logits = output.next_token_logits[: self.raw_num_token]

            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        assert isinstance(output, PPProxyTensors)
        return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})

    CudaGraphRunner.replay = replay  # type: ignore[assignment]
    _BENCH_CUDA_GRAPH_REPLAY_PATCHED = True


@dataclasses.dataclass
class ReplayArgs:
    requests: str = ""
    stream_groups: Tuple[int, ...] = (1,)
    prefill_batch_sizes: Tuple[int, ...] = (1, 2, 4, 8)
    decode_batch_sizes: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    max_requests: int = 0
    warmup_batches: int = 1
    decode_steps: int = 1
    output: str = "pdmux_offline_tables.json"
    disable_tqdm: bool = False

    # If pdmux-config-path is NOT provided, allow a manual split for a simple 3-group config.
    prefill_sm: int = 0
    decode_sm: int = 0

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--requests", type=str, required=True, help="requests.jsonl")
        parser.add_argument(
            "--stream-groups",
            type=int,
            nargs="+",
            default=list(ReplayArgs.stream_groups),
            help="Stream group indices to measure (fixed, no auto switching).",
        )
        parser.add_argument(
            "--prefill-batch-sizes",
            type=int,
            nargs="+",
            default=list(ReplayArgs.prefill_batch_sizes),
            help="Chunk sizes for prefill (and request pool) when recording prefill_* tables.",
        )
        parser.add_argument(
            "--decode-batch-sizes",
            type=int,
            nargs="+",
            default=list(ReplayArgs.decode_batch_sizes),
            help="Chunk sizes for decode_* tables; can be larger than prefill (e.g. 256). "
            "Sweep uses the union of prefill and decode sizes; each size records only the tables it applies to.",
        )
        parser.add_argument(
            "--batch-sizes",
            type=int,
            nargs="+",
            default=None,
            help="Deprecated: equivalent to setting both --prefill-batch-sizes and --decode-batch-sizes to this list.",
        )
        parser.add_argument(
            "--max-requests",
            type=int,
            default=ReplayArgs.max_requests,
            help="Optional cap on how many requests to load (0 = all).",
        )
        parser.add_argument(
            "--warmup-batches",
            type=int,
            default=ReplayArgs.warmup_batches,
            help="Warm up N batches per (stream_group, bs) before recording.",
        )
        parser.add_argument(
            "--decode-steps",
            type=int,
            default=ReplayArgs.decode_steps,
            help="How many decode steps to run after prefill for timing (default 1).",
        )
        parser.add_argument("--output", type=str, default=ReplayArgs.output)
        parser.add_argument("--prefill-sm", type=int, default=ReplayArgs.prefill_sm)
        parser.add_argument("--decode-sm", type=int, default=ReplayArgs.decode_sm)
        parser.add_argument(
            "--disable-tqdm",
            action="store_true",
            help="Disable tqdm progress bars (default: show bars on rank 0).",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        kwargs: Dict[str, Any] = {}
        if getattr(args, "batch_sizes", None) is not None:
            t = tuple(args.batch_sizes)
            kwargs["prefill_batch_sizes"] = t
            kwargs["decode_batch_sizes"] = t
        attrs = [(a.name, type(a.default)) for a in dataclasses.fields(cls)]
        for k, t in attrs:
            if k in kwargs:
                continue
            if k in ("prefill_batch_sizes", "decode_batch_sizes"):
                kwargs[k] = tuple(getattr(args, k))
            else:
                kwargs[k] = t(getattr(args, k))
        return cls(**kwargs)


def setup_pdmux_streams(gpu_id: int, server_args: ServerArgs, replay_args: ReplayArgs):
    if server_args.pdmux_config_path:
        config = load_pdmux_config(server_args.pdmux_config_path)
    else:
        if replay_args.prefill_sm <= 0 or replay_args.decode_sm <= 0:
            raise ValueError(
                "Either --pdmux-config-path or both --prefill-sm/--decode-sm must be set."
            )
        config = PDMuxConfig(
            sm_group_num=3,
            manual_divisions=[[replay_args.prefill_sm, replay_args.decode_sm, 0]],
        )
    server_args.sm_group_num = config.sm_group_num
    initialize_stream_groups(gpu_id, config)
    return config


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


def _make_dummy_tree_cache(model_runner):
    return SimpleNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner: ModelRunner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
            enable_special_dp_attention=model_runner.server_args.enable_special_dp_attention,
        )


def build_schedule_batch(
    reqs: List[Req],
    model_runner: ModelRunner,
) -> Tuple[ScheduleBatch, ForwardBatch, float]:
    """Returns ``(batch, forward_batch, t_build_start)`` where ``t_build_start`` is perf_counter at entry."""
    t_build_start = time.perf_counter()
    tree_cache = _make_dummy_tree_cache(model_runner)
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    return batch, forward_batch, t_build_start


@torch.no_grad()
def prefill_forward_and_sample(
    forward_batch: ForwardBatch,
    model_runner: ModelRunner,
    stream: torch.cuda.Stream,
    t_build_start: float,
) -> Tuple[float, float, float, torch.Tensor]:
    """一次 extend forward + sample。

    返回 ``(gpu_run_ms, cpu_prepare_and_launch_ms, cpu_launch_only_ms, next_ids)``。
    ``gpu_run_ms`` 为 CUDA event 对 ``forward`` 的计时；CPU 两值为 ``perf_counter``（毫秒）。
    """
    t_launch_start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start_event.record(stream)
        logits_output = model_runner.forward(forward_batch).logits_output
        end_event.record(stream)
    t_launch_end = time.perf_counter()
    stream.synchronize()
    gpu_run_ms = float(start_event.elapsed_time(end_event))
    prepare_and_launch_ms = (t_launch_end - t_build_start) * 1000.0
    launch_only_ms = (t_launch_end - t_launch_start) * 1000.0
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return gpu_run_ms, prepare_and_launch_ms, launch_only_ms, next_token_ids


@torch.no_grad()
def timed_decode_run(
    batch: ScheduleBatch,
    input_token_ids: torch.Tensor,
    model_runner: ModelRunner,
    stream: torch.cuda.Stream,
) -> Tuple[float, float, float, torch.Tensor]:
    """Decode: GPU（CUDA event）+ 两套 CPU 计时。语义同 ``prefill_forward_and_sample``。"""
    global _BENCH_LAST_CUDA_GRAPH_LAUNCH_S
    batch.output_ids = input_token_ids
    t_prepare_start = time.perf_counter()
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)

    _BENCH_LAST_CUDA_GRAPH_LAUNCH_S = None
    t_forward_start = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start_event.record(stream)
        logits_output = model_runner.forward(forward_batch).logits_output
        end_event.record(stream)
    t_forward_end = time.perf_counter()
    stream.synchronize()
    gpu_run_ms = float(start_event.elapsed_time(end_event))
    prepare_and_launch_ms = (t_forward_end - t_prepare_start) * 1000.0
    if _BENCH_LAST_CUDA_GRAPH_LAUNCH_S is not None:
        launch_only_ms = _BENCH_LAST_CUDA_GRAPH_LAUNCH_S * 1000.0
    else:
        launch_only_ms = (t_forward_end - t_forward_start) * 1000.0
    _BENCH_LAST_CUDA_GRAPH_LAUNCH_S = None
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return gpu_run_ms, prepare_and_launch_ms, launch_only_ms, next_token_ids


def load_requests_jsonl(path: Path, max_requests: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_requests and len(items) >= max_requests:
                break
    return items


def build_reqs_from_prompts(
    tokenizer,
    prompts: List[Any],
    *,
    max_new_tokens: int,
) -> List[Req]:
    # The offline replay focuses on GPU time; we keep sampling deterministic.
    sampling_params = SamplingParams(temperature=0, max_new_tokens=max_new_tokens)
    reqs: List[Req] = []
    for i, p in enumerate(prompts):
        if isinstance(p, str):
            input_ids = tokenizer.encode(p)
        elif isinstance(p, list):
            # chat messages or multi-turn prompts: serialize via tokenizer if possible, else fallback to str()
            try:
                input_ids = tokenizer.encode(p)
            except Exception:
                input_ids = tokenizer.encode(str(p))
        else:
            input_ids = tokenizer.encode(str(p))
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)
    return reqs


def chunked(iterable: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable) - (len(iterable) % n), n):
        yield iterable[i : i + n]


def replay_worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    replay_args: ReplayArgs,
    gpu_id: int,
    tp_rank: int,
):
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, tp_rank
        )

    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    show_pbar = tp_rank == 0 and not replay_args.disable_tqdm

    # Initialize PDMUX streams prior to ModelRunner (for cudagraph capture per stream group)
    setup_pdmux_streams(gpu_id, server_args, replay_args)
    server_args.enable_pdmux = True

    model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
    install_bench_cuda_graph_launch_timing_patch()
    stream_groups = get_stream_groups()
    sm_counts = get_sm_counts()

    req_items = load_requests_jsonl(Path(replay_args.requests), replay_args.max_requests)
    prompts_all = [x.get("prompt") for x in req_items]
    out_lens_all = [int(x.get("output_len", 1)) for x in req_items]
    default_out_len = int(np.median(out_lens_all)) if out_lens_all else 1

    prefill_table: Dict[int, Dict[str, float]] = defaultdict(dict)
    decode_table: Dict[int, Dict[str, float]] = defaultdict(dict)
    prefill_launch_only_table: Dict[int, Dict[str, float]] = defaultdict(dict)
    decode_launch_only_table: Dict[int, Dict[str, float]] = defaultdict(dict)
    prefill_gpu_table: Dict[int, Dict[str, float]] = defaultdict(dict)
    decode_gpu_table: Dict[int, Dict[str, float]] = defaultdict(dict)

    # aggregate accumulators: GPU ms (CUDA events), CPU ms (perf_counter)
    prefill_gpu_acc: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    prefill_pl_acc: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    prefill_lo_acc: Dict[Tuple[int, int, int], List[float]] = defaultdict(list)
    decode_gpu_acc: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    decode_pl_acc: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    decode_lo_acc: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    for sg in replay_args.stream_groups:
        if sg < 0 or sg >= len(stream_groups):
            rank_print(f"Skip invalid stream_group_idx={sg}, len={len(stream_groups)}")
            continue

        set_current_stream_idx(sg)
        model_runner.update_decode_attn_backend(sg)
        prefill_stream, decode_stream = stream_groups[sg]
        prefill_sm, decode_sm = sm_counts[sg]

        rank_print(
            f"Measuring stream_group={sg} prefill_sm={prefill_sm} decode_sm={decode_sm}"
        )

        prefill_bs_set = set(replay_args.prefill_batch_sizes)
        decode_bs_set = set(replay_args.decode_batch_sizes)
        chunk_bs_list = sorted(prefill_bs_set | decode_bs_set)
        if not chunk_bs_list:
            rank_print("Empty prefill/decode batch size sets; nothing to measure.")
            continue
        if max(chunk_bs_list) > len(prompts_all):
            rank_print(
                f"Warning: max chunk_bs={max(chunk_bs_list)} > loaded requests={len(prompts_all)}; "
                "larger sizes may produce no batches."
            )

        bs_iter = chunk_bs_list
        if show_pbar:
            bs_iter = tqdm(
                chunk_bs_list,
                desc=f"stream_group={sg} · chunk_bs (union)",
                unit="cfg",
                leave=False,
            )
        for bs in bs_iter:
            # warmup and measure using consecutive chunks in request order
            batches = list(chunked(prompts_all, bs))
            if not batches:
                continue

            warmup_n = min(replay_args.warmup_batches, len(batches))
            measure_batches = batches[warmup_n:]
            if not measure_batches:
                measure_batches = batches[-1:]

            total_batches = warmup_n + len(measure_batches)
            inner_pbar = (
                tqdm(
                    total=total_batches,
                    desc=f"  sg={sg} bs={bs} warmup+measure",
                    unit="batch",
                    leave=False,
                )
                if show_pbar
                else None
            )

            # Warmup
            for wb in batches[:warmup_n]:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                reqs = build_reqs_from_prompts(
                    tokenizer, wb, max_new_tokens=default_out_len
                )
                batch, fwd, t_build = build_schedule_batch(reqs, model_runner)
                _, _, _, next_ids = prefill_forward_and_sample(
                    fwd, model_runner, prefill_stream, t_build
                )
                _, _, _, next_ids = timed_decode_run(
                    batch, next_ids, model_runner, decode_stream
                )
                torch.cuda.synchronize()
                if inner_pbar is not None:
                    inner_pbar.update(1)

            # Measure
            for mb in measure_batches:
                model_runner.req_to_token_pool.clear()
                model_runner.token_to_kv_pool_allocator.clear()
                reqs = build_reqs_from_prompts(
                    tokenizer, mb, max_new_tokens=default_out_len
                )
                max_len = max(len(r.fill_ids) for r in reqs) if reqs else 0
                batch, fwd, t_build = build_schedule_batch(reqs, model_runner)

                pre_gpu, pre_pl, pre_lo, next_ids = prefill_forward_and_sample(
                    fwd, model_runner, prefill_stream, t_build
                )
                if bs in prefill_bs_set:
                    prefill_gpu_acc[(sg, bs, max_len)].append(pre_gpu)
                    prefill_pl_acc[(sg, bs, max_len)].append(pre_pl)
                    prefill_lo_acc[(sg, bs, max_len)].append(pre_lo)

                for _ in range(max(1, replay_args.decode_steps)):
                    dec_gpu, dec_pl, dec_lo, next_ids = timed_decode_run(
                        batch, next_ids, model_runner, decode_stream
                    )
                    if bs in decode_bs_set:
                        decode_gpu_acc[(sg, bs)].append(dec_gpu)
                        decode_pl_acc[(sg, bs)].append(dec_pl)
                        decode_lo_acc[(sg, bs)].append(dec_lo)
                if inner_pbar is not None:
                    inner_pbar.update(1)

            if inner_pbar is not None:
                inner_pbar.close()

    # reduce / write only on tp_rank==0
    if tp_rank == 0:
        # build tables as JSON-serializable dicts
        for (sg, bs, maxlen), vals in prefill_gpu_acc.items():
            key = f"bs={bs},max_seq_len={maxlen}"
            prefill_gpu_table[sg][key] = float(np.mean(vals)) if vals else 0.0
        for (sg, bs, maxlen), vals in prefill_pl_acc.items():
            key = f"bs={bs},max_seq_len={maxlen}"
            v = float(np.mean(vals)) if vals else 0.0
            prefill_table[sg][key] = v
        for (sg, bs, maxlen), vals in prefill_lo_acc.items():
            key = f"bs={bs},max_seq_len={maxlen}"
            prefill_launch_only_table[sg][key] = float(np.mean(vals)) if vals else 0.0
        for (sg, bs), vals in decode_gpu_acc.items():
            key = f"bs={bs}"
            decode_gpu_table[sg][key] = float(np.mean(vals)) if vals else 0.0
        for (sg, bs), vals in decode_pl_acc.items():
            key = f"bs={bs}"
            v = float(np.mean(vals)) if vals else 0.0
            decode_table[sg][key] = v
        for (sg, bs), vals in decode_lo_acc.items():
            key = f"bs={bs}"
            decode_launch_only_table[sg][key] = float(np.mean(vals)) if vals else 0.0

        payload = {
            "source_requests": replay_args.requests,
            "stream_groups": list(replay_args.stream_groups),
            "prefill_batch_sizes": list(replay_args.prefill_batch_sizes),
            "decode_batch_sizes": list(replay_args.decode_batch_sizes),
            "chunk_batch_sizes_sweep": sorted(
                set(replay_args.prefill_batch_sizes) | set(replay_args.decode_batch_sizes)
            ),
            "prefill_gpu_run_ms": prefill_gpu_table,
            "decode_gpu_run_ms": decode_gpu_table,
            "prefill_cpu_prepare_and_launch_ms": prefill_table,
            "prefill_cpu_launch_only_ms": prefill_launch_only_table,
            "decode_cpu_prepare_and_launch_ms": decode_table,
            "decode_cpu_launch_only_ms": decode_launch_only_table,
            # Backward-compatible aliases (CPU prepare+launch, for PDMuxOfflineTables).
            "prefill_table": prefill_table,
            "decode_table": decode_table,
            "sm_counts": get_sm_counts(),
        }
        out_path = Path(replay_args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        rank_print(f"Wrote tables to {out_path}")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args: ServerArgs, replay_args: ReplayArgs):
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        replay_worker(server_args, port_args, replay_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=replay_worker,
                    args=(server_args, port_args, replay_args, gpu_id, tp_rank),
                )
                proc.start()
                workers.append(proc)
        for p in workers:
            p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline replay for PDMUX GPU timing tables")
    ServerArgs.add_cli_args(parser)
    ReplayArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    replay_args = ReplayArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )
    try:
        main(server_args, replay_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)

