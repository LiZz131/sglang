"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

import torch.cuda.nvtx as nvtx
from sglang.srt.distributed.parallel_state import set_pdmux_status
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
    set_current_stream_idx,
)
from sglang.srt.multiplex.pdmux_offline_tables import (
    PDMuxOfflineTables,
    load_pdmux_offline_tables,
    normalize_tie_break,
)
from sglang.srt.multiplex.pdmux_time_model import (
    PDMuxTimePredictor,
    decode_lb_from_schedule_batch,
    default_coefficients_for_num_groups,
    load_pdmux_fitted_coefficients_yaml,
    match_coefficients_for_sm_counts,
    prefill_fg_from_schedule_batch,
)
from sglang.srt.mem_cache.common import release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _pdmux_nvtx_len_list_str(
    batch: "ScheduleBatch", *, add_decode_global_num_tokens: bool = False
) -> str:
    """Build NVTX list suffix for PD multiplex profiling.

    - **Local**: per-req sequence lengths on this worker (``seq_lens_cpu`` or
      ``extend_input_len``), e.g. ``[1024,512]`` — bs = number of entries.
    - **Decode +** ``add_decode_global_num_tokens``: append ``|`` and
      ``batch.global_num_tokens`` (per-DP-rank token counts after MLP sync),
      e.g. ``[128,256]|[3,5]``. If global is missing, ``|[]`` for stable shape.
    Prefill keeps only the local segment (no ``|``).
    """
    n = len(batch.reqs)
    slc = batch.seq_lens_cpu
    if slc is not None and slc.numel() == n:
        vals = [int(x) for x in slc.tolist()]
    else:
        if n > 0:
            logger.debug(
                "NVTX: seq_lens_cpu missing or len mismatch; using extend_input_len"
            )
        vals = [int(req.extend_input_len) for req in batch.reqs]
    local_str = "[" + ",".join(str(v) for v in vals) + "]"

    if not add_decode_global_num_tokens:
        return local_str

    gnt = batch.global_num_tokens
    if gnt is not None and len(gnt) > 0:
        global_str = "[" + ",".join(str(int(x)) for x in gnt) + "]"
    else:
        global_str = "[]"
    return f"{local_str}|{global_str}"


def _pdmux_special_dp_nvtx_prefill_pred_chunks(
    pred: Optional[PDMuxTimePredictor],
    stream_idx: int,
    prefill_batch: "ScheduleBatch",
    forward_count: int,
    num_hidden_layers: int,
) -> Optional[dict]:
    """Scaled prefill launch/run predictions for one split-prefill chunk (special DP loop).

    Fitted coefficients are treated as **full-model** (all layers) times; this chunk scales by
    ``forward_count / num_hidden_layers`` (e.g. 10/15 of full).
    """
    if pred is None:
        return None
    nh = max(1, int(num_hidden_layers))
    frac = float(forward_count) / float(nh)
    F, G = prefill_fg_from_schedule_batch(prefill_batch)
    g = stream_idx
    pl_full = pred.predict_prefill_launch_ms(g, F, G)
    pr_full = pred.predict_prefill_run_ms(g, F, G)
    return {
        "pl_chunk_ms": pl_full * frac,
        "pr_chunk_ms": pr_full * frac,
        "pl_full_ms": pl_full,
        "pr_full_ms": pr_full,
        "layer_frac": frac,
    }


class SchedulerMultiplexMixin:

    def init_pdmux(self: Scheduler):
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None
        self.decode_or_idle_batch: Optional[ScheduleBatch] = None
        self.decode_forward_count = 0

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(self.server_args.pdmux_config_path)
        initialize_stream_groups(self.gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )
        self.pdmux_no_all_decode_SMs = getattr(self.server_args, "pdmux_no_all_decode_SMs", False)
        if self.pdmux_no_all_decode_SMs:
            logger.info("PD-Multiplexing: disable all decode SMs stream group")
        self.log_stream_groups()

        self._pdmux_time_predictor: Optional[PDMuxTimePredictor] = None
        if getattr(self.server_args, "enable_pdmux_time_predictor", False):
            budget_ms = float(self.server_args.pdmux_decode_budget_ms)
            fitted_path = getattr(
                self.server_args, "pdmux_fitted_coefficients_path", None
            )
            fallback_one = default_coefficients_for_num_groups(1)[0]
            per_group = default_coefficients_for_num_groups(self.real_sm_group_num)
            if fitted_path:
                try:
                    coeff_map = load_pdmux_fitted_coefficients_yaml(fitted_path)
                    per_group = match_coefficients_for_sm_counts(
                        self.sm_counts, coeff_map, fallback=fallback_one
                    ) # for 0 sg and last sg, we not use predict
                    logger.info(
                        "PDMux time predictor: loaded fitted coefficients from %s "
                        "(%s stream groups matched to sm_counts)",
                        fitted_path,
                        len(per_group),
                    )
                except Exception as e:
                    logger.warning(
                        "PDMux time predictor: failed to load %s (%s); using built-in defaults",
                        fitted_path,
                        e,
                    )
            else:
                logger.info(
                    "PDMux time predictor: no --pdmux-fitted-coefficients-path; using built-in defaults"
                )
            self._pdmux_time_predictor = PDMuxTimePredictor(
                per_group,
                budget_ms=budget_ms,
            )
            logger.info(
                "PDMux time predictor enabled: decode_budget_ms=%s",
                budget_ms,
            )

        self._pdmux_offline_tables: Optional[PDMuxOfflineTables] = None
        offline_path = getattr(self.server_args, "pdmux_offline_tables_path", None)
        if offline_path:
            try:
                self._pdmux_offline_tables = load_pdmux_offline_tables(
                    offline_path,
                    decode_bs_tie_break=normalize_tie_break(
                        getattr(
                            self.server_args,
                            "pdmux_offline_decode_bs_tie_break",
                            "up",
                        ),
                        "up",
                    ),
                    prefill_bs_tie_break=normalize_tie_break(
                        getattr(
                            self.server_args,
                            "pdmux_offline_prefill_bs_tie_break",
                            "down",
                        ),
                        "down",
                    ),
                    prefill_max_seq_len_tie_break=normalize_tie_break(
                        getattr(
                            self.server_args,
                            "pdmux_offline_prefill_max_seq_len_tie_break",
                            "down",
                        ),
                        "down",
                    ),
                )
                logger.info(
                    "PDMux offline timing tables loaded from %s", offline_path
                )
            except Exception as e:
                logger.warning(
                    "PDMux offline tables: failed to load %s (%s); overlap budget uses defaults",
                    offline_path,
                    e,
                )

    def log_stream_groups(self: Scheduler):
        for i, stream_group in enumerate(self.stream_groups):
            # group, prefill stream id/sms, decode stream id/sms
            logger.info(f"stream_group {i}: ")
            logger.info(f"  prefill stream id: {stream_group[0].stream_id}, sms: {self.sm_counts[i][0]}")
            logger.info(f"  decode stream id: {stream_group[1].stream_id}, sms: {self.sm_counts[i][1]}")

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self: Scheduler,
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if not self.running_batch.is_empty() and self.split_prefill_batch:
            decode_bs = self.running_batch.batch_size()
            manual_divisions = self.pdmux_config.manual_divisions
            if manual_divisions:
                for i in range(len(manual_divisions)):
                    _, _, threshold = manual_divisions[i]
                    if decode_bs >= threshold:
                        stream_idx = i + 1
            else:
                stream_idx = max(
                    1,
                    min(
                        self.real_sm_group_num - 2,
                        decode_bs
                        * (self.real_sm_group_num - 2)
                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)
        elif not self.running_batch.is_empty():
            # set_current_stream_idx(self.real_sm_group_num - 1)
            # DEBUG(lbz): use the second last, for prefill need SMs
            if self.pdmux_no_all_decode_SMs:
                set_current_stream_idx(self.real_sm_group_num - 2)
            else:
                set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def adjust_stream_groups_for_special_dp_attention(
        self: Scheduler,
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if not getattr(self.server_args, "auto_adjust_stream_group", True):
            stream_idx = int(getattr(self.server_args, "manual_stream_group_idx", 0))
            stream_idx = max(0, min(stream_idx, self.real_sm_group_num - 1))
            set_current_stream_idx(stream_idx)
            self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
            return stream_idx, self.stream_groups[stream_idx]

        max_decode_bs = 0
        if self.decode_or_idle_batch is not None:
            gnt = getattr(self.decode_or_idle_batch, "global_num_tokens", None)
            if gnt:
                max_decode_bs = sum(int(x) for x in gnt)

        # Decode-only (no active split prefill): use full-GPU decode partition. Skip predictor —
        # PD multiplex tradeoff only matters when prefill can run alongside decode.
        if not self.split_prefill_batch and max_decode_bs != 0:
            stream_idx = self.real_sm_group_num - 1
            set_current_stream_idx(stream_idx)
            self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
            return stream_idx, self.stream_groups[stream_idx]

        # Time predictor: min decode_sm among groups with predicted decode_run <= budget.
        # Do NOT require decode_or_idle_batch.is_empty()==False: under special DP attention,
        # only the rank whose decode_dp_rank matches has local decode reqs; other ranks get an
        # IDLE batch with 0 reqs but identical global_num_tokens / global_seq_lens_sum_per_dp
        # after MLP sync — we must use those globals so all DP ranks pick the same stream_idx.
        pred = getattr(self, "_pdmux_time_predictor", None)
        diag = getattr(self.server_args, "enable_pdmux_diag_log", False)
        if pred is not None and self.decode_or_idle_batch is not None:
            lb = decode_lb_from_schedule_batch(self.decode_or_idle_batch)
            if lb is not None:
                L, B = lb
                b = self.decode_or_idle_batch
                gnt = getattr(b, "global_num_tokens", None)
                gss = getattr(b, "global_seq_lens_sum_per_dp", None)
                logger.debug(
                    "pdmux_time_predictor inputs: attn_dp_rank=%s tp_rank=%s batch_size=%s "
                    "L=%s B=%s global_num_tokens=%s global_seq_lens_sum_per_dp=%s",
                    getattr(self, "attn_dp_rank", None),
                    getattr(self, "tp_rank", None),
                    b.batch_size(),
                    L,
                    B,
                    gnt,
                    gss,
                )
                # choose_stream_group_decode_slo skips g=0 (full-prefill / decode_sm=0 endpoint).
                chosen = pred.choose_stream_group_decode_slo(
                    L, B, self.sm_counts, diag_log=diag
                )
                logger.info(
                    "pdmux_time_predictor result: attn_dp_rank=%s tp_rank=%s chosen_stream_group=%s",
                    getattr(self, "attn_dp_rank", None),
                    getattr(self, "tp_rank", None),
                    chosen,
                )
                if chosen is not None:
                    stream_idx = max(0, min(chosen, self.real_sm_group_num - 1))
                    set_current_stream_idx(stream_idx)
                    self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
                    return stream_idx, self.stream_groups[stream_idx]
                logger.debug(
                    "pdmux_time_predictor: no feasible stream group for L=%s B=%s; fallback to heuristic",
                    L,
                    B,
                )

        # use the max decode bs to adjust the stream group (max_decode_bs computed above)
        if not max_decode_bs == 0 and self.split_prefill_batch:
            manual_divisions = self.pdmux_config.manual_divisions
            if manual_divisions:
                for i in range(len(manual_divisions)):
                    _, _, threshold = manual_divisions[i]
                    if max_decode_bs >= threshold:
                        stream_idx = i + 1
            else:
                stream_idx = max(
                    1,
                    min(
                        self.real_sm_group_num - 2,
                        max_decode_bs
                        * (self.real_sm_group_num - 2)
                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)
        elif not max_decode_bs == 0:
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(self: Scheduler, sm_count: int) -> bool:
        if self.split_prefill_batch:
            return False

        # add new request
        batch = self.get_new_batch_prefill()
        if batch and not batch.is_empty():
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            self.split_prefill_batch = batch
            return True
        return False

    @torch.inference_mode()
    def event_loop_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # for nvtx profiling, we need to use the range_start and range_end
        loop_range_handle = None
        decode_forward_count = 0
        decode_gpu_handle = None
        prefill_whole_batch_count = 0
        prefill_forward_count = 0
        prefill_gpu_handle = None
        prefill_launch_handle = None

        def get_sg_msg():
            return f"stream_group: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"

        logger.debug("Starting event loop for pd multiplexing...")

        while True:
            if loop_range_handle is None:
                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()

                nvtx.range_end(loop_range_handle)
                loop_range_handle = None

                stream_idx, stream_group = self.adjust_stream_groups()

                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if self.running_batch and not self.running_batch.is_empty():
                    decode_gpu_handle = nvtx.range_start(
                        "run decode batch"
                        + f": {decode_forward_count} "
                        + _pdmux_nvtx_len_list_str(
                            self.running_batch, add_decode_global_num_tokens=True
                        )
                    )
                    decode_forward_count += 1
                    decode_result = self.run_batch(self.running_batch)
                    decode_done = True
                else:
                    decode_done = False

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    if prefill_gpu_handle is None:
                        prefill_gpu_handle = nvtx.range_start(
                        "run prefill batch"
                        + f": {prefill_whole_batch_count} "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch))
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start(
                        "launch prefill batch"
                        + f": {prefill_forward_count} "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                    )
                    prefill_forward_count += 1
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    nvtx.range_end(prefill_launch_handle)
                    prefill_launch_handle = None
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    nvtx.range_end(decode_gpu_handle)
                    decode_gpu_handle = None
                    self.process_batch_result(self.running_batch, decode_result)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        nvtx.range_end(prefill_gpu_handle)
                        prefill_gpu_handle = None
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        # log prefill stats late
                        self.log_prefill_stats_late(self.split_prefill_batch)
                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True

    @torch.inference_mode()
    def event_loop_overlap_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing with overlap."""
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # for nvtx profiling, we need to use the range_start and range_end
        loop_range_handle = None
        decode_forward_count = 0
        prefill_whole_batch_count = 0
        prefill_forward_count = 0
        prefill_gpu_handle = None
        prefill_launch_handle = None

        decode_result_queue = deque()
        # Keep compatibility with shared overlap checks (e.g. flush_cache/_is_no_request).
        self.result_queue = decode_result_queue
        decode_last_batch = None
        decode_need_wait = False
        decode_run_done = None
        decode_run_done_step = None
        arm_decode_need_wait_after_prefill = False

        def get_sg_msg():
            return f"stream_group: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"

        OVERLAP_LOG_LEVEL = False
        def overlap_log(message: str):
            if OVERLAP_LOG_LEVEL:
                logger.info(f"[pdmux-overlap] {message}")

        overlap_log(
            f"start loop stream_group={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}"
        )

        while True:
            if loop_range_handle is None:
                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")
            
            # decode recv requests
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            # prefill update batch
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            # decode update batch
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            # adjust stream group
            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()

                while decode_result_queue:
                    (
                        decode_batch_to_process,
                        decode_result_to_process,
                        decode_gpu_handle_to_process,
                    ) = decode_result_queue.popleft()
                    nvtx.range_end(decode_gpu_handle_to_process)
                    self.process_batch_result(
                        decode_batch_to_process, decode_result_to_process
                    )
                    all_req_finished = (
                        len(decode_batch_to_process.reqs) > 0
                        and all(req.finished() for req in decode_batch_to_process.reqs)
                    )
                    if all_req_finished:
                        decode_need_wait = False
                        decode_run_done = None
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "drain decode queue: all req finished, set decode_need_wait=False"
                        )
                # Drain already processed all pending results from previous loops.
                # Reset decode_last_batch so the pop-and-process guard below does not
                # fire on the batch we are about to launch in this same loop.
                decode_last_batch = None
                overlap_log("drain done: reset decode_last_batch=None")

                nvtx.range_end(loop_range_handle)
                loop_range_handle = None

                stream_idx, stream_group = self.adjust_stream_groups()

                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )
                overlap_log(
                    f"adjust stream group -> idx={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}"
                )

            decode_batch = None
            # run decode batch
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if self.running_batch and not self.running_batch.is_empty():
                    decode_batch = self.running_batch
                    decode_gpu_handle = nvtx.range_start(
                        "run decode batch"
                        + f": {decode_forward_count} "
                        + _pdmux_nvtx_len_list_str(
                            decode_batch, add_decode_global_num_tokens=True
                        )
                    )
                    decode_forward_count += 1
                    if decode_need_wait and decode_run_done is not None:
                        overlap_log(
                            f"decode wait event before launch, prev_step={decode_run_done_step}, queue_len={len(decode_result_queue)}"
                        )
                        decode_stream.wait_event(decode_run_done)
                    # Attach a stable step id for cross-module logging (run_batch/copy_done).
                    decode_step = decode_forward_count - 1
                    setattr(decode_batch, "_pdmux_decode_step", decode_step)
                    decode_result = self.run_batch(decode_batch)
                    decode_run_done = decode_stream.record_event()
                    decode_run_done_step = decode_step
                    overlap_log(
                        f"decode_run_done <- decode_stream.record_event (fallback), step={decode_step}"
                    )
                    decode_result_queue.append(
                        (decode_batch.copy(), decode_result, decode_gpu_handle)
                    )
                    overlap_log(
                        f"launch decode batch#{decode_forward_count-1}, bs={decode_batch.batch_size()}, need_wait={decode_need_wait}, arm_after_prefill={arm_decode_need_wait_after_prefill}, queue_len={len(decode_result_queue)}"
                    )
                    if arm_decode_need_wait_after_prefill:
                        # Strict sync semantics: the 2nd decode after prefill must wait.
                        decode_need_wait = True
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "launch decode batch: consume arm_after_prefill, set decode_need_wait=True"
                        )

            # run prefill batch
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    if prefill_gpu_handle is None:
                        prefill_gpu_handle = nvtx.range_start(
                        "run prefill batch"
                        + f": {prefill_whole_batch_count} "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch))
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start(
                        "launch prefill batch"
                        + f": {prefill_forward_count} "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                    )
                    prefill_forward_count += 1
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    nvtx.range_end(prefill_launch_handle)
                    prefill_launch_handle = None
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            # process decode result
            if decode_last_batch and len(decode_result_queue) > 0:
                (
                    decode_batch_to_process,
                    decode_result_to_process,
                    decode_gpu_handle_to_process,
                ) = decode_result_queue.popleft()
                nvtx.range_end(decode_gpu_handle_to_process)
                overlap_log(
                    f"process decode result: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                )
                self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                all_req_finished = (
                    len(decode_batch_to_process.reqs) > 0
                    and all(req.finished() for req in decode_batch_to_process.reqs)
                )
                if all_req_finished:
                    decode_need_wait = False
                    decode_run_done = None
                    arm_decode_need_wait_after_prefill = False
                    overlap_log(
                        "process decode result: all req finished, set decode_need_wait=False"
                    )

            # process prefill result
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        nvtx.range_end(prefill_gpu_handle)
                        prefill_gpu_handle = None
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        # log prefill stats late
                        self.log_prefill_stats_late(self.split_prefill_batch)

                        # before merge, clear decode_result_queue
                        while decode_result_queue:
                            (
                                decode_batch_to_process,
                                decode_result_to_process,
                                decode_gpu_handle_to_process,
                            ) = decode_result_queue.popleft()
                            nvtx.range_end(decode_gpu_handle_to_process)
                            self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                            overlap_log(
                                f"before merge, clear decode_result_queue: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                            )

                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
                        decode_inflight = len(decode_result_queue) > 0
                        if decode_inflight:
                            # Keep decode dependency chain when previous decode steps are
                            # still in flight / pending CPU postprocess.
                            arm_decode_need_wait_after_prefill = False
                            overlap_log(
                                f"prefill finished and merged while decode inflight: keep decode_need_wait={decode_need_wait}, queue_len={len(decode_result_queue)}"
                            )
                        else:
                            # If decode pipeline is idle, the 1st decode after prefill
                            # can run without wait; the 2nd decode should wait again.
                            decode_need_wait = False
                            decode_run_done = None
                            arm_decode_need_wait_after_prefill = True
                            overlap_log(
                                "prefill finished and merged: set decode_need_wait=False, arm_after_prefill=True"
                            )

            # update decode last batch
            decode_last_batch = decode_batch

    def _max_seq_len_for_offline_prefill(self: Scheduler, prefill_batch: "ScheduleBatch") -> int:
        """Match bench_replay keys: max per-request length in the extend/prefill batch."""
        n = len(prefill_batch.reqs)
        slc = prefill_batch.seq_lens_cpu
        if slc is not None and slc.numel() == n:
            return int(max(int(x) for x in slc.tolist()))
        if n > 0:
            return max(int(req.extend_input_len) for req in prefill_batch.reqs)
        return 0

    def _predict_decode_run_time_from_map(
        self: Scheduler,
        stream_idx: int,
        decode_batch: Optional["ScheduleBatch"],
    ) -> Optional[float]:
        tables = getattr(self, "_pdmux_offline_tables", None)
        if tables is None or decode_batch is None or decode_batch.is_empty():
            return None
        return tables.lookup_decode(stream_idx, decode_batch.batch_size(), metric="gpu_run_ms")

    def _predict_prefill_run_time_from_map(
        self: Scheduler,
        stream_idx: int,
        prefill_batch: Optional["ScheduleBatch"],
    ) -> Optional[float]:
        tables = getattr(self, "_pdmux_offline_tables", None)
        if tables is None or prefill_batch is None or prefill_batch.is_empty():
            return None
        bs = prefill_batch.batch_size()
        max_len = self._max_seq_len_for_offline_prefill(prefill_batch)
        return tables.lookup_prefill(stream_idx, bs, max_len, metric="gpu_run_ms")

    def _compute_max_overlap_decode_round(
        self: Scheduler,
        stream_idx: int,
        decode_batch: Optional["ScheduleBatch"] = None,
        prefill_batch: Optional["ScheduleBatch"] = None,
    ) -> int:
        """
        Compute the maximum overlap decode round for the given stream index.
        Uses offline table lookup: floor(prefill_ms / decode_ms), clamped.
        Stream group 0 (prefill-only) and last (decode-only) skip table lookup and return default.
        """
        last_sg = len(self.stream_groups) - 1
        default_rounds = 2
        if stream_idx == 0 or stream_idx == last_sg:
            return default_rounds
        if getattr(self, "_pdmux_offline_tables", None) is None:
            return default_rounds
        pre_ms = self._predict_prefill_run_time_from_map(stream_idx, prefill_batch)
        dec_ms = self._predict_decode_run_time_from_map(stream_idx, decode_batch)
        if pre_ms is None or dec_ms is None or dec_ms <= 0:
            return default_rounds
        ratio = pre_ms / dec_ms
        rounds = max(1, int(ratio))
        return min(rounds, 64)

    def _overlap_latency_prefill_plan(
        self: Scheduler,
        stream_idx: int,
        split_batch: Optional["ScheduleBatch"],
        decode_batch: Optional["ScheduleBatch"],
    ) -> Optional[Dict[str, Any]]:
        """Offline plan for split-prefill layer count (no-double-launch loop).

        Match bench identity (ms):

            decode_gpu_run_ms ≈ L * per_layer_prefill_cpu_launch_only_ms
                + decode_cpu_prepare_and_launch_ms

        with ``per_layer = prefill_cpu_launch_only_full_model_ms / num_hidden_layers``.
        """
        tables: Optional[PDMuxOfflineTables] = getattr(self, "_pdmux_offline_tables", None)
        if (
            tables is None
            or split_batch is None
            or split_batch.is_empty()
            or decode_batch is None
            or decode_batch.is_empty()
        ):
            return None
        H = int(self.model_config.num_hidden_layers)
        if H <= 0:
            return None
        dec_bs = decode_batch.batch_size()
        dec_gpu = tables.lookup_decode(stream_idx, dec_bs, metric="gpu_run_ms")
        dec_cpu_pl = tables.lookup_decode(
            stream_idx, dec_bs, metric="cpu_prepare_and_launch_ms"
        )
        if dec_gpu is None or dec_cpu_pl is None:
            return None
        p_bs = split_batch.batch_size()
        p_msl = self._max_seq_len_for_offline_prefill(split_batch)
        full_lo = tables.lookup_prefill(
            stream_idx, p_bs, p_msl, metric="cpu_launch_only_ms"
        )
        if full_lo is None or full_lo <= 0:
            return None
        per_layer_lo = float(full_lo) / float(H)
        if per_layer_lo <= 0:
            return None
        budget = float(dec_gpu) - float(dec_cpu_pl)
        if budget <= 0:
            L = 1
        else:
            L = max(1, int(budget / per_layer_lo))
        remain = H - int(split_batch.split_index)
        if remain <= 0:
            return None
        L = min(int(L), int(remain))
        return {
            "L": int(L),
            "dec_gpu_ms": float(dec_gpu),
            "dec_cpu_pl_ms": float(dec_cpu_pl),
            "prefill_lo_full_ms": float(full_lo),
            "per_layer_lo_ms": float(per_layer_lo),
            "budget_ms": float(budget),
            "remain_layers": int(remain),
        }

    @torch.inference_mode()
    def event_loop_clever_overlap_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing with overlap."""
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # for nvtx profiling, we need to use the range_start and range_end
        loop_range_handle = None
        decode_forward_count = 0
        prefill_whole_batch_count = 0
        prefill_forward_count = 0
        prefill_gpu_handle = None
        prefill_launch_handle = None
        # Recorded on prefill_stream after the last split-prefill layer; used with .query().
        prefill_exe_done = None

        decode_result_queue = deque()
        # Keep compatibility with shared overlap checks (e.g. flush_cache/_is_no_request).
        self.result_queue = decode_result_queue
        decode_last_batch = None
        decode_need_wait = False
        decode_run_done = None
        decode_run_done_step = None
        arm_decode_need_wait_after_prefill = False

        # `max_overlap_decode_round` / `remain_overlap_decode_round`:
        # - Decremented on every decode launch (first + optional double-launch); resets
        #   when `adjust_stream_group` runs. While remain > 0, double-launch may queue a
        #   second decode before prefill in the same loop iteration.
        # - When remain_overlap_decode_round <= 0, we drain `decode_result_queue` fully
        #   before any new decode launch (see decode block): no pending results while in
        #   the "sync" regime — avoids stacking another async decode on top of queued
        #   results from the last overlap burst.
        max_overlap_decode_round = 2
        remain_overlap_decode_round = max_overlap_decode_round
        last_stream_group_idx = len(self.stream_groups) - 1

        # Second decode launch before prefill improves GPU utilization but can queue an
        # extra decode after EOS; overlap skips those results on CPU — spill KV is trimmed
        # in process_batch_result_decode via free_overlap_decode_kv_spill_before_finish.
        enable_double_launch_before_prefill = not getattr(
            self.server_args, "pdmux_disable_double_launch_before_prefill", False
        )
        double_launch_before_prefill = False

        def get_sg_msg():
            return f"stream_group: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"

        OVERLAP_LOG_LEVEL = True
        def overlap_log(message: str):
            if OVERLAP_LOG_LEVEL:
                logger.info(f"[pdmux-overlap] {message}")

        overlap_log(
            f"start loop stream_group={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}"
        )

        while True:
            if loop_range_handle is None:
                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")
            
            # decode recv requests
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            # prefill update batch
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            # decode update batch
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # Order decode work before drain / prefill_exe_done.query(): run on
                # decode_stream so synchronize + query reflect real GPU state (not stale
                # vs in-flight decode kernels from prior iterations).
                need_decode_sync_before_skip = (
                    stream_idx != last_stream_group_idx
                    and remain_overlap_decode_round <= 0
                ) or (
                    self.split_prefill_batch is not None
                    and self.split_prefill_batch.split_prefill_finished
                    and prefill_exe_done is not None
                )
                if need_decode_sync_before_skip:
                    decode_stream.synchronize()
                # Overlap budget exhausted: drain pending decode results before launch.
                if stream_idx != last_stream_group_idx and remain_overlap_decode_round <= 0:
                    while decode_result_queue:
                        (
                            decode_batch_to_process,
                            decode_result_to_process,
                            decode_gpu_handle_to_process,
                        ) = decode_result_queue.popleft()
                        nvtx.range_end(decode_gpu_handle_to_process)
                        overlap_log(
                            f"remain<=0: drain decode queue before update, queue_len_after={len(decode_result_queue)}"
                        )
                        self.process_batch_result(
                            decode_batch_to_process, decode_result_to_process
                        )
                    decode_last_batch = None
                # Last split-prefill layer done on GPU: skip decode this iteration to merge
                # next — must not prepare_for_decode(); same flag gates update + run_batch.
                skip_decode_before_prefill_merge = (
                    self.split_prefill_batch is not None
                    and self.split_prefill_batch.split_prefill_finished
                    and prefill_exe_done is not None
                    and prefill_exe_done.query()
                )
                self.running_batch = self.update_running_batch(
                    self.running_batch,
                    skip_prepare_decode=skip_decode_before_prefill_merge,
                )
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    # Idle memory check assumes no in-flight decode results: each queued
                    # decode still holds KV until process_batch_result runs. Double-launch
                    # can also queue an extra step after EOS; draining avoids false "leak".
                    while decode_result_queue:
                        (
                            decode_batch_to_process,
                            decode_result_to_process,
                            decode_gpu_handle_to_process,
                        ) = decode_result_queue.popleft()
                        nvtx.range_end(decode_gpu_handle_to_process)
                        self.process_batch_result(
                            decode_batch_to_process, decode_result_to_process
                        )
                    decode_last_batch = None
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            # adjust stream group
            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()

                while decode_result_queue:
                    (
                        decode_batch_to_process,
                        decode_result_to_process,
                        decode_gpu_handle_to_process,
                    ) = decode_result_queue.popleft()
                    nvtx.range_end(decode_gpu_handle_to_process)
                    self.process_batch_result(
                        decode_batch_to_process, decode_result_to_process
                    )
                    all_req_finished = (
                        len(decode_batch_to_process.reqs) > 0
                        and all(req.finished() for req in decode_batch_to_process.reqs)
                    )
                    if all_req_finished:
                        decode_need_wait = False
                        decode_run_done = None
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "drain decode queue: all req finished, set decode_need_wait=False"
                        )
                # Drain already processed all pending results from previous loops.
                # Reset decode_last_batch so the pop-and-process guard below does not
                # fire on the batch we are about to launch in this same loop.
                decode_last_batch = None
                overlap_log("drain done: reset decode_last_batch=None")

                nvtx.range_end(loop_range_handle)
                loop_range_handle = None

                stream_idx, stream_group = self.adjust_stream_groups()
                last_stream_group_idx = len(self.stream_groups) - 1
                decode_for_pred = (
                    self.running_batch
                    if self.running_batch is not None
                    and not self.running_batch.is_empty()
                    else None
                )
                # New stream group is a natural "safe point" to restart overlap budget.
                max_overlap_decode_round = self._compute_max_overlap_decode_round(
                    stream_idx,
                    decode_batch=decode_for_pred,
                    prefill_batch=self.split_prefill_batch,
                )
                remain_overlap_decode_round = max_overlap_decode_round
                if (
                    enable_double_launch_before_prefill
                    and stream_idx != 0
                    and stream_idx != last_stream_group_idx
                ):
                    double_launch_before_prefill = True

                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )
                overlap_log(
                    f"adjust stream group -> idx={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}"
                )

            decode_batch = None
            # run decode batch
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if (
                    self.running_batch
                    and not self.running_batch.is_empty()
                    and not skip_decode_before_prefill_merge
                ):
                    # skip_decode_before_prefill_merge: see flag above (no prepare, no launch).
                    decode_batch = self.running_batch
                    decode_gpu_handle = nvtx.range_start(
                        "run decode batch"
                        + f": {decode_forward_count} "
                        + _pdmux_nvtx_len_list_str(
                            decode_batch, add_decode_global_num_tokens=True
                        )
                    )
                    decode_forward_count += 1
                    if decode_need_wait and decode_run_done is not None:
                        overlap_log(
                            f"decode wait event before launch, prev_step={decode_run_done_step}, queue_len={len(decode_result_queue)}"
                        )
                        decode_stream.wait_event(decode_run_done)
                    # Attach a stable step id for cross-module logging (run_batch/copy_done).
                    decode_step = decode_forward_count - 1
                    setattr(decode_batch, "_pdmux_decode_step", decode_step)
                    decode_result = self.run_batch(decode_batch)
                    remain_overlap_decode_round -= 1
                    decode_run_done = decode_stream.record_event()
                    decode_run_done_step = decode_step
                    overlap_log(
                        f"decode_run_done <- decode_stream.record_event (fallback), step={decode_step}"
                    )
                    decode_result_queue.append(
                        (decode_batch.copy(), decode_result, decode_gpu_handle)
                    )
                    overlap_log(
                        f"launch decode batch#{decode_forward_count-1}, bs={decode_batch.batch_size()}, need_wait={decode_need_wait}, arm_after_prefill={arm_decode_need_wait_after_prefill}, queue_len={len(decode_result_queue)}"
                    )
                    if double_launch_before_prefill:
                        double_launch_before_prefill = False
                        if remain_overlap_decode_round > 0:
                            # Opportunistic 2nd decode launch before prefill launch to reduce decode bubbles.
                            # This is only safe if we:
                            #  - update_running_batch() again to prepare next-step tensors (likely future placeholders)
                            #  - honor decode_need_wait/decode_run_done before resolve_future+forward
                            #  - do NOT exceed overlap budget
                            #
                            # NOTE: This does not require CPU postprocess of the previous step, because
                            # ScheduleBatch.output_ids has already been updated to future indices by run_batch.

                            # Prepare next decode step immediately.
                            self.running_batch = self.update_running_batch(self.running_batch)
                            if self.running_batch and (not self.running_batch.is_empty()):
                                decode_batch2 = self.running_batch
                                decode_gpu_handle2 = nvtx.range_start(
                                    "run decode batch"
                                    + f": {decode_forward_count} "
                                    + _pdmux_nvtx_len_list_str(
                                        decode_batch2, add_decode_global_num_tokens=True
                                    )
                                )
                                decode_forward_count += 1

                                # If the next step depends on previous decode (future placeholders),
                                # wait for the event recorded after store_to_map.
                                if decode_need_wait and decode_run_done is not None:
                                    overlap_log(
                                        f"double-launch: decode wait event before launch, prev_step={decode_run_done_step}, queue_len={len(decode_result_queue)}"
                                    )
                                    decode_stream.wait_event(decode_run_done)

                                decode_step2 = decode_forward_count - 1
                                setattr(decode_batch2, "_pdmux_decode_step", decode_step2)
                                decode_result2 = self.run_batch(decode_batch2)
                                remain_overlap_decode_round -= 1

                                forward_done_evt2 = getattr(
                                    decode_result2, "_pdmux_decode_run_done", None
                                )
                                if forward_done_evt2 is not None:
                                    decode_run_done = forward_done_evt2
                                    decode_run_done_step = decode_step2
                                    overlap_log(
                                        f"double-launch: decode_run_done <- forward_done_evt, step={decode_step2}"
                                    )
                                else:
                                    decode_run_done = decode_stream.record_event()
                                    decode_run_done_step = decode_step2
                                    overlap_log(
                                        f"double-launch: decode_run_done <- decode_stream.record_event (fallback), step={decode_step2}"
                                    )

                                decode_result_queue.append(
                                    (decode_batch2.copy(), decode_result2, decode_gpu_handle2)
                                )
                                overlap_log(
                                    f"double-launch decode batch#{decode_step2}, bs={decode_batch2.batch_size()}, need_wait={decode_need_wait}, queue_len={len(decode_result_queue)}, remain_budget={remain_overlap_decode_round}"
                                )
                    if arm_decode_need_wait_after_prefill:
                        # Strict sync semantics: the 2nd decode after prefill must wait.
                        decode_need_wait = True
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "launch decode batch: consume arm_after_prefill, set decode_need_wait=True"
                        )

            # run prefill batch
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )    
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    if prefill_gpu_handle is None:
                        prefill_gpu_handle = nvtx.range_start(
                        "run prefill batch"
                        + f": {prefill_whole_batch_count} "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch))
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start(
                        "launch prefill batch"
                        + f": {prefill_forward_count} "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                    )
                    prefill_forward_count += 1
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    nvtx.range_end(prefill_launch_handle)
                    prefill_launch_handle = None
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            # process decode result
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if decode_last_batch and len(decode_result_queue) > 0:
                    (
                        decode_batch_to_process,
                        decode_result_to_process,
                        decode_gpu_handle_to_process,
                    ) = decode_result_queue.popleft()
                    nvtx.range_end(decode_gpu_handle_to_process)
                    overlap_log(
                        f"process decode result: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                    )
                    self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                    
                    is_all_decode_group = stream_idx == last_stream_group_idx
                    if (not is_all_decode_group) and (remain_overlap_decode_round <= 0) and decode_result_queue:
                        # force sync decode_stream before drain decode_result_queue
                        decode_stream.synchronize()
                        overlap_log(
                            f"force_sync_decode: drain decode queue before launch, queue_len={len(decode_result_queue)}"
                        )
                        while decode_result_queue:
                            (
                                decode_batch_to_process,
                                decode_result_to_process,
                                decode_gpu_handle_to_process,
                            ) = decode_result_queue.popleft()
                            nvtx.range_end(decode_gpu_handle_to_process)
                            self.process_batch_result(
                                decode_batch_to_process, decode_result_to_process
                            )
                        decode_last_batch = None
                    
                    all_req_finished = (
                        len(decode_batch_to_process.reqs) > 0
                        and all(req.finished() for req in decode_batch_to_process.reqs)
                    )
                    if all_req_finished:
                        decode_need_wait = False
                        decode_run_done = None
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "process decode result: all req finished, set decode_need_wait=False"
                        )

            # process prefill result and merge prefill batch into running batch
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        nvtx.range_end(prefill_gpu_handle)
                        prefill_gpu_handle = None
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        # log prefill stats late
                        self.log_prefill_stats_late(self.split_prefill_batch)

                        # before merge, clear decode_result_queue and sync decode_stream
                        while decode_result_queue:
                            (
                                decode_batch_to_process,
                                decode_result_to_process,
                                decode_gpu_handle_to_process,
                            ) = decode_result_queue.popleft()
                            nvtx.range_end(decode_gpu_handle_to_process)
                            self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                            overlap_log(
                                f"before merge, clear decode_result_queue: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                            )
                        decode_stream.synchronize()

                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
                        # TODO(lbz): just change!!
                        decode_inflight = len(decode_result_queue) > 0
                        if decode_inflight:
                            # Keep decode dependency chain when previous decode steps are
                            # still in flight / pending CPU postprocess.
                            arm_decode_need_wait_after_prefill = False
                            overlap_log(
                                f"prefill finished and merged while decode inflight: keep decode_need_wait={decode_need_wait}, queue_len={len(decode_result_queue)}"
                            )
                        else:
                            # If decode pipeline is idle, the 1st decode after prefill
                            # can run without wait; the 2nd decode should wait again.
                            decode_need_wait = False
                            decode_run_done = None
                            arm_decode_need_wait_after_prefill = True
                            overlap_log(
                                "prefill finished and merged: set decode_need_wait=False, arm_after_prefill=True"
                            )

            # update decode last batch
            decode_last_batch = decode_batch

    @torch.inference_mode()
    def event_loop_clever_overlap_pdmux_no_double_launch(self: Scheduler):
        """PD multiplex overlap without double decode; split-prefill depth from offline latency when available."""
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # for nvtx profiling, we need to use the range_start and range_end
        loop_range_handle = None
        decode_forward_count = 0
        prefill_whole_batch_count = 0
        prefill_forward_count = 0
        prefill_gpu_handle = None
        prefill_launch_handle = None
        # Recorded on prefill_stream after the last split-prefill layer; used with .query().
        prefill_exe_done = None

        decode_result_queue = deque()
        # Keep compatibility with shared overlap checks (e.g. flush_cache/_is_no_request).
        self.result_queue = decode_result_queue
        decode_last_batch = None
        decode_need_wait = False
        decode_run_done = None
        decode_run_done_step = None
        arm_decode_need_wait_after_prefill = False

        # ``max_overlap_decode_round`` / ``remain_overlap_decode_round``: decremented on
        # each decode ``run_batch``; reset on ``adjust_stream_group``. When budget is
        # exhausted, drain queued decode results before launching another decode.
        max_overlap_decode_round = 2
        remain_overlap_decode_round = max_overlap_decode_round
        last_stream_group_idx = len(self.stream_groups) - 1

        def get_sg_msg():
            return f"stream_group: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"

        OVERLAP_LOG_LEVEL = False
        def overlap_log(message: str):
            if OVERLAP_LOG_LEVEL:
                logger.info(f"[pdmux-overlap] {message}")


        overlap_log(
            f"pdmux-overlap-no-double-launch: start loop"
            f"stream_group={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}"
        )

        while True:
            if loop_range_handle is None:
                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")
            
            # decode recv requests
            recv_requests_handle = nvtx.range_start("recv_requests")
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
            nvtx.range_end(recv_requests_handle)

            # prefill update batch
            update_split_prefill_batch_handle = nvtx.range_start("update_split_prefill_batch")
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )
            nvtx.range_end(update_split_prefill_batch_handle)

            # decode update batch
            update_running_batch_handle = nvtx.range_start("update_running_batch")
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # Order decode work before drain / prefill_exe_done.query(): run on
                # decode_stream so synchronize + query reflect real GPU state (not stale
                # vs in-flight decode kernels from prior iterations).
                need_decode_sync_before_skip = (
                    stream_idx != last_stream_group_idx
                    and remain_overlap_decode_round <= 0
                ) or (
                    self.split_prefill_batch is not None
                    and self.split_prefill_batch.split_prefill_finished
                    and prefill_exe_done is not None
                    and prefill_exe_done.query() # DEBUG(lbz)
                )
                if need_decode_sync_before_skip or (stream_idx != last_stream_group_idx and remain_overlap_decode_round <= 0):
                    while decode_result_queue:
                        (
                            decode_batch_to_process,
                            decode_result_to_process,
                            decode_gpu_handle_to_process,
                        ) = decode_result_queue.popleft()
                        nvtx.range_end(decode_gpu_handle_to_process)
                        overlap_log(
                            f"remain<=0: drain decode queue before update, queue_len_after={len(decode_result_queue)}"
                        )
                        self.process_batch_result(
                            decode_batch_to_process, decode_result_to_process
                        )
                    # sync decode_stream after drain decode_result_queue, save the time for decode process_batch_result
                    decode_stream.synchronize()
                    decode_last_batch = None
                # Last split-prefill layer done on GPU: skip decode this iteration to merge
                # next — must not prepare_for_decode(); same flag gates update + run_batch.
                skip_decode_before_prefill_merge = (
                    self.split_prefill_batch is not None
                    and self.split_prefill_batch.split_prefill_finished
                    and prefill_exe_done is not None
                    and prefill_exe_done.query()
                )
                self.running_batch = self.update_running_batch(
                    self.running_batch,
                    skip_prepare_decode=skip_decode_before_prefill_merge,
                )
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    # Idle: drain queued decode results before memory checks.
                    while decode_result_queue:
                        (
                            decode_batch_to_process,
                            decode_result_to_process,
                            decode_gpu_handle_to_process,
                        ) = decode_result_queue.popleft()
                        nvtx.range_end(decode_gpu_handle_to_process)
                        self.process_batch_result(
                            decode_batch_to_process, decode_result_to_process
                        )
                    decode_last_batch = None
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()
            nvtx.range_end(update_running_batch_handle)

            # adjust stream group (no_double_launch loop)
            adjust_stream_group_handle = nvtx.range_start("adjust_stream_group")
            if adjust_stream_group:
                while decode_result_queue:
                    (
                        decode_batch_to_process,
                        decode_result_to_process,
                        decode_gpu_handle_to_process,
                    ) = decode_result_queue.popleft()
                    nvtx.range_end(decode_gpu_handle_to_process)
                    self.process_batch_result(
                        decode_batch_to_process, decode_result_to_process
                    )
                    all_req_finished = (
                        len(decode_batch_to_process.reqs) > 0
                        and all(req.finished() for req in decode_batch_to_process.reqs)
                    )
                    if all_req_finished:
                        decode_need_wait = False
                        decode_run_done = None
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "drain decode queue: all req finished, set decode_need_wait=False"
                        )
                # sync prefill_stream and decode_stream after drain decode_result_queue, save the time for process_batch_result
                prefill_stream.synchronize()
                decode_stream.synchronize()
                # Drain already processed all pending results from previous loops.
                # Reset decode_last_batch so the pop-and-process guard below does not
                # fire on the batch we are about to launch in this same loop.
                decode_last_batch = None
                overlap_log("drain done: reset decode_last_batch=None")

                nvtx.range_end(loop_range_handle)
                loop_range_handle = None

                stream_idx, stream_group = self.adjust_stream_groups()
                last_stream_group_idx = len(self.stream_groups) - 1
                decode_for_pred = (
                    self.running_batch
                    if self.running_batch is not None
                    and not self.running_batch.is_empty()
                    else None
                )
                # New stream group is a natural "safe point" to restart overlap budget.
                max_overlap_decode_round = self._compute_max_overlap_decode_round(
                    stream_idx,
                    decode_batch=decode_for_pred,
                    prefill_batch=self.split_prefill_batch,
                )
                remain_overlap_decode_round = max_overlap_decode_round

                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group" + f"max_overlap_decode_round={max_overlap_decode_round}")

                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )
                overlap_log(
                    f"adjust stream group -> idx={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}"
                )
            nvtx.range_end(adjust_stream_group_handle)

            decode_batch = None
            overlap_prefill_plan: Optional[Dict[str, Any]] = None
            # run decode batch
            run_decode_batch_handle = nvtx.range_start("run_decode_batch")
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if (
                    self.running_batch
                    and not self.running_batch.is_empty()
                    and not skip_decode_before_prefill_merge
                ):
                    # skip_decode_before_prefill_merge: see flag above (no prepare, no launch).
                    decode_batch = self.running_batch
                    overlap_prefill_plan = self._overlap_latency_prefill_plan(
                        stream_idx,
                        self.split_prefill_batch,
                        decode_batch,
                    )
                    plan_nvtx = ""
                    if overlap_prefill_plan is not None:
                        p = overlap_prefill_plan
                        plan_nvtx = (
                            f" | tbl dec_gpu_ms={p['dec_gpu_ms']:.3f}"
                            f" dec_cpu_pl_ms={p['dec_cpu_pl_ms']:.3f}"
                            f" prefill_lo_full_ms={p['prefill_lo_full_ms']:.3f}"
                            f" per_layer_lo_ms={p['per_layer_lo_ms']:.4f}"
                            f" budget_ms={p['budget_ms']:.3f}"
                            f" planned_prefill_L={p['L']}"
                        )
                        overlap_log(f"overlap_prefill_plan: {overlap_prefill_plan}")
                    else:
                        overlap_log(f"overlap_prefill_plan is None")
                    decode_gpu_handle = nvtx.range_start(
                        "launch decode "
                        + f"step={decode_forward_count} bs={decode_batch.batch_size()}"
                        + plan_nvtx
                        + " | "
                        + _pdmux_nvtx_len_list_str(
                            decode_batch, add_decode_global_num_tokens=True
                            )
                        + "|"
                        + f"remain_overlap_decode_round={remain_overlap_decode_round}"
                        + f"max_overlap_decode_round={max_overlap_decode_round}"
                    )
                    decode_forward_count += 1
                    if decode_need_wait and decode_run_done is not None:
                        overlap_log(
                            f"decode wait event before launch, prev_step={decode_run_done_step}, queue_len={len(decode_result_queue)}"
                        )
                        decode_stream.wait_event(decode_run_done)
                    # Attach a stable step id for cross-module logging (run_batch/copy_done).
                    decode_step = decode_forward_count - 1
                    setattr(decode_batch, "_pdmux_decode_step", decode_step)
                    rb_decode = nvtx.range_start(
                        "run_batch decode forward" + plan_nvtx
                    )
                    decode_result = self.run_batch(decode_batch)
                    nvtx.range_end(rb_decode)
                    remain_overlap_decode_round -= 1
                    decode_run_done = decode_stream.record_event()
                    decode_run_done_step = decode_step
                    overlap_log(
                        f"decode_run_done <- decode_stream.record_event (fallback), step={decode_step}"
                    )
                    decode_result_queue.append(
                        (decode_batch.copy(), decode_result, decode_gpu_handle)
                    )
                    overlap_log(
                        f"launch decode batch#{decode_forward_count-1}, bs={decode_batch.batch_size()}, need_wait={decode_need_wait}, arm_after_prefill={arm_decode_need_wait_after_prefill}, queue_len={len(decode_result_queue)}"
                    )
                    if arm_decode_need_wait_after_prefill:
                        # Strict sync semantics: the 2nd decode after prefill must wait.
                        decode_need_wait = True
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "launch decode batch: consume arm_after_prefill, set decode_need_wait=True"
                        )
            nvtx.range_end(run_decode_batch_handle)

            # run prefill batch
            run_prefill_batch_handle = nvtx.range_start("run_prefill_batch")
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    H = int(self.model_config.num_hidden_layers)
                    idx = int(self.split_prefill_batch.split_index)
                    remain_layers = max(0, H - idx)
                    forward_count_base = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else H
                    )
                    forward_count = min(forward_count_base, remain_layers) if remain_layers > 0 else 0
                    if forward_count <= 0:
                        forward_count = min(H, remain_layers) if remain_layers > 0 else 1
                    if overlap_prefill_plan is not None:
                        forward_count = min(
                            forward_count,
                            int(overlap_prefill_plan["L"]),
                            remain_layers,
                        )
                    forward_count = max(1, forward_count)
                    next_split_index = min(idx + forward_count, H)
                    forward_count = next_split_index - idx

                    self.split_prefill_batch.split_forward_count = forward_count
                    prefill_nvtx_plan = ""
                    if overlap_prefill_plan is not None:
                        p = overlap_prefill_plan
                        prefill_nvtx_plan = (
                            f" | tbl L={p['L']} dec_gpu={p['dec_gpu_ms']:.3f}"
                            f" dec_cpu_pl={p['dec_cpu_pl_ms']:.3f} lo/L={p['per_layer_lo_ms']:.4f}"
                        )
                    if prefill_gpu_handle is None:
                        prefill_gpu_handle = nvtx.range_start(
                            "run prefill batch"
                            + f": {prefill_whole_batch_count} "
                            + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                        )
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start(
                        "launch prefill split_forward "
                        + f": {prefill_forward_count} L={forward_count}{prefill_nvtx_plan} | "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                    )
                    prefill_forward_count += 1
                    rb_prefill = nvtx.range_start(
                        "run_batch prefill forward "
                        + f"L={forward_count}{prefill_nvtx_plan}"
                    )
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    nvtx.range_end(rb_prefill)
                    nvtx.range_end(prefill_launch_handle)
                    prefill_launch_handle = None
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False
            nvtx.range_end(run_prefill_batch_handle)

            # process decode result
            process_decode_result_handle = nvtx.range_start("process_decode_result")
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if decode_last_batch and len(decode_result_queue) > 0:
                    (
                        decode_batch_to_process,
                        decode_result_to_process,
                        decode_gpu_handle_to_process,
                    ) = decode_result_queue.popleft()
                    nvtx.range_end(decode_gpu_handle_to_process)
                    overlap_log(
                        f"process decode result: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                    )
                    self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                    
                    is_all_decode_group = stream_idx == last_stream_group_idx
                    if (not is_all_decode_group) and (remain_overlap_decode_round <= 0) and decode_result_queue:
                        # force sync decode_stream before drain decode_result_queue
                        overlap_log(
                            f"force_sync_decode: drain decode queue before launch, queue_len={len(decode_result_queue)}"
                        )
                        while decode_result_queue:
                            (
                                decode_batch_to_process,
                                decode_result_to_process,
                                decode_gpu_handle_to_process,
                            ) = decode_result_queue.popleft()
                            nvtx.range_end(decode_gpu_handle_to_process)
                            self.process_batch_result(
                                decode_batch_to_process, decode_result_to_process
                            )
                        decode_stream.synchronize()
                        decode_last_batch = None
                    
                    all_req_finished = (
                        len(decode_batch_to_process.reqs) > 0
                        and all(req.finished() for req in decode_batch_to_process.reqs)
                    )
                    if all_req_finished:
                        decode_need_wait = False
                        decode_run_done = None
                        arm_decode_need_wait_after_prefill = False
                        overlap_log(
                            "process decode result: all req finished, set decode_need_wait=False"
                        )
            nvtx.range_end(process_decode_result_handle)

            # process prefill result and merge prefill batch into running batch
            process_prefill_result_and_merge_prefill_batch_into_running_batch_handle = nvtx.range_start("process_prefill_result_and_merge_prefill_batch_into_running_batch")
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        nvtx.range_end(prefill_gpu_handle)
                        prefill_gpu_handle = None
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        # log prefill stats late
                        self.log_prefill_stats_late(self.split_prefill_batch)

                        # before merge, clear decode_result_queue and sync decode_stream
                        while decode_result_queue:
                            (
                                decode_batch_to_process,
                                decode_result_to_process,
                                decode_gpu_handle_to_process,
                            ) = decode_result_queue.popleft()
                            nvtx.range_end(decode_gpu_handle_to_process)
                            self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                            overlap_log(
                                f"before merge, clear decode_result_queue: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                            )
                        decode_stream.synchronize()

                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
                        # TODO(lbz): just change!!
                        decode_inflight = len(decode_result_queue) > 0
                        if decode_inflight:
                            # Keep decode dependency chain when previous decode steps are
                            # still in flight / pending CPU postprocess.
                            arm_decode_need_wait_after_prefill = False
                            overlap_log(
                                f"prefill finished and merged while decode inflight: keep decode_need_wait={decode_need_wait}, queue_len={len(decode_result_queue)}"
                            )
                        else:
                            # If decode pipeline is idle, the 1st decode after prefill
                            # can run without wait; the 2nd decode should wait again.
                            decode_need_wait = False
                            decode_run_done = None
                            arm_decode_need_wait_after_prefill = True
                            overlap_log(
                                "prefill finished and merged: set decode_need_wait=False, arm_after_prefill=True"
                            )
            nvtx.range_end(process_prefill_result_and_merge_prefill_batch_into_running_batch_handle)
            # update decode last batch
            decode_last_batch = decode_batch

    @torch.inference_mode()
    def event_loop_overlap_pdmux_minimal(self: Scheduler):
        """A scheduler loop for pd multiplexing with overlap."""
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # for nvtx profiling, we need to use the range_start and range_end
        loop_range_handle = None
        decode_forward_count = 0
        prefill_whole_batch_count = 0
        prefill_forward_count = 0
        prefill_gpu_handle = None
        prefill_launch_handle = None

        decode_result_queue = deque()
        # Keep compatibility with shared overlap checks (e.g. flush_cache/_is_no_request).
        self.result_queue = decode_result_queue
        decode_last_batch = None
        prefill_exe_done = None
        # Budget of async (overlapped) decode launches allowed while prefill runs.
        # Reset to max at every stream-group adjustment; decremented each launch.
        # When exhausted, fall back to strict-sync mode (drain-or-immediate).
        remain_overlap_decode_round = 0

        def get_sg_msg():
            return f"stream_group: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"

        OVERLAP_LOG_LEVEL = False
        def overlap_log(message: str):
            if OVERLAP_LOG_LEVEL:
                logger.info(f"[pdmux-overlap] {message}")

        overlap_log(
            f"start loop stream_group={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}"
        )

        while True:
            if loop_range_handle is None:
                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

            # Early check: has the prefill GPU kernel already finished?
            # Queried before update_running_batch to skip GPU memory allocations
            # for newly admitted decode requests right before the prefill-merge step.
            prefill_gpu_done = (
                prefill_exe_done is not None
                and self.split_prefill_batch is not None
                and getattr(self.split_prefill_batch, "split_prefill_finished", False)
                and prefill_exe_done.query()
            )

            # decode recv requests
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)

            # prefill update batch
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )

            # decode update batch — skipped when prefill GPU is already done to avoid
            # allocating memory for new decode requests right before the merge step.
            if not prefill_gpu_done:
                # If we know upfront that this iteration will skip the decode launch
                # (strict-sync Case A: budget exhausted and queue still has pending
                # results to drain), pass skip_prepare_decode=True so that
                # prepare_for_decode() does NOT set output_ids = None.
                # If output_ids is None when merge_batch() runs, the merge condition
                # `if self.output_ids is not None` fails and output_ids stays None,
                # causing input_ids = None in the next prepare_for_decode() call.
                _will_skip_decode = (
                    self.split_prefill_batch is not None   # prefill_active
                    and remain_overlap_decode_round <= 0
                    and bool(decode_result_queue)          # Case A: queue not empty
                )
                with torch.cuda.stream(decode_stream):
                    set_pdmux_status(False)
                    self.running_batch = self.update_running_batch(
                        self.running_batch,
                        skip_prepare_decode=_will_skip_decode,
                    )
                    adjust_stream_group = adjust_stream_group or (
                        stream_idx > 0 and self.running_batch.is_empty()
                    )
                    if self.running_batch.is_empty() and self.split_prefill_batch is None:
                        self.check_memory()
                        self.check_tree_cache()
                        self.new_token_ratio = self.init_new_token_ratio
                        self.maybe_sleep_on_idle()

            # adjust stream group
            if adjust_stream_group:

                while decode_result_queue:
                    (
                        decode_batch_to_process,
                        decode_result_to_process,
                        decode_gpu_handle_to_process,
                        _,
                    ) = decode_result_queue.popleft()
                    nvtx.range_end(decode_gpu_handle_to_process)
                    self.process_batch_result(
                        decode_batch_to_process, decode_result_to_process
                    )
                decode_stream.synchronize()
                prefill_stream.synchronize()
                # Drain already processed all pending results from previous loops.
                # Reset decode_last_batch so the pop-and-process guard below does not
                # fire on the batch we are about to launch in this same loop.
                decode_last_batch = None
                overlap_log("drain done: reset decode_last_batch=None")

                nvtx.range_end(loop_range_handle)
                loop_range_handle = None

                stream_idx, stream_group = self.adjust_stream_groups()

                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                remain_overlap_decode_round = self._compute_max_overlap_decode_round(
                    stream_idx,
                    decode_batch=self.running_batch,
                    prefill_batch=self.split_prefill_batch,
                )
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )
                overlap_log(
                    f"adjust stream group -> idx={stream_idx}, prefill_sm={self.sm_counts[stream_idx][0]}, decode_sm={self.sm_counts[stream_idx][1]}, remain_rounds={remain_overlap_decode_round}"
                )

            decode_batch = None
            overlap_prefill_plan: Optional[Dict[str, Any]] = None
            prefill_active = self.split_prefill_batch is not None

            # Decide how to handle the decode launch this round.
            #
            # Overlap mode  (prefill running, remain > 0):
            #   Launch decode concurrently with prefill on decode_stream; decrement
            #   the budget.  The result is processed next iteration via decode_last_batch.
            #
            # Strict-sync mode  (prefill running, remain <= 0):
            #   Case A — queue has pending results: drain one, skip launching this round.
            #   Case B — queue is empty: launch decode, then immediately launch prefill
            #             to overlap with the decode GPU kernel, then process the decode
            #             result (blocks on copy_done.synchronize).
            #
            # Prefill-done / no-prefill:
            #   Skip decode launch; let the loop reach the prefill-result section.
            skip_decode_launch = prefill_gpu_done
            strict_sync_immediate = False  # True when we must process the just-launched result immediately

            if not skip_decode_launch and prefill_active and remain_overlap_decode_round <= 0:
                # Strict-sync: budget exhausted while prefill is still running.
                if decode_result_queue:
                    # Case A: drain one pending overlapped result; skip new launch.
                    (
                        decode_batch_to_process,
                        decode_result_to_process,
                        decode_gpu_handle_to_process,
                        _,
                    ) = decode_result_queue.popleft()
                    nvtx.range_end(decode_gpu_handle_to_process)
                    overlap_log(
                        f"strict-sync drain: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                    )
                    self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                    skip_decode_launch = True
                    # Prevent decode_last_batch from re-processing a result this round.
                    decode_last_batch = None
                else:
                    # Case B: queue empty — will launch below, then launch prefill to
                    # overlap with decode GPU, then process decode result immediately.
                    strict_sync_immediate = True

            # run decode batch
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if (
                    not skip_decode_launch
                    and self.running_batch
                    and not self.running_batch.is_empty()
                ):
                    decode_batch = self.running_batch
                    # Compute offline layer-count plan: how many prefill layers fit
                    # inside the decode GPU execution window.  Used by the prefill
                    # launch section below; None when offline tables are unavailable.
                    overlap_prefill_plan = self._overlap_latency_prefill_plan(
                        stream_idx,
                        self.split_prefill_batch,
                        decode_batch,
                    )
                    plan_nvtx = ""
                    if overlap_prefill_plan is not None:
                        p = overlap_prefill_plan
                        plan_nvtx = (
                            f" | tbl dec_gpu={p['dec_gpu_ms']:.3f}"
                            f" dec_cpu_pl={p['dec_cpu_pl_ms']:.3f}"
                            f" prefill_launch_full_ms={p['prefill_lo_full_ms']:.3f}"
                            f" lo/L={p['per_layer_lo_ms']:.4f}"
                            f" bgt={p['budget_ms']:.3f}"
                            f" L={p['L']}"
                        )
                        overlap_log(f"overlap_prefill_plan: {overlap_prefill_plan}")
                    decode_gpu_handle = nvtx.range_start(
                        "run decode batch"
                        + f": {decode_forward_count} "
                        + f"remain={remain_overlap_decode_round} "
                        + f"strict_imm={strict_sync_immediate}"
                        + plan_nvtx
                        + " | "
                        + _pdmux_nvtx_len_list_str(
                            decode_batch, add_decode_global_num_tokens=True
                        )
                    )
                    decode_forward_count += 1
                    decode_step = decode_forward_count - 1
                    setattr(decode_batch, "_pdmux_decode_step", decode_step)
                    decode_result = self.run_batch(decode_batch)
                    # Keep a strong ref to model_worker_batch so its GPU tensors
                    # are not freed while the kernel is still in flight.
                    # batch_record_buf is a 2-slot ring that gets overwritten every
                    # 2 decode launches; grabbing the slot right after run_batch
                    # ensures this result owns a reference for its lifetime in queue.
                    _mwb_ref = self.batch_record_buf[self.batch_record_ct]
                    decode_result_queue.append(
                        (decode_batch.copy(), decode_result, decode_gpu_handle, _mwb_ref)
                    )
                    if prefill_active:
                        remain_overlap_decode_round -= 1
                    overlap_log(
                        f"launch decode#{decode_step}, bs={decode_batch.batch_size()}, remain={remain_overlap_decode_round}, strict_imm={strict_sync_immediate}"
                    )

            # run prefill batch
            # In strict-sync Case B the prefill launch is deliberately placed HERE,
            # between the decode launch and the immediate process step, so that the
            # CPU can submit prefill kernels to prefill_stream while the decode kernel
            # is already running on the GPU — maximising CPU-GPU overlap before we
            # block on copy_done.synchronize() in process_batch_result below.
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    H = int(self.model_config.num_hidden_layers)
                    idx = int(self.split_prefill_batch.split_index)
                    remain_layers = max(0, H - idx)
                    # Base forward count from token-budget heuristic (fallback).
                    forward_count_base = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else H
                    )
                    forward_count = (
                        min(forward_count_base, remain_layers) if remain_layers > 0 else 0
                    )
                    if forward_count <= 0:
                        forward_count = min(H, remain_layers) if remain_layers > 0 else 1
                    # Tighten with offline plan when available: only launch as many
                    # layers as the CPU can submit inside the decode GPU window.
                    if overlap_prefill_plan is not None:
                        forward_count = min(
                            forward_count,
                            int(overlap_prefill_plan["L"]),
                            remain_layers,
                        )
                    # DEBUG(lbz): 
                    # if remain_overlap_decode_round <= 0:
                    #     forward_count = remain_layers
                    forward_count = max(1, forward_count)
                    next_split_index = min(idx + forward_count, H)
                    forward_count = next_split_index - idx

                    self.split_prefill_batch.split_forward_count = forward_count
                    prefill_nvtx_plan = ""
                    if overlap_prefill_plan is not None:
                        p = overlap_prefill_plan
                        prefill_nvtx_plan = (
                            f" | tbl L={p['L']} dec_gpu={p['dec_gpu_ms']:.3f}"
                            f" dec_cpu_pl={p['dec_cpu_pl_ms']:.3f} lo/L={p['per_layer_lo_ms']:.4f}"
                        )
                    if prefill_gpu_handle is None:
                        prefill_gpu_handle = nvtx.range_start(
                            "run prefill batch"
                            + f": {prefill_whole_batch_count} "
                            + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                        )
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start(
                        "launch prefill batch"
                        + f": {prefill_forward_count} L={forward_count}{prefill_nvtx_plan} | "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                    )
                    prefill_forward_count += 1
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    nvtx.range_end(prefill_launch_handle)
                    prefill_launch_handle = None
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            # Strict-sync immediate (Case B): process the just-launched decode result
            # now that prefill kernels have been submitted.  The decode kernel has been
            # running on the GPU since the launch above; copy_done.synchronize() will
            # wait for it while the prefill GPU work runs concurrently.
            if strict_sync_immediate and decode_batch is not None:
                (
                    decode_batch_to_process,
                    decode_result_to_process,
                    decode_gpu_handle_to_process,
                    _,
                ) = decode_result_queue.popleft()
                nvtx.range_end(decode_gpu_handle_to_process)
                overlap_log(
                    f"strict-sync immediate process: bs={decode_batch_to_process.batch_size()}"
                )
                self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                # Mark as processed so decode_last_batch path does not re-process.
                decode_batch = None

            # process decode result (overlap-mode path: the previous iteration's launch)
            if decode_last_batch and len(decode_result_queue) > 0:
                (
                    decode_batch_to_process,
                    decode_result_to_process,
                    decode_gpu_handle_to_process,
                    _,
                ) = decode_result_queue.popleft()
                nvtx.range_end(decode_gpu_handle_to_process)
                overlap_log(
                    f"process decode result: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                )
                self.process_batch_result(decode_batch_to_process, decode_result_to_process)

            # process prefill result
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        nvtx.range_end(prefill_gpu_handle)
                        prefill_gpu_handle = None
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        # log prefill stats late
                        self.log_prefill_stats_late(self.split_prefill_batch)

                        # before merge, clear decode_result_queue
                        while decode_result_queue:
                            (
                                decode_batch_to_process,
                                decode_result_to_process,
                                decode_gpu_handle_to_process,
                                _,
                            ) = decode_result_queue.popleft()
                            nvtx.range_end(decode_gpu_handle_to_process)
                            self.process_batch_result(decode_batch_to_process, decode_result_to_process)
                            overlap_log(
                                f"before merge, clear decode_result_queue: bs={decode_batch_to_process.batch_size()}, queue_len_after={len(decode_result_queue)}"
                            )
                        prefill_stream.synchronize()
                        decode_stream.synchronize()

                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        prefill_exe_done = None
                        adjust_stream_group = True

            # update decode last batch
            decode_last_batch = decode_batch

    @torch.inference_mode()
    def event_loop_pdmux_for_special_dp_attention(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        # for nvtx profiling, we need to use the range_start and range_end
        loop_range_handle = None
        decode_forward_count = 0
        decode_gpu_handle = None
        prefill_whole_batch_count = 0
        prefill_forward_count = 0
        prefill_gpu_handle = None
        prefill_launch_handle = None

        def get_sg_msg():
            return f"stream_group: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"

        logger.info("Starting event loop for pd multiplexing...")

        while True:
            if loop_range_handle is None:
                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                if recv_reqs:
                    logger.info(f"in event_loop_pdmux, decode_stream: recv_reqs: {len(recv_reqs)}")
                self.process_input_requests(recv_reqs)
                
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )
                
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # TODO(lbz): in update_running_batch, we filter the batch, 
                if not self.running_batch.is_empty():
                    self.running_batch = self.update_running_batch(self.running_batch)
                    self.decode_or_idle_batch = self.running_batch if not self.running_batch.is_empty() else None
                else:
                    self.decode_or_idle_batch = None
                
                self.decode_or_idle_batch = self.maybe_prepare_mlp_sync_batch_and_log_stats(self.decode_or_idle_batch, need_sync=self.require_mlp_sync, log_stats=False)
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.decode_or_idle_batch is None
                )
                if self.decode_or_idle_batch is not None:
                    pass
                else:
                    pass
                # TODO(lbz): running_batch use mlp sync batch, so can it be empty?
                # if self.running_batch is not None and self.running_batch.is_empty() and self.split_prefill_batch is None:
                if self.decode_or_idle_batch is None and self.split_prefill_batch is None:
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()
                
            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()

                nvtx.range_end(loop_range_handle)
                loop_range_handle = None

                # TODO(lbz): we need make all ranks adjust to the same stream group
                stream_idx, stream_group = self.adjust_stream_groups_for_special_dp_attention()

                loop_range_handle = nvtx.range_start(get_sg_msg() + "adjust_stream_group")

                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                # logger.info(
                #    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                # )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if self.decode_or_idle_batch:
                    # logger.info(f"before run_batch, decode_or_idle_batch: {self.decode_or_idle_batch.batch_size()}")
                    # self.decode_forward_count += 1
                    # logger.info(f"decode_forward_count: {self.decode_forward_count}")
                    # logger.info("="*80)
                    # for req in self.decode_or_idle_batch.reqs:
                    #     # rid, len(fill_ids), len_output_ids, finished()
                    #     logger.info(f"req: {req.rid}, len(fill_ids): {len(req.fill_ids)}, len_output_ids: {len(req.output_ids)}, finished: {req.finished()}")
                    decode_nvtx_extra = ""
                    pred_dec = getattr(self, "_pdmux_time_predictor", None)
                    if pred_dec is not None:
                        lb_dec = decode_lb_from_schedule_batch(
                            self.decode_or_idle_batch
                        )
                        if lb_dec is not None:
                            Ld, Bd = lb_dec
                            dr_ms = pred_dec.predict_decode_run_ms(
                                stream_idx, Ld, Bd
                            )
                            decode_nvtx_extra = (
                                f" | pred_dr_ms={dr_ms:.4f} g={stream_idx}"
                            )
                    decode_gpu_handle = nvtx.range_start(
                        "run decode batch"
                        + f": {decode_forward_count} "
                        + _pdmux_nvtx_len_list_str(
                            self.decode_or_idle_batch,
                            add_decode_global_num_tokens=True,
                        )
                        + decode_nvtx_extra
                    )
                    decode_forward_count += 1
                    decode_result = self.run_batch(self.decode_or_idle_batch)
                    decode_done = True

                else:
                    decode_done = False
                
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    base_forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // self.split_prefill_batch.extend_num_tokens,
                        )
                        if self.split_prefill_batch.extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    forward_count = base_forward_count
                    # Optional: align split-prefill depth with predicted decode run vs prefill launch
                    pred = getattr(self, "_pdmux_time_predictor", None)
                    if pred is not None:
                        g = stream_idx
                        F, G = prefill_fg_from_schedule_batch(self.split_prefill_batch)
                        decode_run_pred = None
                        if self.decode_or_idle_batch is not None:
                            lb = decode_lb_from_schedule_batch(self.decode_or_idle_batch)
                            if lb is not None:
                                L, B = lb
                                decode_run_pred = pred.predict_decode_run_ms(g, L, B)
                        aligned_fc = pred.suggest_forward_count_from_decode_prefill_ratio(
                            g=g,
                            F=F,
                            G=G,
                            decode_run_pred_ms=decode_run_pred,
                            num_hidden_layers=self.model_config.num_hidden_layers,
                            split_index=self.split_prefill_batch.split_index,
                        )
                        if aligned_fc is not None:
                            forward_count = aligned_fc
                            logger.info(
                                "pdmux prefill align: attn_dp_rank=%s tp_rank=%s stream_idx=%s "
                                "forward_count=%s base=%s decode_run_pred_ms=%s g=%s L=%s B=%s F=%s G=%s",
                                getattr(self, "attn_dp_rank", None),
                                getattr(self, "tp_rank", None),
                                stream_idx,
                                forward_count,
                                base_forward_count,
                                decode_run_pred,
                                g,
                                L if decode_run_pred is not None else None,
                                B if decode_run_pred is not None else None,
                                F,
                                G,
                            )
                        elif getattr(self.server_args, "enable_pdmux_diag_log", False):
                            logger.info(
                                "pdmux prefill align skipped: attn_dp_rank=%s tp_rank=%s "
                                "decode_run_pred=%s (need decode batch + L,B for alignment)",
                                getattr(self, "attn_dp_rank", None),
                                getattr(self, "tp_rank", None),
                                decode_run_pred,
                            )

                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    pn = _pdmux_special_dp_nvtx_prefill_pred_chunks(
                        getattr(self, "_pdmux_time_predictor", None),
                        stream_idx,
                        self.split_prefill_batch,
                        forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    prefill_launch_nvtx_extra = ""
                    if pn is not None:
                        prefill_launch_nvtx_extra = (
                            f" | pred_pl_chunk_ms={pn['pl_chunk_ms']:.4f}"
                            f" pred_pl_full_ms={pn['pl_full_ms']:.4f}"
                            f" layer_frac={pn['layer_frac']:.4f}"
                            f" fc={forward_count} nh={self.model_config.num_hidden_layers}"
                        )
                    prefill_gpu_nvtx_extra = ""
                    if pn is not None:
                        # Whole-bar estimate: this range spans all split chunks until prefill finishes.
                        prefill_gpu_nvtx_extra = (
                            f" | pred_pr_full_ms={pn['pr_full_ms']:.4f} g={stream_idx}"
                        )
                    if prefill_gpu_handle is None:
                        prefill_gpu_handle = nvtx.range_start(
                            "run prefill batch"
                            + f": {prefill_whole_batch_count} "
                            + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                            + prefill_gpu_nvtx_extra
                        )
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start(
                        "launch prefill batch"
                        + f": {prefill_forward_count} "
                        + _pdmux_nvtx_len_list_str(self.split_prefill_batch)
                        + prefill_launch_nvtx_extra
                    )
                    prefill_forward_count += 1
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    nvtx.range_end(prefill_launch_handle)
                    prefill_launch_handle = None
                    # logger.info(f"after run_prefill_batch, split_prefill_batch: {self.split_prefill_batch.batch_size()}")
                    # logger.info(f"after run_prefill_batch, split_prefill_batch.input_ids: {self.split_prefill_batch.input_ids.shape}")
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False
                
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    nvtx.range_end(decode_gpu_handle)
                    decode_gpu_handle = None
                    self.process_batch_result(self.decode_or_idle_batch, decode_result)
                
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        nvtx.range_end(prefill_gpu_handle)
                        prefill_gpu_handle = None
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        # log prefill stats late
                        self.log_prefill_stats_late(self.split_prefill_batch)
                        # TODO(lbz): for special dp attention, here, we need to convert the split_prefill_batch to the running_batch
                        if self.enable_special_dp_attention:
                            keep_indices = self.split_prefill_batch.dp_local_req_indices
                            if keep_indices is None:
                                logger.warning(
                                    "dp_local_req_indices is None; ensure prepare_for_extend "
                                    "sets it when server enable_special_dp_attention is True"
                                )
                                keep_indices = []
                            drop_indices = [
                                i
                                for i in range(self.split_prefill_batch.batch_size())
                                if i not in keep_indices
                            ]
                            # logger.info(
                            #     f"special_dp_attention: keep_indices={keep_indices}, "
                            #     f"drop_indices={drop_indices}, batch_size={self.split_prefill_batch.batch_size()}"
                            # )
                            # when we enable_save_kv_cache_for_dp, we do not need to release the kv cache, 
                            #  because we didn't alloc or save cache for these reqs
                            if not self.enable_save_kv_cache_for_dp:
                                for i in drop_indices:
                                    req = self.split_prefill_batch.reqs[i]
                                    # logger.info(
                                    #     f"releasing dropped req i={i} rid={req.rid} "
                                    #     f"req_pool_idx={req.req_pool_idx}"
                                    # )
                                    release_kv_cache(
                                        req,
                                        self.split_prefill_batch.tree_cache,
                                        is_insert=False,
                                    )
                            # TODO(lbz): bugfix, after filter batch, bs=0 but output_ids is not None, need to fix it
                            self.split_prefill_batch.filter_batch(
                                keep_indices=keep_indices,
                                req_pool_indices_is_dp_local=True,
                            )
                            # logger.info(
                            #     f"keep indices: {keep_indices}, "
                            #     f"after filter split_prefill_batch: {self.split_prefill_batch.batch_size()}"
                            #     f"split_prefill_batch.input_ids: {self.split_prefill_batch.input_ids.shape if self.split_prefill_batch.input_ids is not None else None}"
                            #     f"split_prefill_batch.output_ids: {self.split_prefill_batch.output_ids.shape if self.split_prefill_batch.output_ids is not None else None}"
                            # )

                        if self.split_prefill_batch is not None and self.split_prefill_batch.batch_size() != 0:
                            if self.running_batch and not self.running_batch.is_empty():
                                self.running_batch.merge_batch(self.split_prefill_batch)
                            else:
                                self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
                    nvtx.range_pop()
