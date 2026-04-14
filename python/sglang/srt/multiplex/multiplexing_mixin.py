"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

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
from sglang.srt.mem_cache.common import release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


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
        self.log_stream_groups()

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

        # use the max decode bs to adjust the stream group
        max_decode_bs = max(self.decode_or_idle_batch.global_num_tokens) if self.decode_or_idle_batch is not None else 0
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
                    decode_gpu_handle = nvtx.range_start("run decode batch" + f": {decode_forward_count}")
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
                        prefill_gpu_handle = nvtx.range_start("run prefill batch" + f": {prefill_whole_batch_count}")
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start("launch prefill batch" + f": {prefill_forward_count}")
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
        arm_decode_need_wait_after_prefill = False

        def get_sg_msg():
            return f"stream_group: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"

        def overlap_log(message: str):
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
                            f"decode wait event before launch, queue_len={len(decode_result_queue)}"
                        )
                        decode_stream.wait_event(decode_run_done)
                    decode_result = self.run_batch(decode_batch)
                    decode_run_done = decode_stream.record_event()
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
                        if self.running_batch and not self.running_batch.is_empty():
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
                        decode_need_wait = False
                        decode_run_done = None
                        arm_decode_need_wait_after_prefill = True
                        overlap_log(
                            "prefill finished and merged: set decode_need_wait=False, arm_after_prefill=True"
                        )

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
                    # logger.info(f"after maybe_prepare_mlp_sync_batch_and_log_stats, decode_or_idle_batch: {self.decode_or_idle_batch.batch_size()}, global_num_tokens: {self.decode_or_idle_batch.global_num_tokens}")
                    pass
                else:
                    # logger.info(f"after maybe_prepare_mlp_sync_batch_and_log_stats, decode_or_idle_batch is None")
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
                    decode_gpu_handle = nvtx.range_start("run decode batch" + f": {decode_forward_count}")
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
                        prefill_gpu_handle = nvtx.range_start("run prefill batch" + f": {prefill_whole_batch_count}")
                        prefill_whole_batch_count += 1
                    prefill_launch_handle = nvtx.range_start("launch prefill batch" + f": {prefill_forward_count}")
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
