"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

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

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:

    def init_pdmux(self: Scheduler):
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(self.server_args.pdmux_config_path)
        initialize_stream_groups(self.gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )

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

    # @torch.inference_mode()
    # def event_loop_pdmux(self: Scheduler):
    #     """A scheduler loop for pd multiplexing."""
    #     decode_done = False
    #     prefill_done = False
    #     wait_prefill_kernel_done = False
    #     adjust_stream_group = False
    #     stream_idx = get_current_stream_idx()
    #     stream_group = self.stream_groups[stream_idx]
    #     prefill_stream = stream_group[0]
    #     decode_stream = stream_group[1]
    #     torch.cuda.empty_cache()

    #     logger.debug("Starting event loop for pd multiplexing...")

    #     while True:
    #         with torch.cuda.stream(decode_stream):
    #             set_pdmux_status(False)
    #             recv_reqs = self.recv_requests()
    #             self.process_input_requests(recv_reqs)

    #         with torch.cuda.stream(prefill_stream):
    #             set_pdmux_status(True)
    #             sm_count = self.sm_counts[stream_idx][0]
    #             if not wait_prefill_kernel_done:
    #                 adjust_stream_group = (
    #                     self.update_split_prefill_batch(sm_count) or adjust_stream_group
    #                 )

    #         with torch.cuda.stream(decode_stream):
    #             set_pdmux_status(False)
    #             self.running_batch = self.update_running_batch(self.running_batch)
    #             adjust_stream_group = adjust_stream_group or (
    #                 stream_idx > 0 and self.running_batch.is_empty()
    #             )
    #             if self.running_batch.is_empty() and self.split_prefill_batch is None:
    #                 self.check_memory()
    #                 self.check_tree_cache()
    #                 self.new_token_ratio = self.init_new_token_ratio
    #                 self.maybe_sleep_on_idle()

    #         if adjust_stream_group:
    #             prefill_stream.synchronize()
    #             decode_stream.synchronize()
    #             stream_idx, stream_group = self.adjust_stream_groups()
    #             prefill_stream = stream_group[0]
    #             decode_stream = stream_group[1]
    #             adjust_stream_group = False
    #             logger.debug(
    #                 f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
    #             )

    #         with torch.cuda.stream(decode_stream):
    #             set_pdmux_status(False)
    #             # process decode batch
    #             if self.running_batch and not self.running_batch.is_empty():
    #                 decode_result = self.run_batch(self.running_batch)
    #                 decode_done = True
    #             else:
    #                 decode_done = False
    #         with torch.cuda.stream(prefill_stream):
    #             set_pdmux_status(True)
    #             if (
    #                 self.split_prefill_batch
    #                 and not self.split_prefill_batch.is_empty()
    #                 and not wait_prefill_kernel_done
    #             ):
    #                 prefill_done = True
    #                 forward_count = (
    #                     max(
    #                         1,
    #                         self.pdmux_config.split_forward_token_budget
    #                         // self.split_prefill_batch.extend_num_tokens,
    #                     )
    #                     if self.split_prefill_batch.extend_num_tokens > 0
    #                     else self.model_config.num_hidden_layers
    #                 )
    #                 next_split_index = min(
    #                     self.split_prefill_batch.split_index + forward_count,
    #                     self.model_config.num_hidden_layers,
    #                 )
    #                 forward_count = (
    #                     next_split_index - self.split_prefill_batch.split_index
    #                 )

    #                 self.split_prefill_batch.split_forward_count = forward_count
    #                 prefill_result = self.run_batch(self.split_prefill_batch)
    #                 if next_split_index == self.model_config.num_hidden_layers:
    #                     self.split_prefill_batch.split_prefill_finished = True
    #                     prefill_exe_done = prefill_stream.record_event()
    #                 self.split_prefill_batch.split_index = next_split_index

    #             elif wait_prefill_kernel_done:
    #                 prefill_done = True
    #             else:
    #                 prefill_done = False

    #         with torch.cuda.stream(decode_stream):
    #             set_pdmux_status(False)
    #             decode_stream.synchronize()
    #             if decode_done:
    #                 self.process_batch_result(self.running_batch, decode_result)

    #         with torch.cuda.stream(prefill_stream):
    #             set_pdmux_status(True)
    #             if prefill_done and self.split_prefill_batch.split_prefill_finished:
    #                 wait_prefill_kernel_done = True
    #                 prefill_exe_done_flag = prefill_exe_done.query()
    #                 flags = (
    #                     torch.ones(1, device="cpu", dtype=torch.int32)
    #                     if prefill_exe_done_flag
    #                     else torch.zeros(1, device="cpu", dtype=torch.int32)
    #                 )

    #                 self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
    #                 if flags.item() == self.tp_size:
    #                     self.process_batch_result(
    #                         self.split_prefill_batch, prefill_result
    #                     )
    #                     if self.running_batch and not self.running_batch.is_empty():
    #                         self.running_batch.merge_batch(self.split_prefill_batch)
    #                     else:
    #                         self.running_batch = self.split_prefill_batch

    #                     self.split_prefill_batch = None
    #                     wait_prefill_kernel_done = False
    #                     adjust_stream_group = True

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

        logger.debug("Starting event loop for pd multiplexing...")

        while True:
            # ========== 阶段1: 接收请求 (decode_stream) ==========
            with torch.cuda.stream(decode_stream):
                logger.debug(
                    f"[Stage 1: Receive Requests] decode_stream | "
                    f"stream_idx={stream_idx}, "
                    f"running_batch_size={self.running_batch.batch_size() if self.running_batch else 0}, "
                    f"split_prefill_batch={'exists' if self.split_prefill_batch else 'None'}, "
                    f"wait_prefill_kernel_done={wait_prefill_kernel_done}"
                )
                set_pdmux_status(False)
                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                if recv_reqs:
                    logger.debug(f"[Stage 1] Received {len(recv_reqs)} new requests")

            # ========== 阶段2: 尝试获取新的 Prefill Batch (prefill_stream) ==========
            with torch.cuda.stream(prefill_stream):
                logger.debug(
                    f"[Stage 2: Try Get Prefill Batch] prefill_stream | "
                    f"stream_idx={stream_idx}, "
                    f"sm_count={self.sm_counts[stream_idx][0]}, "
                    f"split_prefill_batch={'exists' if self.split_prefill_batch else 'None'}, "
                    f"wait_prefill_kernel_done={wait_prefill_kernel_done}, "
                    f"adjust_stream_group={adjust_stream_group}"
                )
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                if not wait_prefill_kernel_done:
                    old_adjust = adjust_stream_group
                    adjust_stream_group = (
                        self.update_split_prefill_batch(sm_count) or adjust_stream_group
                    )
                    if adjust_stream_group != old_adjust:
                        logger.debug(
                            f"[Stage 2] update_split_prefill_batch returned True, "
                            f"new split_prefill_batch={'exists' if self.split_prefill_batch else 'None'}"
                        )

            # ========== 阶段3: 更新 Running Batch (decode_stream) ==========
            with torch.cuda.stream(decode_stream):
                logger.debug(
                    f"[Stage 3: Update Running Batch] decode_stream | "
                    f"stream_idx={stream_idx}, "
                    f"running_batch_size_before={self.running_batch.batch_size() if self.running_batch else 0}, "
                    f"split_prefill_batch={'exists' if self.split_prefill_batch else 'None'}, "
                    f"adjust_stream_group={adjust_stream_group}"
                )
                set_pdmux_status(False)
                self.running_batch = self.update_running_batch(self.running_batch)
                running_bs_after = self.running_batch.batch_size() if self.running_batch else 0
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and self.running_batch.is_empty()
                )
                logger.debug(
                    f"[Stage 3] running_batch_size_after={running_bs_after}, "
                    f"is_empty={self.running_batch.is_empty() if self.running_batch else True}, "
                    f"adjust_stream_group={adjust_stream_group}"
                )
                if self.running_batch.is_empty() and self.split_prefill_batch is None:
                    logger.debug("[Stage 3] Both batches empty, entering idle state")
                    self.check_memory()
                    self.check_tree_cache()
                    self.new_token_ratio = self.init_new_token_ratio
                    self.maybe_sleep_on_idle()

            # ========== 阶段4: 调整流组 (如果需要) ==========
            if adjust_stream_group:
                logger.debug(
                    f"[Stage 4: Adjust Stream Groups] | "
                    f"old_stream_idx={stream_idx}, "
                    f"running_batch_size={self.running_batch.batch_size() if self.running_batch else 0}, "
                    f"split_prefill_batch={'exists' if self.split_prefill_batch else 'None'}"
                )
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups()
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"[Stage 4] Adjusted to stream_idx={stream_idx}, "
                    f"prefill_sm={self.sm_counts[stream_idx][0]}, "
                    f"decode_sm={self.sm_counts[stream_idx][1]}"
                )

            # ========== 阶段5: 执行 Decode Batch (decode_stream) ==========
            with torch.cuda.stream(decode_stream):
                logger.debug(
                    f"[Stage 5: Execute Decode Batch] decode_stream | "
                    f"stream_idx={stream_idx}, "
                    f"running_batch={'exists' if self.running_batch else 'None'}, "
                    f"is_empty={self.running_batch.is_empty() if self.running_batch else True}, "
                    f"batch_size={self.running_batch.batch_size() if self.running_batch and not self.running_batch.is_empty() else 0}"
                )
                set_pdmux_status(False)
                # process decode batch
                if self.running_batch and not self.running_batch.is_empty():
                    decode_result = self.run_batch(self.running_batch)
                    decode_done = True
                    logger.debug(f"[Stage 5] Decode batch executed, batch_size={self.running_batch.batch_size()}")
                else:
                    decode_done = False
                    logger.debug("[Stage 5] No decode batch to execute")

            # ========== 阶段6: 执行 Split Prefill Batch (prefill_stream) ==========
            with torch.cuda.stream(prefill_stream):
                logger.debug(
                    f"[Stage 6: Execute Split Prefill Batch] prefill_stream | "
                    f"stream_idx={stream_idx}, "
                    f"split_prefill_batch={'exists' if self.split_prefill_batch else 'None'}, "
                    f"is_empty={self.split_prefill_batch.is_empty() if self.split_prefill_batch else True}, "
                    f"wait_prefill_kernel_done={wait_prefill_kernel_done}, "
                    f"split_index={self.split_prefill_batch.split_index if self.split_prefill_batch else 'N/A'}, "
                    f"split_prefill_finished={getattr(self.split_prefill_batch, 'split_prefill_finished', False) if self.split_prefill_batch else 'N/A'}"
                )
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    current_split_index = self.split_prefill_batch.split_index
                    extend_num_tokens = self.split_prefill_batch.extend_num_tokens
                    forward_count = (
                        max(
                            1,
                            self.pdmux_config.split_forward_token_budget
                            // extend_num_tokens,
                        )
                        if extend_num_tokens > 0
                        else self.model_config.num_hidden_layers
                    )
                    next_split_index = min(
                        current_split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - current_split_index
                    )

                    logger.debug(
                        f"[Stage 6] Executing split prefill: "
                        f"current_split_index={current_split_index}, "
                        f"forward_count={forward_count}, "
                        f"next_split_index={next_split_index}, "
                        f"extend_num_tokens={extend_num_tokens}, "
                        f"num_hidden_layers={self.model_config.num_hidden_layers}"
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                        logger.debug("[Stage 6] Split prefill finished all layers, recorded event")
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                    logger.debug("[Stage 6] Waiting for prefill kernel to complete")
                else:
                    prefill_done = False
                    logger.debug("[Stage 6] No prefill batch to execute")

            # ========== 阶段7: 处理 Decode 结果 (decode_stream) ==========
            with torch.cuda.stream(decode_stream):
                logger.debug(
                    f"[Stage 7: Process Decode Result] decode_stream | "
                    f"decode_done={decode_done}, "
                    f"running_batch_size={self.running_batch.batch_size() if self.running_batch else 0}"
                )
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    logger.debug(f"[Stage 7] Processing decode result for batch_size={self.running_batch.batch_size()}")
                    self.process_batch_result(self.running_batch, decode_result)
                else:
                    logger.debug("[Stage 7] No decode result to process")

            # ========== 阶段8: 处理 Prefill 结果并合并 (prefill_stream) ==========
            with torch.cuda.stream(prefill_stream):
                logger.debug(
                    f"[Stage 8: Process Prefill Result] prefill_stream | "
                    f"prefill_done={prefill_done}, "
                    f"split_prefill_finished={getattr(self.split_prefill_batch, 'split_prefill_finished', False) if self.split_prefill_batch else False}, "
                    f"wait_prefill_kernel_done={wait_prefill_kernel_done}, "
                    f"running_batch_size={self.running_batch.batch_size() if self.running_batch else 0}, "
                    f"split_prefill_batch_size={self.split_prefill_batch.batch_size() if self.split_prefill_batch else 0}"
                )
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    logger.debug(
                        f"[Stage 8] Checking TP sync: "
                        f"prefill_exe_done_flag={prefill_exe_done_flag}, "
                        f"tp_size={self.tp_size}"
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.tp_size:
                        logger.debug(
                            f"[Stage 8] All TP ranks finished, processing prefill result. "
                            f"split_prefill_batch_size={self.split_prefill_batch.batch_size()}, "
                            f"running_batch_size={self.running_batch.batch_size() if self.running_batch else 0}"
                        )
                        self.process_batch_result(
                            self.split_prefill_batch, prefill_result
                        )
                        if self.running_batch and not self.running_batch.is_empty():
                            logger.debug(
                                f"[Stage 8] Merging split_prefill_batch (size={self.split_prefill_batch.batch_size()}) "
                                f"into running_batch (size={self.running_batch.batch_size()})"
                            )
                            self.running_batch.merge_batch(self.split_prefill_batch)
                        else:
                            logger.debug(
                                f"[Stage 8] Setting running_batch to split_prefill_batch "
                                f"(size={self.split_prefill_batch.batch_size()})"
                            )
                            self.running_batch = self.split_prefill_batch

                        self.split_prefill_batch = None
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
                        logger.debug(
                            f"[Stage 8] Prefill completed and merged. "
                            f"New running_batch_size={self.running_batch.batch_size()}, "
                            f"adjust_stream_group={adjust_stream_group}"
                        )
                    else:
                        logger.debug(
                            f"[Stage 8] TP sync not complete: flags.item()={flags.item()}, "
                            f"expected={self.tp_size}"
                        )
                else:
                    logger.debug(
                        f"[Stage 8] Prefill not ready: "
                        f"prefill_done={prefill_done}, "
                        f"split_prefill_finished={getattr(self.split_prefill_batch, 'split_prefill_finished', False) if self.split_prefill_batch else False}"
                    )
