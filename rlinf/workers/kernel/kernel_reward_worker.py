# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from typing import Optional, Union

from omegaconf import DictConfig
import torch

from rlinf.algorithms.rewards.kernel import KernelReward
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.kernel.kernel_eval_worker import KernelEvalClient, KernelEvalWorker


class KernelRewardWorker(Worker):
    """Reward worker that evaluates kernels and computes rewards."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        kernel_eval_worker: Optional[Union[KernelEvalWorker, KernelEvalClient]] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.component_placement = placement
        self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )

        if kernel_eval_worker is None:
            device_id = self.cfg.reward.get("device_id", None)
            if device_id is None:
                device_id = getattr(self, "_local_accelerator_rank", 0)
            device_id = int(device_id)
            pool_size = self.cfg.reward.get("eval_pool_size", 1)
            if pool_size is None:
                pool_size = 1
            max_tasks_per_worker = self.cfg.reward.get("eval_max_tasks_per_worker", 1)
            if max_tasks_per_worker is None:
                max_tasks_per_worker = 1
            pool_size = int(pool_size)
            max_tasks_per_worker = int(max_tasks_per_worker)
            self.kernel_eval_worker = KernelEvalClient(
                cfg,
                device_id=device_id,
                pool_size=pool_size,
                max_tasks_per_worker=max_tasks_per_worker,
            )
        else:
            self.kernel_eval_worker = kernel_eval_worker

    def init_worker(self):
        """Initialize the reward and kernel evaluation workers."""
        self.reward = KernelReward(self.cfg.reward)
        self.kernel_eval_worker.init_worker()

    def compute_rewards(self, input_channel: Channel, output_channel: Channel) -> None:
        """Compute rewards by evaluating kernels.

        Args:
            input_channel: Channel containing RolloutResult with generated kernel code.
            output_channel: Channel to send RolloutResult with computed rewards.
        """
        entry_point = self.cfg.reward.get("entry_point", "ModelNew")
        backend = self.cfg.reward.get("backend", "triton")
        run_correctness = self.cfg.reward.get("run_correctness", True)
        if run_correctness is None:
            run_correctness = True
        run_performance = self.cfg.reward.get("run_performance", True)
        if run_performance is None:
            run_performance = True
        num_perf_trials = self.cfg.reward.get("num_perf_trials", 100)
        if num_perf_trials is None:
            num_perf_trials = 100
        num_perf_trials = int(num_perf_trials)

        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence

            with self.worker_timer():
                if rollout_result.rewards is None:
                    texts = rollout_result.response_texts
                    if texts is None:
                        texts = self.tokenizer.batch_decode(
                            rollout_result.response_ids, skip_special_tokens=True
                        )

                    eval_results = []
                    for index, text in enumerate(texts):
                        kernel_code = self._extract_kernel_code(text)
                        reference_code = self._get_reference_code(
                            rollout_result, index
                        )
                        eval_result = self.kernel_eval_worker.evaluate_kernel(
                            kernel_code=kernel_code,
                            reference_code=reference_code,
                            entry_point=entry_point,
                            backend=backend,
                            run_correctness=run_correctness,
                            run_performance=run_performance,
                            num_perf_trials=num_perf_trials,
                        )
                        eval_results.append(eval_result)

                    rewards = self.reward.get_reward_from_texts(texts, eval_results)
                    rollout_result.rewards = torch.as_tensor(
                        rewards, dtype=torch.float, device=torch.device("cpu")
                    ).view(-1)
                    self._attach_eval_results(rollout_result, eval_results)

            output_channel.put(rollout_result, async_op=True)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences, got {recv_batch_size}"
        )

    def _extract_kernel_code(self, text: str) -> str:
        """Extract kernel code from generated text."""
        match = re.search(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _get_reference_code(
        self, rollout_result: RolloutResult, index: int
    ) -> Optional[str]:
        """Get reference code from rollout result if available."""
        answers = rollout_result.answers
        if answers is None or not isinstance(answers, list):
            return None
        if index >= len(answers):
            return None
        reference = answers[index]
        if isinstance(reference, dict):
            return reference.get("reference_code")
        if isinstance(reference, list):
            return reference[0] if reference else None
        if isinstance(reference, str):
            return reference
        return None

    def _attach_eval_results(
        self, rollout_result: RolloutResult, eval_results: list[dict]
    ) -> None:
        """Attach evaluation results to the rollout metadata."""
        metadata = getattr(rollout_result, "metadata", None)
        if metadata is None:
            metadata = {}
            setattr(rollout_result, "metadata", metadata)
        metadata["kernel_eval_results"] = eval_results
