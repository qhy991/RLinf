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
        run_correctness = self._get_cfg_bool("run_correctness", True)
        run_performance = self._get_cfg_bool("run_performance", True)
        num_perf_trials = int(self._get_cfg_value("num_perf_trials", 100))
        op_atol = float(self._get_cfg_value("op_atol", 1e-3))
        op_rtol = float(self._get_cfg_value("op_rtol", 1e-3))
        warmup_time = int(self._get_cfg_value("warmup_time", 25))
        rep_time = int(self._get_cfg_value("rep_time", 100))
        eval_type = self._get_cfg_value("eval_type", "kernelbench")
        multi_init_settings = self._get_cfg_bool("multi_init_settings", False)
        multi_input_settings = self._get_cfg_bool("multi_input_settings", False)
        timeout = int(self._get_cfg_value("timeout", 600))
        num_correct_trials = int(self._get_cfg_value("num_correct_trials", 5))
        correctness_first = self._get_cfg_bool("correctness_first", False)
        skip_torch_eval = self._get_cfg_bool("skip_torch_eval", False)
        reference_type = self._get_cfg_value("reference_type", "torch_native")
        isolate_execution = self._get_cfg_bool("isolate_execution", True)
        backward = self._get_cfg_bool("backward", False)
        enable_profile = self._get_cfg_bool("enable_profile", False)
        kernel_output_dir = self.cfg.reward.get("kernel_output_dir")

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
                        task_meta = self._get_task_metadata(rollout_result, index)
                        kernel_code = self._extract_kernel_code(text)
                        reference_code = self._get_reference_code(rollout_result, index)
                        eval_result = self.kernel_eval_worker.evaluate_kernel(
                            kernel_code=kernel_code,
                            reference_code=reference_code,
                            entry_point=entry_point,
                            backend=backend,
                            run_correctness=run_correctness,
                            run_performance=run_performance,
                            num_perf_trials=num_perf_trials,
                            task_dir=task_meta.get("task_dir"),
                            cuda_code_path=task_meta.get("cuda_code_path"),
                            op_atol=float(task_meta.get("op_atol", op_atol)),
                            op_rtol=float(task_meta.get("op_rtol", op_rtol)),
                            warmup_time=int(task_meta.get("warmup_time", warmup_time)),
                            rep_time=int(task_meta.get("rep_time", rep_time)),
                            eval_type=task_meta.get("eval_type", eval_type),
                            multi_init_settings=bool(
                                task_meta.get("multi_init_settings", multi_init_settings)
                            ),
                            multi_input_settings=bool(
                                task_meta.get(
                                    "multi_input_settings", multi_input_settings
                                )
                            ),
                            timeout=int(task_meta.get("timeout", timeout)),
                            num_correct_trials=int(
                                task_meta.get("num_correct_trials", num_correct_trials)
                            ),
                            correctness_first=bool(
                                task_meta.get("correctness_first", correctness_first)
                            ),
                            skip_torch_eval=bool(
                                task_meta.get("skip_torch_eval", skip_torch_eval)
                            ),
                            reference_type=task_meta.get(
                                "reference_type", reference_type
                            ),
                            isolate_execution=bool(
                                task_meta.get("isolate_execution", isolate_execution)
                            ),
                            backward=bool(task_meta.get("backward", backward)),
                            enable_profile=bool(
                                task_meta.get("enable_profile", enable_profile)
                            ),
                            kernel_output_dir=task_meta.get(
                                "kernel_output_dir", kernel_output_dir
                            ),
                            cuda_filename=task_meta.get("cuda_filename"),
                            round=task_meta.get("round"),
                            branch=task_meta.get("branch"),
                            iter=task_meta.get("iter"),
                            run_id=task_meta.get("run_id"),
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

    def _get_cfg_value(self, key: str, default):
        value = self.cfg.reward.get(key, default)
        return default if value is None else value

    def _get_cfg_bool(self, key: str, default: bool) -> bool:
        return bool(self._get_cfg_value(key, default))

    def _get_task_metadata(self, rollout_result: RolloutResult, index: int) -> dict:
        answers = rollout_result.answers
        if isinstance(answers, list) and index < len(answers):
            if isinstance(answers[index], dict):
                return answers[index]
        return {}

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
