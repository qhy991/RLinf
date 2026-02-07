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

import typing
from typing import Optional, Union

from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.kernel.kernel_eval_worker import KernelEvalWorker
from rlinf.workers.kernel.kernel_reward_worker import KernelRewardWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker


class KernelRunner(ReasoningRunner):
    """Runner for kernel generation RL training."""

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        train_dataset: Dataset,
        val_dataset: Dataset,
        rollout: Union["SGLangWorker", "VLLMWorker"],
        inference: Optional[MegatronInference],
        actor: MegatronActor,
        reward: Optional[KernelRewardWorker],
        kernel_eval: Optional[KernelEvalWorker] = None,
    ):
        super().__init__(
            cfg,
            placement,
            train_dataset,
            val_dataset,
            rollout,
            inference,
            actor,
            reward,
        )
        self.kernel_eval = kernel_eval
        if self.kernel_eval is not None:
            self.kernel_eval_input_channel = Channel.create("KernelEvalInput")
            self.kernel_eval_output_channel = Channel.create("KernelEvalOutput")

    def evaluate_kernels(self, kernel_texts: list[str]) -> list[dict]:
        """Evaluate kernels using a dedicated kernel eval worker if available."""
        if self.kernel_eval is None:
            return []

        for kernel_code in kernel_texts:
            self.kernel_eval_input_channel.put(
                {"kernel_code": kernel_code}, async_op=True
            )

        results = []
        for _ in kernel_texts:
            results.append(self.kernel_eval_output_channel.get())
        return results
