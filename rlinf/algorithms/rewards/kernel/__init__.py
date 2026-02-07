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

from typing import Any, Optional

from omegaconf import DictConfig


class KernelReward:
    """Reward function for kernel generation tasks."""

    def __init__(self, config: DictConfig):
        """Initialize kernel reward configuration."""
        self.config = config
        self.compilation_weight = float(config.get("compilation_weight", 1.0))
        self.correctness_weight = float(config.get("correctness_weight", 5.0))
        self.performance_weight = float(config.get("performance_weight", 2.0))
        self.speedup_weight = float(config.get("speedup_weight", 3.0))

        self.baseline_runtime = float(config.get("baseline_runtime", 1.0))
        max_speedup = float(config.get("max_speedup", 10.0))
        if max_speedup <= 0:
            max_speedup = 1.0
        self.max_speedup = max_speedup
        self.reward_scale = float(config.get("reward_scale", 1.0))

    def get_reward(
        self,
        evaluation_results: list[dict[str, Any]],
        reference_runtimes: Optional[list[float]] = None,
    ) -> list[float]:
        """Compute rewards based on kernel evaluation results.

        Args:
            evaluation_results: List of evaluation result dictionaries.
            reference_runtimes: Optional list of reference runtimes for speedup.

        Returns:
            List of reward values.
        """
        rewards: list[float] = []

        for index, result in enumerate(evaluation_results):
            reward = 0.0

            if result.get("compiled", False):
                reward += self.compilation_weight

            if result.get("correctness") is True:
                reward += self.correctness_weight

            kernel_runtime = float(result.get("kernel_runtime", -1.0))
            if kernel_runtime > 0:
                speedup = self.baseline_runtime / kernel_runtime
                speedup = min(speedup, self.max_speedup)
                reward += self.performance_weight * (speedup / self.max_speedup)

            speedup = float(result.get("speedup", 0.0))
            if speedup <= 0 and reference_runtimes is not None:
                if index < len(reference_runtimes):
                    reference_runtime = reference_runtimes[index]
                    if reference_runtime > 0 and kernel_runtime > 0:
                        speedup = reference_runtime / kernel_runtime

            if speedup > 0:
                normalized_speedup = min(speedup / self.max_speedup, 1.0)
                reward += self.speedup_weight * normalized_speedup

            reward *= self.reward_scale
            rewards.append(reward)

        return rewards

    def get_reward_from_texts(
        self,
        texts: list[str],
        evaluation_results: list[dict[str, Any]],
    ) -> list[float]:
        """Compute rewards from generated texts and evaluation results.

        Args:
            texts: Generated kernel code texts.
            evaluation_results: Evaluation results for each text.

        Returns:
            List of reward values.
        """
        return self.get_reward(evaluation_results)
