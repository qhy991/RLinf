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

# ruff: noqa: D103
import pytest

from rlinf.algorithms.rewards.kernel import KernelReward


def test_kernel_reward_components():
    cfg = {
        "compilation_weight": 1.0,
        "correctness_weight": 5.0,
        "performance_weight": 2.0,
        "speedup_weight": 3.0,
        "baseline_runtime": 2.0,
        "max_speedup": 10.0,
        "reward_scale": 1.0,
    }
    reward = KernelReward(cfg)
    results = [
        {
            "compiled": True,
            "correctness": True,
            "kernel_runtime": 1.0,
            "speedup": 4.0,
        }
    ]
    rewards = reward.get_reward(results)
    assert rewards == pytest.approx([7.6])


def test_kernel_reward_missing_metrics():
    cfg = {"reward_scale": 1.0}
    reward = KernelReward(cfg)
    results = [{"compiled": False, "correctness": False, "kernel_runtime": -1.0}]
    rewards = reward.get_reward(results)
    assert rewards == [0.0]
