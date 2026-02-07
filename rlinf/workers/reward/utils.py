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

from omegaconf import DictConfig

from rlinf.scheduler import Worker
from rlinf.workers.kernel.kernel_reward_worker import KernelRewardWorker
from rlinf.workers.reward.reward_worker import RewardWorker


def get_reward_worker(cfg: DictConfig) -> type[Worker]:
    """Select the reward worker class based on configuration."""
    if cfg.reward.reward_type == "kernel":
        return KernelRewardWorker
    return RewardWorker
