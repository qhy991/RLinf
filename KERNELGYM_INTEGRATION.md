# 在 RLinf 中实现 KernelGYM 类似功能的实现指南

## 概述

本文档说明如何在 RLinf 框架中实现类似 KernelGYM 的 GPU 内核评估功能。KernelGYM 的核心功能包括：
1. GPU 内核编译、执行和性能测量
2. 分布式任务调度
3. 子进程隔离（防止 CUDA 错误影响主进程）
4. 正确性验证和性能测量
5. 工作流编排

## 架构设计

### 1. 核心组件

#### 1.1 KernelEvalWorker
负责在子进程中执行 GPU 内核评估，提供 CUDA 错误隔离。

#### 1.2 KernelReward
实现奖励计算逻辑，基于内核评估结果（编译成功、正确性、性能）计算奖励。

#### 1.3 KernelRunner
扩展现有的 AgentRunner 或创建新的 Runner，用于组织内核生成的 RL 训练流程。

## 实现步骤

### 步骤 1: 创建 KernelEvalWorker

创建文件：`rlinf/workers/kernel/kernel_eval_worker.py`

```python
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

import asyncio
import logging
import multiprocessing
import subprocess
import sys
from typing import Any, Dict, Optional
import torch

from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class KernelEvalWorker(Worker):
    """Worker for evaluating GPU kernels with subprocess isolation."""
    
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        device_id: int = 0,
        pool_size: int = 1,
        max_tasks_per_worker: int = 1,
    ):
        Worker.__init__(self)
        self.cfg = cfg
        self.component_placement = placement
        self.device_id = device_id
        self.pool_size = pool_size
        self.max_tasks_per_worker = max_tasks_per_worker
        self.worker_pool = None
        
    def init_worker(self):
        """Initialize the worker pool for subprocess isolation."""
        # 初始化子进程池
        # 这里可以使用类似 KernelGYM 的 SubprocessWorkerPool
        logger.info(
            f"Initializing kernel eval worker on device {self.device_id} "
            f"with pool_size={self.pool_size}"
        )
        
    def evaluate_kernel(
        self,
        kernel_code: str,
        reference_code: Optional[str] = None,
        entry_point: str = "ModelNew",
        backend: str = "triton",
        device: Optional[str] = None,
        run_correctness: bool = True,
        run_performance: bool = True,
        num_perf_trials: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate a kernel in a subprocess.
        
        Args:
            kernel_code: The kernel code to evaluate
            reference_code: Optional reference implementation
            entry_point: Entry point class name
            backend: Backend type (triton/cuda)
            device: Device string (e.g., "cuda:0")
            run_correctness: Whether to run correctness checks
            run_performance: Whether to measure performance
            num_perf_trials: Number of performance trials
            
        Returns:
            Dictionary containing evaluation results:
            - compiled: bool
            - correctness: bool
            - kernel_runtime: float (ms)
            - reference_runtime: float (ms) (if reference_code provided)
            - speedup: float (if reference_code provided)
            - error_message: Optional[str]
        """
        if device is None:
            device = f"cuda:{self.device_id}"
            
        # 在子进程中执行评估
        # 这里需要实现类似 KernelGYM 的 subprocess execution
        # 使用 multiprocessing 或 subprocess 来隔离 CUDA 错误
        
        try:
            # 创建评估任务
            task_data = {
                "kernel_code": kernel_code,
                "reference_code": reference_code,
                "entry_point": entry_point,
                "backend": backend,
                "device": device,
                "run_correctness": run_correctness,
                "run_performance": run_performance,
                "num_perf_trials": num_perf_trials,
            }
            
            # 在子进程中执行（使用进程池）
            result = self._execute_in_subprocess(task_data)
            return result
            
        except Exception as e:
            logger.error(f"Kernel evaluation failed: {e}")
            return {
                "compiled": False,
                "correctness": False,
                "kernel_runtime": -1.0,
                "reference_runtime": -1.0,
                "speedup": 0.0,
                "error_message": str(e),
            }
    
    def _execute_in_subprocess(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute kernel evaluation in a subprocess for isolation."""
        # 使用 multiprocessing.Pool 或 subprocess 来执行
        # 这里需要实现实际的评估逻辑
        # 可以参考 KernelGYM 的 subprocess_pool.py
        
        # 临时实现：直接在当前进程执行（生产环境应使用子进程）
        # TODO: 实现真正的子进程隔离
        return self._evaluate_kernel_internal(task_data)
    
    def _evaluate_kernel_internal(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal kernel evaluation logic."""
        # 这里需要实现实际的编译、执行、性能测量逻辑
        # 可以参考 KernelGYM 的 toolkit/kernelbench/ 实现
        
        # 占位实现
        kernel_code = task_data["kernel_code"]
        backend = task_data["backend"]
        device = task_data["device"]
        
        try:
            # 1. 编译内核
            compiled = self._compile_kernel(kernel_code, backend)
            if not compiled:
                return {
                    "compiled": False,
                    "correctness": False,
                    "kernel_runtime": -1.0,
                    "error_message": "Compilation failed",
                }
            
            # 2. 正确性检查（如果启用）
            correctness = True
            if task_data.get("run_correctness", True):
                correctness = self._check_correctness(kernel_code, backend, device)
            
            # 3. 性能测量（如果启用）
            kernel_runtime = -1.0
            if task_data.get("run_performance", True):
                kernel_runtime = self._measure_performance(
                    kernel_code, backend, device, task_data.get("num_perf_trials", 100)
                )
            
            return {
                "compiled": True,
                "correctness": correctness,
                "kernel_runtime": kernel_runtime,
                "reference_runtime": -1.0,
                "speedup": 0.0,
            }
            
        except Exception as e:
            logger.error(f"Kernel evaluation error: {e}")
            return {
                "compiled": False,
                "correctness": False,
                "kernel_runtime": -1.0,
                "error_message": str(e),
            }
    
    def _compile_kernel(self, kernel_code: str, backend: str) -> bool:
        """Compile kernel code."""
        # TODO: 实现实际的编译逻辑
        # 对于 Triton: 使用 triton.jit
        # 对于 CUDA: 使用 PyTorch CUDA extensions
        return True
    
    def _check_correctness(
        self, kernel_code: str, backend: str, device: str
    ) -> bool:
        """Check kernel correctness."""
        # TODO: 实现正确性检查
        # 需要执行内核并与参考实现比较
        return True
    
    def _measure_performance(
        self, kernel_code: str, backend: str, device: str, num_trials: int
    ) -> float:
        """Measure kernel performance."""
        # TODO: 实现性能测量
        # 使用 CUDA events 进行精确计时
        # 包含预热和多次试验
        return 0.0
    
    def process_kernel_batch(
        self, input_channel: Channel, output_channel: Channel
    ):
        """Process a batch of kernel evaluation tasks.
        
        Args:
            input_channel: Channel containing kernel evaluation requests
            output_channel: Channel to send evaluation results
        """
        while True:
            try:
                # 从 channel 获取任务
                task = input_channel.get()
                if task is None:  # Sentinel value to stop
                    break
                
                # 执行评估
                result = self.evaluate_kernel(**task)
                
                # 发送结果
                output_channel.put(result, async_op=True)
                
            except Exception as e:
                logger.error(f"Error processing kernel batch: {e}")
                # 发送错误结果
                error_result = {
                    "compiled": False,
                    "correctness": False,
                    "kernel_runtime": -1.0,
                    "error_message": str(e),
                }
                output_channel.put(error_result, async_op=True)
```

### 步骤 2: 创建 KernelReward

创建文件：`rlinf/algorithms/rewards/kernel/__init__.py`

```python
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

from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


class KernelReward:
    """Reward function for kernel generation tasks."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        # 奖励权重
        self.compilation_weight = config.get("compilation_weight", 1.0)
        self.correctness_weight = config.get("correctness_weight", 5.0)
        self.performance_weight = config.get("performance_weight", 2.0)
        self.speedup_weight = config.get("speedup_weight", 3.0)
        
        # 性能奖励参数
        self.baseline_runtime = config.get("baseline_runtime", 1.0)  # ms
        self.max_speedup = config.get("max_speedup", 10.0)
        
        # 奖励缩放
        self.reward_scale = config.get("reward_scale", 1.0)
    
    def get_reward(
        self,
        evaluation_results: list[dict],
        reference_runtimes: Optional[list[float]] = None,
    ) -> list[float]:
        """Compute rewards based on kernel evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            reference_runtimes: Optional list of reference runtimes for speedup calculation
            
        Returns:
            List of reward values
        """
        rewards = []
        
        for i, result in enumerate(evaluation_results):
            reward = 0.0
            
            # 1. 编译奖励
            if result.get("compiled", False):
                reward += self.compilation_weight
            
            # 2. 正确性奖励
            if result.get("correctness", False):
                reward += self.correctness_weight
            
            # 3. 性能奖励
            kernel_runtime = result.get("kernel_runtime", -1.0)
            if kernel_runtime > 0:
                # 基于运行时间的奖励（越快越好）
                speedup = self.baseline_runtime / kernel_runtime
                speedup = min(speedup, self.max_speedup)
                reward += self.performance_weight * (speedup / self.max_speedup)
            
            # 4. 加速比奖励（如果有参考实现）
            speedup = result.get("speedup", 0.0)
            if speedup > 0:
                normalized_speedup = min(speedup / self.max_speedup, 1.0)
                reward += self.speedup_weight * normalized_speedup
            
            # 应用缩放
            reward *= self.reward_scale
            rewards.append(reward)
        
        return rewards
    
    def get_reward_from_texts(
        self,
        texts: list[str],
        evaluation_results: list[dict],
    ) -> list[float]:
        """Compute rewards from generated texts and evaluation results.
        
        This is the interface expected by RewardWorker.
        
        Args:
            texts: Generated kernel code texts
            evaluation_results: Evaluation results for each text
            
        Returns:
            List of reward values
        """
        return self.get_reward(evaluation_results)
```

### 步骤 3: 创建 KernelRewardWorker

创建文件：`rlinf/workers/kernel/kernel_reward_worker.py`

```python
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

import logging
from typing import Optional

import torch
from omegaconf import DictConfig

from rlinf.algorithms.rewards.kernel import KernelReward
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.kernel.kernel_eval_worker import KernelEvalWorker

logger = logging.getLogger(__name__)


class KernelRewardWorker(Worker):
    """Reward worker that evaluates kernels and computes rewards."""
    
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        kernel_eval_worker: Optional[KernelEvalWorker] = None,
    ):
        Worker.__init__(self)
        self.cfg = cfg
        self.component_placement = placement
        self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )
        
        # 初始化内核评估 worker
        if kernel_eval_worker is None:
            device_id = self.cfg.reward.get("device_id", 0)
            self.kernel_eval_worker = KernelEvalWorker(
                cfg, placement, device_id=device_id
            )
        else:
            self.kernel_eval_worker = kernel_eval_worker
    
    def init_worker(self):
        """Initialize the reward worker."""
        # 初始化奖励计算器
        self.reward = KernelReward(self.cfg.reward)
        
        # 初始化内核评估 worker
        self.kernel_eval_worker.init_worker()
    
    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards by evaluating kernels.
        
        Args:
            input_channel: Channel containing RolloutResult with generated kernel code
            output_channel: Channel to send RolloutResult with computed rewards
        """
        recv_batch_size = 0
        evaluation_results = []
        
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence
            
            with self.worker_timer():
                if rollout_result.rewards is None:
                    # 解码生成的代码
                    texts = rollout_result.response_texts
                    if texts is None:
                        texts = self.tokenizer.batch_decode(
                            rollout_result.response_ids, skip_special_tokens=True
                        )
                    
                    # 评估每个内核
                    eval_results = []
                    for text in texts:
                        # 从文本中提取内核代码
                        kernel_code = self._extract_kernel_code(text)
                        
                        # 获取参考代码（如果有）
                        reference_code = self._get_reference_code(rollout_result)
                        
                        # 评估内核
                        eval_result = self.kernel_eval_worker.evaluate_kernel(
                            kernel_code=kernel_code,
                            reference_code=reference_code,
                            entry_point=self.cfg.reward.get("entry_point", "ModelNew"),
                            backend=self.cfg.reward.get("backend", "triton"),
                            run_correctness=self.cfg.reward.get("run_correctness", True),
                            run_performance=self.cfg.reward.get("run_performance", True),
                            num_perf_trials=self.cfg.reward.get("num_perf_trials", 100),
                        )
                        eval_results.append(eval_result)
                    
                    # 计算奖励
                    rollout_result.rewards = self.reward.get_reward_from_texts(
                        texts, eval_results
                    )
                    
                    # 将评估结果存储在 metadata 中
                    rollout_result.metadata = rollout_result.metadata or {}
                    rollout_result.metadata["kernel_eval_results"] = eval_results
            
            output_channel.put(rollout_result, async_op=True)
        
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences, got {recv_batch_size}"
        )
    
    def _extract_kernel_code(self, text: str) -> str:
        """Extract kernel code from generated text.
        
        This might involve parsing markdown code blocks or other formats.
        """
        # TODO: 实现代码提取逻辑
        # 可能需要处理 markdown 代码块、Python 代码等
        return text
    
    def _get_reference_code(self, rollout_result: RolloutResult) -> Optional[str]:
        """Get reference code from rollout result if available."""
        # TODO: 从 rollout_result 中提取参考代码
        # 可能存储在 metadata 或 answers 中
        return None
```

### 步骤 4: 创建 KernelRunner

创建文件：`rlinf/runners/kernel_runner.py`

```python
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

import logging
from typing import Optional, Union

from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.actor.megatron_actor_worker import MegatronActor
from rlinf.workers.inference.megatron_inference_worker import MegatronInference
from rlinf.workers.kernel.kernel_reward_worker import KernelRewardWorker
from rlinf.workers.kernel.kernel_eval_worker import KernelEvalWorker

if typing.TYPE_CHECKING:
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
    from rlinf.workers.rollout.vllm.vllm_worker import VLLMWorker

logger = logging.getLogger(__name__)


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
        
        # 创建用于内核评估的 channels
        self.kernel_eval_input_channel = Channel.create("KernelEvalInput")
        self.kernel_eval_output_channel = Channel.create("KernelEvalOutput")
    
    def train_step(self):
        """Execute one training step."""
        # 1. Rollout: 生成内核代码
        rollout_result = self._rollout_step()
        
        # 2. 评估内核（如果需要独立评估）
        if self.kernel_eval is not None:
            # 将生成的内核代码发送到评估 worker
            for kernel_code in rollout_result.response_texts:
                self.kernel_eval_input_channel.put({
                    "kernel_code": kernel_code,
                    # ... 其他参数
                })
            
            # 获取评估结果
            eval_results = []
            for _ in range(len(rollout_result.response_texts)):
                result = self.kernel_eval_output_channel.get()
                eval_results.append(result)
        
        # 3. 计算奖励（在 RewardWorker 中已经集成了评估）
        # 4. 训练 actor
        
        # 调用父类方法继续训练流程
        return super().train_step()
```

### 步骤 5: 集成到现有系统

#### 5.1 更新奖励注册

在 `rlinf/algorithms/rewards/__init__.py` 中添加：

```python
from .kernel import KernelReward

REWARD_CLASSES = {
    # ... existing rewards
    "kernel": KernelReward,
}
```

#### 5.2 配置示例

创建配置文件：`configs/kernel_rl.yaml`

```yaml
# Kernel Generation RL Configuration

data:
  rollout_batch_size: 32
  max_prompt_length: 2048

reward:
  reward_type: "kernel"
  reward_scale: 1.0
  
  # Kernel reward weights
  compilation_weight: 1.0
  correctness_weight: 5.0
  performance_weight: 2.0
  speedup_weight: 3.0
  
  # Performance parameters
  baseline_runtime: 1.0  # ms
  max_speedup: 10.0
  
  # Kernel evaluation parameters
  device_id: 0
  backend: "triton"  # or "cuda"
  entry_point: "ModelNew"
  run_correctness: true
  run_performance: true
  num_perf_trials: 100
  
  # Tokenizer
  tokenizer:
    tokenizer_model: "your-model-name"

algorithm:
  name: "ppo"  # or "grpo", "dapo", etc.
  group_size: 1

# ... other configurations
```

## 关键实现细节

### 1. 子进程隔离

为了隔离 CUDA 错误，需要实现类似 KernelGYM 的 `SubprocessWorkerPool`：

```python
import multiprocessing
import subprocess
import sys

class SubprocessWorkerPool:
    """Pool of subprocess workers for CUDA error isolation."""
    
    def __init__(self, device_id: int, pool_size: int, max_tasks_per_worker: int):
        self.device_id = device_id
        self.pool_size = pool_size
        self.max_tasks_per_worker = max_tasks_per_worker
        self.pool = multiprocessing.Pool(pool_size)
    
    def execute_task(self, task_data: dict, timeout: float = 30.0) -> dict:
        """Execute task in subprocess."""
        try:
            result = self.pool.apply_async(
                _evaluate_kernel_in_subprocess,
                (task_data, self.device_id),
            ).get(timeout=timeout)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error_message": str(e)}
    
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the pool."""
        self.pool.close()
        self.pool.join(timeout=timeout)


def _evaluate_kernel_in_subprocess(task_data: dict, device_id: int) -> dict:
    """Function executed in subprocess."""
    # 设置 CUDA 设备
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    # 执行评估逻辑
    # ... 实现实际的评估代码
    pass
```

### 2. 内核编译和执行

需要实现实际的编译和执行逻辑，可以参考 KernelGYM 的 `backend/kernelbench/` 实现：

- **Triton 后端**: 使用 `triton.jit` 装饰器
- **CUDA 后端**: 使用 PyTorch CUDA extensions
- **性能测量**: 使用 CUDA events 进行精确计时

### 3. 正确性验证

需要实现数值比较逻辑：
- 执行内核和参考实现
- 比较输出结果（支持 rtol/atol）
- 检测欺骗内核（返回固定值）

## 使用示例

```python
from rlinf.runners.kernel_runner import KernelRunner
from rlinf.workers.kernel.kernel_reward_worker import KernelRewardWorker
from rlinf.workers.kernel.kernel_eval_worker import KernelEvalWorker
# ... 其他导入

# 创建 runner
runner = KernelRunner(
    cfg=cfg,
    placement=placement,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    rollout=rollout_worker,
    inference=inference_worker,
    actor=actor_worker,
    reward=kernel_reward_worker,
    kernel_eval=kernel_eval_worker,
)

# 训练
runner.train()
```

## 与 KernelGYM 的对比

| 特性 | KernelGYM | RLinf 实现 |
|------|-----------|------------|
| 分布式调度 | Redis + FastAPI | Ray |
| 任务队列 | Redis Queue | Ray Channels |
| Worker 管理 | 自定义 Worker Manager | Ray Worker Groups |
| 子进程隔离 | SubprocessWorkerPool | multiprocessing.Pool |
| 奖励计算 | 外部集成 | 内置 RewardWorker |
| RL 训练 | 外部框架 (VERL) | 内置 RL 算法 |

## 优势

1. **统一框架**: 所有组件都在 RLinf 框架内，无需外部服务
2. **更好的集成**: 与 RLinf 的 RL 算法无缝集成
3. **灵活扩展**: 可以轻松添加新的评估指标和奖励函数
4. **分布式支持**: 利用 Ray 的强大分布式能力

## 后续工作

1. 实现完整的子进程隔离机制
2. 实现实际的编译和执行逻辑
3. 添加更多后端支持（CUDA, OpenCL 等）
4. 优化性能测量精度
5. 添加性能分析功能
6. 支持批量评估优化

## 参考

- KernelGYM: https://github.com/hkust-nlp/KernelGYM
- RLinf 文档: https://rlinf.readthedocs.io/
- Ray 文档: https://docs.ray.io/
