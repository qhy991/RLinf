内核评估（KernelGYM 风格）
===========================

本指南说明如何在 RLinf 中集成 KernelGYM 风格的 GPU 内核评估流程，
重点覆盖子进程隔离、正确性验证与性能驱动的奖励设计。

核心组件
--------

- ``KernelEvalWorker``：在子进程中编译与执行内核，隔离 CUDA 错误。
- ``KernelReward``：将评估结果转换为标量奖励。
- ``KernelRewardWorker``：解码生成代码、执行评估并附加结果。
- ``KernelRunner``（可选）：当你需要独立评估通道时使用。

配置示例
--------

.. code-block:: yaml

   reward:
     reward_type: kernel
     reward_scale: 1.0

     compilation_weight: 1.0
     correctness_weight: 5.0
     performance_weight: 2.0
     speedup_weight: 3.0

     baseline_runtime: 1.0
     max_speedup: 10.0

     device_id: 0
     backend: triton
     entry_point: ModelNew
     run_correctness: true
     run_performance: true
     num_perf_trials: 100

     eval_pool_size: 1
     eval_max_tasks_per_worker: 1
     eval_timeout_s: 30

     tokenizer:
       tokenizer_model: your-model-name

使用说明
--------

1. 在入口脚本中使用 ``get_reward_worker(cfg)`` 选择 ``KernelRewardWorker``。
2. ``KernelRewardWorker`` 需要每个样本提供 ``task_dir``。你可以放在 ``answers``（字典）
   或单独放在 ``meta`` 字段并在数据配置里启用 ``data.meta_key``。示例::

     answers = [{"task_dir": "/path/to/robust-kbench/tasks/kernelbench/level_2/task_7"}]
     meta = {"task_dir": "/path/to/robust-kbench/tasks/kernelbench/level_2/task_7"}
     data.meta_key: meta

3. ``KernelRewardWorker`` 默认使用进程内评估器（``KernelEvalClient``），但仍会通过子进程隔离执行。
4. （可选）启动独立的 ``KernelEvalWorker`` 作为评估通道，以便自定义调度。
5. 根据训练流程选择 ``KernelRunner`` 或现有 reasoning runner。

.. note::
   评估逻辑已经改为调用 robust-kbench 的评估接口（``correct_cuda_kernel`` /
   ``eval_cuda_kernel`` / ``eval_torch_runtime``）。请确保 GPU 机器上能导入
   ``robust-kbench``（安装或与 ``RLinf`` 同级目录放置）。
