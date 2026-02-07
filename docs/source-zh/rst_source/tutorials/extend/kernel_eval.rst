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
2. ``KernelRewardWorker`` 默认使用进程内评估器（``KernelEvalClient``），但仍会通过子进程隔离执行。
3. （可选）启动独立的 ``KernelEvalWorker`` 作为评估通道，以便自定义调度。
4. 根据训练流程选择 ``KernelRunner`` 或现有 reasoning runner。

.. note::
   ``rlinf/workers/kernel/kernel_eval_worker.py`` 中的编译、正确性与性能测量
   目前为占位逻辑，请在真实 GPU 评估前完成后端实现。
