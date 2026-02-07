Kernel Evaluation (KernelGYM-style)
===================================

This guide shows how to integrate a KernelGYM-style GPU kernel evaluator into RLinf.
It focuses on subprocess isolation, kernel correctness checks, and performance-based
reward shaping for kernel generation tasks.

Core Components
---------------

- ``KernelEvalWorker``: Runs kernel compile/run in subprocesses to isolate CUDA errors.
- ``KernelReward``: Converts evaluation results into scalar rewards.
- ``KernelRewardWorker``: Decodes generated code, evaluates kernels, and attaches results.
- ``KernelRunner`` (optional): Helper runner when you want a dedicated kernel-eval channel.

Configuration Snippet
---------------------

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

Usage Notes
-----------

1. In your entrypoint, use ``get_reward_worker(cfg)`` to select ``KernelRewardWorker``.
2. ``KernelRewardWorker`` uses an in-process evaluator by default (``KernelEvalClient``)
   that still executes kernels in subprocesses for isolation.
3. (Optional) Launch ``KernelEvalWorker`` as a separate group if you want a dedicated
   evaluation channel and custom orchestration.
3. Use ``KernelRunner`` or the standard reasoning runner depending on your training flow.

.. note::
   The compile, correctness, and performance hooks in
   ``rlinf/workers/kernel/kernel_eval_worker.py`` are stubs. Implement backend-specific
   logic before running real GPU evaluations.
