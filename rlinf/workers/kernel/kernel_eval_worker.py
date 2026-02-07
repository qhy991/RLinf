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

import ast
import multiprocessing
import os
from typing import Any, Optional

from omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement

KernelEvalResult = dict[str, Any]


def _evaluate_kernel_in_subprocess(
    task_data: dict[str, Any],
    device_id: int,
) -> KernelEvalResult:
    """Run kernel evaluation in a subprocess.

    Args:
        task_data: Serialized evaluation inputs.
        device_id: CUDA device index to expose.

    Returns:
        Kernel evaluation result dictionary.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return _evaluate_kernel_task(task_data)


def _evaluate_kernel_task(task_data: dict[str, Any]) -> KernelEvalResult:
    """Evaluate kernel code for compile/correctness/performance.

    This function is intentionally lightweight and should be extended with
    backend-specific logic (e.g., Triton/CUDA compilation and CUDA timing).
    """
    kernel_code = task_data.get("kernel_code", "")
    reference_code = task_data.get("reference_code")
    backend = task_data.get("backend", "triton")
    device = task_data.get("device", "cuda:0")
    run_correctness = task_data.get("run_correctness", True)
    run_performance = task_data.get("run_performance", True)
    num_perf_trials = task_data.get("num_perf_trials", 100)

    if not _compile_kernel(kernel_code, backend):
        return _error_result("Compilation failed.")

    errors: list[str] = []

    correctness: Optional[bool] = None
    if run_correctness:
        try:
            correctness = _check_correctness(
                kernel_code=kernel_code,
                reference_code=reference_code,
                backend=backend,
                device=device,
            )
        except NotImplementedError as exc:
            errors.append(str(exc))
            correctness = False

    kernel_runtime = -1.0
    if run_performance:
        try:
            kernel_runtime = _measure_performance(
                kernel_code=kernel_code,
                backend=backend,
                device=device,
                num_trials=num_perf_trials,
            )
        except NotImplementedError as exc:
            errors.append(str(exc))

    reference_runtime = task_data.get("reference_runtime", -1.0)
    speedup = 0.0
    if reference_runtime > 0 and kernel_runtime > 0:
        speedup = reference_runtime / kernel_runtime

    result: KernelEvalResult = {
        "compiled": True,
        "correctness": correctness,
        "kernel_runtime": kernel_runtime,
        "reference_runtime": reference_runtime,
        "speedup": speedup,
    }
    if errors:
        result["error_message"] = "; ".join(errors)
    return result


def _compile_kernel(kernel_code: str, backend: str) -> bool:
    """Compile kernel code.

    TODO(agent): Replace the syntax-only check with real backend compilation.
    """
    if not kernel_code:
        return False
    if backend not in {"triton", "cuda"}:
        raise ValueError(f"Unsupported backend: {backend}")
    try:
        ast.parse(kernel_code)
    except SyntaxError:
        return False
    return True


def _check_correctness(
    kernel_code: str,
    reference_code: Optional[str],
    backend: str,
    device: str,
) -> bool:
    """Check kernel correctness.

    TODO(agent): Implement correctness checks against a reference kernel.
    """
    raise NotImplementedError("Kernel correctness check is not implemented yet.")


def _measure_performance(
    kernel_code: str,
    backend: str,
    device: str,
    num_trials: int,
) -> float:
    """Measure kernel performance in milliseconds.

    TODO(agent): Implement CUDA timing with warmup and multiple trials.
    """
    raise NotImplementedError("Kernel performance measurement is not implemented yet.")


def _error_result(message: str) -> KernelEvalResult:
    """Create a standardized error result."""
    return {
        "compiled": False,
        "correctness": False,
        "kernel_runtime": -1.0,
        "reference_runtime": -1.0,
        "speedup": 0.0,
        "error_message": message,
    }


class SubprocessWorkerPool:
    """Subprocess worker pool for CUDA error isolation."""

    def __init__(
        self,
        device_id: int,
        pool_size: int,
        max_tasks_per_worker: int,
        start_method: str = "spawn",
    ):
        if pool_size <= 0:
            raise ValueError("pool_size must be greater than 0.")
        self.device_id = device_id
        self.pool_size = pool_size
        self.max_tasks_per_worker = max_tasks_per_worker
        self._ctx = multiprocessing.get_context(start_method)
        maxtasksperchild = max_tasks_per_worker if max_tasks_per_worker > 0 else None
        self._pool = self._ctx.Pool(
            processes=pool_size,
            maxtasksperchild=maxtasksperchild,
        )

    def execute_task(self, task_data: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """Execute a task in the pool with a timeout."""
        async_result = self._pool.apply_async(
            _evaluate_kernel_in_subprocess,
            (task_data, self.device_id),
        )
        try:
            result = async_result.get(timeout=timeout_s)
            return {"success": True, "result": result}
        except multiprocessing.TimeoutError:
            return {
                "success": False,
                "error_message": f"Kernel evaluation timed out after {timeout_s} seconds.",
            }
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error_message": str(exc)}

    def shutdown(self) -> None:
        """Shutdown the subprocess pool."""
        self._pool.close()
        self._pool.join()


class KernelEvalClient:
    """Lightweight kernel evaluator for in-process use."""

    def __init__(
        self,
        cfg: DictConfig,
        device_id: int = 0,
        pool_size: int = 1,
        max_tasks_per_worker: int = 1,
    ):
        self.cfg = cfg
        self.device_id = device_id
        self.pool_size = pool_size
        self.max_tasks_per_worker = max_tasks_per_worker
        self.worker_pool: Optional[SubprocessWorkerPool] = None
        timeout_s = self.cfg.reward.get("eval_timeout_s", 30.0)
        if timeout_s is None:
            timeout_s = 30.0
        self.eval_timeout_s = float(timeout_s)

    def init(self) -> None:
        """Initialize the subprocess pool."""
        if self.worker_pool is not None:
            return
        self.worker_pool = SubprocessWorkerPool(
            device_id=self.device_id,
            pool_size=self.pool_size,
            max_tasks_per_worker=self.max_tasks_per_worker,
        )

    def init_worker(self) -> None:
        """Compatibility wrapper for worker-style initialization."""
        self.init()

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
    ) -> KernelEvalResult:
        """Evaluate a kernel with subprocess isolation."""
        if device is None:
            device = f"cuda:{self.device_id}"
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
        try:
            if self.worker_pool is None:
                return _evaluate_kernel_task(task_data)
            response = self.worker_pool.execute_task(
                task_data=task_data, timeout_s=self.eval_timeout_s
            )
            if response.get("success"):
                return response["result"]
            return _error_result(response.get("error_message", "Unknown error."))
        except Exception as exc:  # noqa: BLE001
            return _error_result(str(exc))


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
        super().__init__()
        self.cfg = cfg
        self.component_placement = placement
        self._client = KernelEvalClient(
            cfg,
            device_id=device_id,
            pool_size=pool_size,
            max_tasks_per_worker=max_tasks_per_worker,
        )

    def init_worker(self):
        """Initialize the worker pool for subprocess isolation."""
        self._client.init()
        self.log_info(
            "Initialized kernel eval worker on device "
            f"{self._client.device_id} with pool_size={self._client.pool_size}."
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
    ) -> KernelEvalResult:
        """Evaluate a kernel in a subprocess.

        Args:
            kernel_code: Kernel code to evaluate.
            reference_code: Optional reference implementation.
            entry_point: Entry point class/function name.
            backend: Backend type ("triton" or "cuda").
            device: Target device string (e.g., "cuda:0").
            run_correctness: Whether to run correctness checks.
            run_performance: Whether to measure performance.
            num_perf_trials: Number of performance trials.

        Returns:
            Kernel evaluation result dictionary.
        """
        try:
            return self._client.evaluate_kernel(
                kernel_code=kernel_code,
                reference_code=reference_code,
                entry_point=entry_point,
                backend=backend,
                device=device,
                run_correctness=run_correctness,
                run_performance=run_performance,
                num_perf_trials=num_perf_trials,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_error(f"Kernel evaluation failed: {exc}")
            return _error_result(str(exc))

    def process_kernel_batch(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ) -> None:
        """Process kernel evaluation requests from a channel."""
        while True:
            task = input_channel.get()
            if task is None:
                break
            try:
                result = self.evaluate_kernel(**task)
            except Exception as exc:  # noqa: BLE001
                self.log_error(f"Error processing kernel batch: {exc}")
                result = _error_result(str(exc))
            output_channel.put(result, async_op=True)
