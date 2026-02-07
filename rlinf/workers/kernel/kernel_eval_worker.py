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

import multiprocessing
import os
import sys
import tempfile
import uuid
from pathlib import Path
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
    """Evaluate kernel code using robust-kbench primitives."""
    task_dir = task_data.get("task_dir")
    if not task_dir:
        return _error_result("task_dir is required for kernelbench evaluation.")

    kernel_code = task_data.get("kernel_code", "")
    cuda_code_path = task_data.get("cuda_code_path")

    try:
        _ensure_robust_kbench_importable()
        from robust_kbench.primitives.evaluate import (
            correct_cuda_kernel,
            eval_cuda_kernel,
            eval_torch_runtime,
            prof_cuda_kernel,
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Failed to import robust-kbench: {exc}")

    def _value(key: str, default: Any) -> Any:
        value = task_data.get(key, default)
        return default if value is None else value

    try:
        cuda_code_path = _prepare_cuda_code_path(
            task_data=task_data, task_dir=task_dir, cuda_code_path=cuda_code_path
        )
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"Failed to prepare CUDA code: {exc}")

    op_atol = float(_value("op_atol", 1e-3))
    op_rtol = float(_value("op_rtol", 1e-3))
    warmup_time = int(_value("warmup_time", 25))
    repetition_time = int(_value("rep_time", 100))
    eval_type = _value("eval_type", "kernelbench")
    multi_init_settings = bool(_value("multi_init_settings", False))
    multi_input_settings = bool(_value("multi_input_settings", False))
    timeout = int(_value("timeout", 600))
    num_correct_trials = int(_value("num_correct_trials", 5))
    backward = bool(_value("backward", False))
    correctness_first = bool(_value("correctness_first", False))
    skip_torch_eval = bool(_value("skip_torch_eval", False))
    run_correctness = bool(_value("run_correctness", True))
    run_performance = bool(_value("run_performance", True))
    enable_profile = bool(_value("enable_profile", False))
    reference_type = _value("reference_type", "torch_native")

    isolate_execution = bool(_value("isolate_execution", True))
    if isolate_execution:
        ext_dir = os.path.join(
            tempfile.gettempdir(), f"torch_extensions_{uuid.uuid4().hex[:8]}"
        )
    else:
        ext_dir = os.path.expanduser("~/.cache/torch_extensions/py311_cu124")

    errors: list[str] = []
    torch_results = None
    torch_compile_results = None
    correct_results = None
    cuda_results = None

    def _run_torch_eval():
        return eval_torch_runtime(
            task_dir=task_dir,
            multi_init_settings=multi_init_settings,
            multi_input_settings=multi_input_settings,
            warmup_time=warmup_time,
            repetition_time=repetition_time,
            eval_type=eval_type,
            timeout=timeout,
            gpu_id=0,
            ext_dir=ext_dir,
            forward=not backward,
            debug=False,
        )

    if not skip_torch_eval and not correctness_first:
        try:
            torch_results, torch_compile_results = _run_torch_eval()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Torch eval failed: {exc}")

    correctness: Optional[bool] = None
    if run_correctness:
        try:
            correct_results = correct_cuda_kernel(
                task_dir=task_dir,
                cuda_code_path=cuda_code_path,
                op_atol=op_atol,
                op_rtol=op_rtol,
                multi_init_settings=multi_init_settings,
                multi_input_settings=multi_input_settings,
                gpu_id=0,
                ext_dir=ext_dir,
                forward=not backward,
                timeout=timeout,
                num_correct_trials=num_correct_trials,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Correctness check failed: {exc}")
            correct_results = None

        if correct_results and "summary" in correct_results:
            correctness = bool(correct_results["summary"].get("correct", False))
        else:
            correctness = False

    if not skip_torch_eval and correctness_first:
        try:
            torch_results, torch_compile_results = _run_torch_eval()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Torch eval failed: {exc}")

    if run_performance and (not run_correctness or correctness):
        try:
            cuda_results = eval_cuda_kernel(
                task_dir=task_dir,
                cuda_code_path=cuda_code_path,
                warmup_time=warmup_time,
                repetition_time=repetition_time,
                eval_type=eval_type,
                multi_init_settings=multi_init_settings,
                multi_input_settings=multi_input_settings,
                gpu_id=0,
                ext_dir=ext_dir,
                timeout=timeout,
                forward=not backward,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Performance eval failed: {exc}")

    if enable_profile and (not run_correctness or correctness):
        try:
            prof_cuda_kernel(
                cuda_code_path=cuda_code_path,
                task_dir=task_dir,
                gpu_id=0,
                ext_dir=ext_dir,
                torch_prof=True,
                ncu_prof=True,
                clang_tidy=True,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Profiling failed: {exc}")

    kernel_runtime = -1.0
    if cuda_results and "summary" in cuda_results:
        kernel_runtime = float(cuda_results["summary"].get("avg_mean_time", -1.0))
        if kernel_runtime > 0:
            kernel_runtime *= 1000.0

    reference_runtime = -1.0
    if reference_type == "torch_compile":
        ref = torch_compile_results
    else:
        ref = torch_results
    if ref and "summary" in ref:
        reference_runtime = float(ref["summary"].get("avg_mean_time", -1.0))
        if reference_runtime > 0:
            reference_runtime *= 1000.0

    speedup = 0.0
    if reference_runtime > 0 and kernel_runtime > 0:
        speedup = reference_runtime / kernel_runtime

    compiled = False
    if run_correctness:
        compiled = correct_results is not None
    elif run_performance:
        compiled = cuda_results is not None
    else:
        compiled = bool(kernel_code or cuda_code_path)

    result: KernelEvalResult = {
        "compiled": compiled,
        "correctness": correctness,
        "kernel_runtime": kernel_runtime,
        "reference_runtime": reference_runtime,
        "speedup": speedup,
        "cuda_code_path": cuda_code_path,
    }
    if errors:
        result["error_message"] = "; ".join(errors)
    return result


def _ensure_robust_kbench_importable() -> None:
    """Ensure robust-kbench is importable without manual install."""
    try:
        import robust_kbench  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "robust-kbench"
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return
    raise ModuleNotFoundError(
        "robust_kbench is not importable and robust-kbench directory was not found."
    )


def _prepare_cuda_code_path(
    task_data: dict[str, Any],
    task_dir: str,
    cuda_code_path: Optional[str],
) -> str:
    kernel_code = task_data.get("kernel_code", "")
    overwrite_cuda = bool(task_data.get("overwrite_cuda", True))

    if cuda_code_path:
        if kernel_code and (overwrite_cuda or not os.path.exists(cuda_code_path)):
            _write_cuda_code(cuda_code_path, kernel_code)
        if not os.path.exists(cuda_code_path):
            raise FileNotFoundError(f"cuda_code_path not found: {cuda_code_path}")
        return cuda_code_path

    if not kernel_code:
        raise ValueError("kernel_code or cuda_code_path must be provided.")

    output_root = task_data.get(
        "kernel_output_dir",
        os.path.join(tempfile.gettempdir(), "rlinf_kernel_eval"),
    )
    round_num = task_data.get("round")
    branch_num = task_data.get("branch")
    iter_id = task_data.get("iter")
    if iter_id is None:
        iter_id = task_data.get("iteration")
    if iter_id is None:
        iter_id = task_data.get("iter_id")

    if round_num is not None and branch_num is not None:
        if iter_id is None:
            iter_id = uuid.uuid4().hex[:8]
        base_dir = os.path.join(
            output_root,
            f"round{round_num}",
            f"branch{branch_num}",
            f"iter_{iter_id}",
        )
    else:
        run_id = task_data.get("run_id") or uuid.uuid4().hex[:8]
        task_name = task_data.get("task_name") or os.path.basename(task_dir)
        base_dir = os.path.join(output_root, task_name, run_id)

    os.makedirs(base_dir, exist_ok=True)
    filename = task_data.get("cuda_filename") or "kernel.cu"
    cuda_code_path = os.path.join(base_dir, filename)
    _write_cuda_code(cuda_code_path, kernel_code)
    return cuda_code_path


def _write_cuda_code(cuda_code_path: str, kernel_code: str) -> None:
    os.makedirs(os.path.dirname(cuda_code_path), exist_ok=True)
    with open(cuda_code_path, "w", encoding="utf-8") as f:
        if kernel_code and not kernel_code.endswith("\n"):
            kernel_code += "\n"
        f.write(kernel_code)


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
        task_dir: Optional[str] = None,
        cuda_code_path: Optional[str] = None,
        op_atol: float = 1e-3,
        op_rtol: float = 1e-3,
        warmup_time: int = 25,
        rep_time: int = 100,
        eval_type: str = "kernelbench",
        multi_init_settings: bool = False,
        multi_input_settings: bool = False,
        timeout: int = 600,
        num_correct_trials: int = 5,
        correctness_first: bool = False,
        skip_torch_eval: bool = False,
        reference_type: str = "torch_native",
        isolate_execution: bool = True,
        backward: bool = False,
        enable_profile: bool = False,
        kernel_output_dir: Optional[str] = None,
        cuda_filename: Optional[str] = None,
        round: Optional[int] = None,
        branch: Optional[int] = None,
        iter: Optional[int] = None,
        run_id: Optional[str] = None,
        overwrite_cuda: bool = True,
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
            "task_dir": task_dir,
            "cuda_code_path": cuda_code_path,
            "op_atol": op_atol,
            "op_rtol": op_rtol,
            "warmup_time": warmup_time,
            "rep_time": rep_time,
            "eval_type": eval_type,
            "multi_init_settings": multi_init_settings,
            "multi_input_settings": multi_input_settings,
            "timeout": timeout,
            "num_correct_trials": num_correct_trials,
            "correctness_first": correctness_first,
            "skip_torch_eval": skip_torch_eval,
            "reference_type": reference_type,
            "isolate_execution": isolate_execution,
            "backward": backward,
            "enable_profile": enable_profile,
            "kernel_output_dir": kernel_output_dir,
            "cuda_filename": cuda_filename,
            "round": round,
            "branch": branch,
            "iter": iter,
            "run_id": run_id,
            "overwrite_cuda": overwrite_cuda,
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
        task_dir: Optional[str] = None,
        cuda_code_path: Optional[str] = None,
        op_atol: float = 1e-3,
        op_rtol: float = 1e-3,
        warmup_time: int = 25,
        rep_time: int = 100,
        eval_type: str = "kernelbench",
        multi_init_settings: bool = False,
        multi_input_settings: bool = False,
        timeout: int = 600,
        num_correct_trials: int = 5,
        correctness_first: bool = False,
        skip_torch_eval: bool = False,
        reference_type: str = "torch_native",
        isolate_execution: bool = True,
        backward: bool = False,
        enable_profile: bool = False,
        kernel_output_dir: Optional[str] = None,
        cuda_filename: Optional[str] = None,
        round: Optional[int] = None,
        branch: Optional[int] = None,
        iter: Optional[int] = None,
        run_id: Optional[str] = None,
        overwrite_cuda: bool = True,
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
                task_dir=task_dir,
                cuda_code_path=cuda_code_path,
                op_atol=op_atol,
                op_rtol=op_rtol,
                warmup_time=warmup_time,
                rep_time=rep_time,
                eval_type=eval_type,
                multi_init_settings=multi_init_settings,
                multi_input_settings=multi_input_settings,
                timeout=timeout,
                num_correct_trials=num_correct_trials,
                correctness_first=correctness_first,
                skip_torch_eval=skip_torch_eval,
                reference_type=reference_type,
                isolate_execution=isolate_execution,
                backward=backward,
                enable_profile=enable_profile,
                kernel_output_dir=kernel_output_dir,
                cuda_filename=cuda_filename,
                round=round,
                branch=branch,
                iter=iter,
                run_id=run_id,
                overwrite_cuda=overwrite_cuda,
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
