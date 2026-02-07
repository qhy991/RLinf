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

"""
示例脚本：准备 kernel 训练数据（answer 为 CUDA 代码，meta 中包含 task_dir）

使用方法（从已有数据补充 meta）：
    python prepare_kernel_data.py \
        --input_data /path/to/your/raw_data.jsonl \
        --output_data /path/to/output_data.jsonl \
        --robust_kbench_root /path/to/robust-kbench

使用方法（从 kernelbench 任务自动生成）：
    python prepare_kernel_data.py \
        --from_kernelbench_tasks \
        --output_data /path/to/output_data.jsonl \
        --robust_kbench_root /path/to/robust-kbench

可选：自定义 prompt 模板
    python prepare_kernel_data.py \
        --from_kernelbench_tasks \
        --output_data /path/to/output_data.jsonl \
        --robust_kbench_root /path/to/robust-kbench \
        --prompt_template_path /path/to/prompt_template.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_PROMPT_TEMPLATE = """你是 CUDA 专家。请根据下面的 PyTorch 参考实现生成等价的 CUDA 扩展代码（forward.cu）。

任务描述：
{operation_info}

PyTorch forward_fn:
{forward_fn}

输入张量名称：
{input_names}

配置（部分输入/初始化参数）：
{config_str}

要求：
1. 输出完整的 CUDA C++ 扩展代码（forward.cu），包含必要的头文件。
2. 实现 torch::Tensor forward(...)，参数与 forward_fn 的输入保持一致。
3. 使用 PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 导出 forward。
4. 只输出 CUDA 代码，不要输出解释文字。
"""


def prepare_kernel_data(
    input_data_path: Optional[str],
    output_data_path: str,
    robust_kbench_root: str,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    meta_key: str = "meta",
    from_kernelbench_tasks: bool = False,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    task_level_filter: Optional[str] = None,
    max_tasks: Optional[int] = None,
) -> None:
    """
    准备 kernel 训练数据，确保每条样本的 meta 中包含 task_dir。

    Args:
        input_data_path: 输入的原始数据文件路径（JSONL 格式）
        output_data_path: 输出的数据文件路径（JSONL 格式）
        robust_kbench_root: robust-kbench 的根目录路径
        prompt_key: 数据中 prompt 字段的键名
        answer_key: 数据中 answer 字段的键名
        meta_key: 数据中 meta 字段的键名
        from_kernelbench_tasks: 是否直接从 kernelbench 任务目录生成
        prompt_template: 自动生成 prompt 的模板
        task_level_filter: 只处理指定 level（如 "level_2"）
        max_tasks: 限制最多生成的任务数量
    """
    robust_kbench_root = Path(robust_kbench_root).expanduser().resolve()
    tasks_dir = robust_kbench_root / "tasks" / "kernelbench"

    if not tasks_dir.exists():
        raise ValueError(
            f"robust-kbench tasks directory not found: {tasks_dir}\n"
            f"Please ensure robust-kbench is installed or provide the correct path."
        )

    processed_data: list[dict[str, Any]] = []
    if from_kernelbench_tasks:
        task_dirs = _collect_kernelbench_tasks(
            tasks_dir, task_level_filter=task_level_filter
        )
        if max_tasks is not None:
            task_dirs = task_dirs[: max_tasks]
        for task_dir in task_dirs:
            prompt = _build_prompt_from_task(task_dir, prompt_template)
            processed_item = {
                prompt_key: prompt,
                answer_key: "",  # 模型训练时生成 CUDA 代码
                meta_key: {"task_dir": task_dir},
            }
            processed_data.append(processed_item)
    else:
        if not input_data_path:
            raise ValueError("input_data_path is required unless --from_kernelbench_tasks is set.")
        with open(input_data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {idx+1} due to JSON decode error: {e}")
                continue

            prompt = item.get(prompt_key, "")
            original_answer = item.get(answer_key, "")
            meta = dict(item.get(meta_key, {}) or {})

            task_dir = (
                meta.get("task_dir")
                or item.get("task_dir")
                or _resolve_task_dir(item, tasks_dir)
            )
            if not task_dir:
                print(
                    f"Warning: Line {idx+1} does not have task_dir. "
                    "Please ensure each sample has task_dir in meta."
                )
                continue

            if not os.path.exists(task_dir):
                print(
                    f"Warning: task_dir does not exist for line {idx+1}: {task_dir}\n"
                    "Skipping this sample."
                )
                continue

            meta["task_dir"] = task_dir
            for key in ("cuda_code_path", "round", "branch", "iter", "op_atol", "op_rtol",
                        "warmup_time", "rep_time", "timeout", "num_correct_trials",
                        "correctness_first", "skip_torch_eval", "isolate_execution",
                        "backward", "enable_profile"):
                if key in item:
                    meta[key] = item[key]

            answer_value = _normalize_answer(original_answer)
            processed_item = {
                prompt_key: prompt,
                answer_key: answer_value,
                meta_key: meta,
            }

            for key, value in item.items():
                if key not in [prompt_key, answer_key, meta_key]:
                    processed_item[key] = value

            processed_data.append(processed_item)

    # 写入处理后的数据
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    with open(output_data_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully processed {len(processed_data)} samples.")
    print(f"Output saved to: {output_data_path}")


def _resolve_task_dir(item: dict[str, Any], tasks_dir: Path) -> Optional[str]:
    task_id = item.get("task_id") or item.get("task_name")
    if task_id:
        level = item.get("level", "level_2")
        return str(tasks_dir / level / f"task_{task_id}")
    return item.get("task_dir")


def _normalize_answer(answer: Any) -> Any:
    if isinstance(answer, dict):
        for key in ("reference_code", "cuda_code", "code", "answer"):
            if key in answer and isinstance(answer[key], str):
                return answer[key]
    return answer


def _collect_kernelbench_tasks(
    tasks_dir: Path, task_level_filter: Optional[str] = None
) -> list[str]:
    task_dirs = []
    for level_dir in sorted(tasks_dir.iterdir()):
        if not level_dir.is_dir():
            continue
        if task_level_filter and level_dir.name != task_level_filter:
            continue
        for task_dir in sorted(level_dir.glob("task_*")):
            if (task_dir / "func_forward.py").exists():
                task_dirs.append(str(task_dir))
    return task_dirs


def _build_prompt_from_task(task_dir: str, template: str) -> str:
    try:
        from robust_kbench.kernel_task import KernelTask
    except Exception:
        KernelTask = None

    forward_fn = ""
    operation_info = ""
    input_names = ""
    config_str = ""

    if KernelTask is not None:
        task = KernelTask(task_dir)
        forward_fn = task.forward_fn_str or ""
        operation_info = task.operation_info or ""
        input_names = ", ".join(task.get_input_names() or [])
        if task.configs_str:
            config_str = "; ".join(task.configs_str[:3])
    else:
        func_path = Path(task_dir) / "func_forward.py"
        if func_path.exists():
            forward_fn = func_path.read_text(encoding="utf-8")

    if not operation_info:
        operation_info = "请参考 forward_fn 的实现细节。"
    if not input_names:
        input_names = "参见 forward_fn"
    if not config_str:
        config_path = Path(task_dir) / "config_forward.json"
        if config_path.exists():
            config_str = config_path.read_text(encoding="utf-8")
        else:
            config_str = "无"

    return template.format(
        operation_info=operation_info,
        forward_fn=forward_fn,
        input_names=input_names,
        config_str=config_str,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare kernel training data with task_dir in meta"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        help="Path to input data file (JSONL format)",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Path to output data file (JSONL format)",
    )
    parser.add_argument(
        "--robust_kbench_root",
        type=str,
        required=True,
        help="Root directory of robust-kbench",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="Key name for prompt in data (default: 'prompt')",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default="answer",
        help="Key name for answer in data (default: 'answer')",
    )
    parser.add_argument(
        "--meta_key",
        type=str,
        default="meta",
        help="Key name for meta in data (default: 'meta')",
    )
    parser.add_argument(
        "--from_kernelbench_tasks",
        action="store_true",
        help="Generate dataset directly from kernelbench tasks",
    )
    parser.add_argument(
        "--task_level",
        type=str,
        default=None,
        help="Only include tasks from a specific level (e.g., level_2)",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to include when auto-generating",
    )
    parser.add_argument(
        "--prompt_template_path",
        type=str,
        default=None,
        help="Optional path to a prompt template file",
    )

    args = parser.parse_args()

    prompt_template = DEFAULT_PROMPT_TEMPLATE
    if args.prompt_template_path:
        template_path = Path(args.prompt_template_path).expanduser()
        if not template_path.exists():
            raise FileNotFoundError(f"prompt_template_path not found: {template_path}")
        prompt_template = template_path.read_text(encoding="utf-8")

    prepare_kernel_data(
        input_data_path=args.input_data,
        output_data_path=args.output_data,
        robust_kbench_root=args.robust_kbench_root,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        meta_key=args.meta_key,
        from_kernelbench_tasks=args.from_kernelbench_tasks,
        prompt_template=prompt_template,
        task_level_filter=args.task_level,
        max_tasks=args.max_tasks,
    )


if __name__ == "__main__":
    main()
