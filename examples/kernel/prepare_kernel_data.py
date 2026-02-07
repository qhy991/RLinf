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
示例脚本：准备包含 task_dir 的 kernel 训练数据

这个脚本展示了如何准备数据，确保每条样本的 answers 字段包含 task_dir。

使用方法：
    python prepare_kernel_data.py \
        --input_data /path/to/your/raw_data.jsonl \
        --output_data /path/to/output_data.jsonl \
        --robust_kbench_root /path/to/robust-kbench
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any


def prepare_kernel_data(
    input_data_path: str,
    output_data_path: str,
    robust_kbench_root: str,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
) -> None:
    """
    准备 kernel 训练数据，确保每条样本的 answers 包含 task_dir。

    Args:
        input_data_path: 输入的原始数据文件路径（JSONL 格式）
        output_data_path: 输出的数据文件路径（JSONL 格式）
        robust_kbench_root: robust-kbench 的根目录路径
        prompt_key: 数据中 prompt 字段的键名
        answer_key: 数据中 answer 字段的键名
    """
    robust_kbench_root = Path(robust_kbench_root).expanduser().resolve()
    tasks_dir = robust_kbench_root / "tasks" / "kernelbench"

    if not tasks_dir.exists():
        raise ValueError(
            f"robust-kbench tasks directory not found: {tasks_dir}\n"
            f"Please ensure robust-kbench is installed or provide the correct path."
        )

    # 读取原始数据
    with open(input_data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    processed_data = []
    for idx, line in enumerate(lines):
        try:
            item = json.loads(line.strip())
        except json.JSONDecodeError as e:
            print(f"Warning: Skipping line {idx+1} due to JSON decode error: {e}")
            continue

        prompt = item.get(prompt_key, "")
        original_answer = item.get(answer_key, "")

        # 从 prompt 或 metadata 中提取 task_dir
        # 这里假设你的数据中已经包含了 task 信息，或者可以通过某种方式映射
        # 你需要根据你的实际数据格式来调整这部分逻辑

        # 方法1: 如果数据中已经有 task_id 或 task_name
        task_id = item.get("task_id") or item.get("task_name")
        if task_id:
            # 根据 task_id 构建 task_dir
            # 例如: task_7 -> /path/to/robust-kbench/tasks/kernelbench/level_2/task_7
            # 你需要根据实际的目录结构来调整
            level = item.get("level", "level_2")  # 默认 level_2
            task_dir = str(tasks_dir / level / f"task_{task_id}")
        else:
            # 方法2: 如果数据中直接包含 task_dir
            task_dir = item.get("task_dir")
            if not task_dir:
                # 方法3: 从 prompt 中解析（需要根据你的 prompt 格式调整）
                # 这里提供一个示例，你需要根据实际情况修改
                print(
                    f"Warning: Line {idx+1} does not have task_dir. "
                    f"Please ensure each sample has task_dir in the data."
                )
                continue

        # 验证 task_dir 是否存在
        if not os.path.exists(task_dir):
            print(
                f"Warning: task_dir does not exist for line {idx+1}: {task_dir}\n"
                f"Skipping this sample."
            )
            continue

        # 构建新的 answer 字典，包含 task_dir
        answer_dict: dict[str, Any] = {
            "task_dir": task_dir,
        }

        # 如果原始 answer 是字符串，可以保留它
        if isinstance(original_answer, str) and original_answer:
            answer_dict["reference_answer"] = original_answer

        # 如果数据中有其他元数据，也可以添加到 answer_dict 中
        # 例如：cuda_code_path, round, branch, iter 等
        if "cuda_code_path" in item:
            answer_dict["cuda_code_path"] = item["cuda_code_path"]
        if "round" in item:
            answer_dict["round"] = item["round"]
        if "branch" in item:
            answer_dict["branch"] = item["branch"]
        if "iter" in item:
            answer_dict["iter"] = item["iter"]

        # 构建处理后的数据项
        processed_item = {
            prompt_key: prompt,
            answer_key: answer_dict,
        }

        # 保留其他字段
        for key, value in item.items():
            if key not in [prompt_key, answer_key]:
                processed_item[key] = value

        processed_data.append(processed_item)

    # 写入处理后的数据
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    with open(output_data_path, "w", encoding="utf-8") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully processed {len(processed_data)} samples.")
    print(f"Output saved to: {output_data_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare kernel training data with task_dir in answers"
    )
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
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

    args = parser.parse_args()

    prepare_kernel_data(
        input_data_path=args.input_data,
        output_data_path=args.output_data,
        robust_kbench_root=args.robust_kbench_root,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
    )


if __name__ == "__main__":
    main()
