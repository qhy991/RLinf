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
验证 kernel 训练数据格式。

当前推荐格式：
  - answer: CUDA 代码字符串（可为空，模型会生成）
  - meta.task_dir: 指向 robust-kbench 任务目录

使用方法：
    python validate_kernel_data.py --data_path /path/to/kernel_data.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def validate_kernel_data(
    data_path: str,
    prompt_key: str = "prompt",
    answer_key: str = "answer",
    meta_key: str = "meta",
) -> bool:
    """
    验证 kernel 训练数据格式。

    Args:
        data_path: 数据文件路径（JSONL 格式）
        prompt_key: prompt 字段的键名
        answer_key: answer 字段的键名
        meta_key: meta 字段的键名

    Returns:
        bool: 如果所有样本都有效则返回 True，否则返回 False
    """
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return False

    errors = []
    warnings = []
    valid_count = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError as e:
                errors.append(f"Line {idx}: JSON decode error: {e}")
                continue

            # 检查 prompt 字段
            if prompt_key not in item:
                errors.append(f"Line {idx}: Missing '{prompt_key}' field")
                continue

            # 检查 answer 字段
            if answer_key not in item:
                errors.append(f"Line {idx}: Missing '{answer_key}' field")
                continue

            answer = item[answer_key]

            # 检查 answer 是否为字符串或字典（兼容旧格式）
            if not isinstance(answer, (str, dict)):
                errors.append(
                    f"Line {idx}: '{answer_key}' must be a string or dict, got {type(answer).__name__}"
                )
                continue

            if isinstance(answer, str) and not answer.strip():
                warnings.append(f"Line {idx}: '{answer_key}' is empty; model must generate CUDA code.")

            # 检查 meta/task_dir
            meta = item.get(meta_key, {})
            task_dir = None
            if meta is not None:
                if not isinstance(meta, dict):
                    errors.append(
                        f"Line {idx}: '{meta_key}' must be a dict if provided, got {type(meta).__name__}"
                    )
                    continue
                task_dir = meta.get("task_dir")

            if task_dir is None and isinstance(answer, dict):
                # 兼容旧格式：task_dir 放在 answer 里
                task_dir = answer.get("task_dir")
                if task_dir is not None:
                    warnings.append(
                        f"Line {idx}: task_dir found in '{answer_key}' (legacy format). "
                        f"Recommend moving it to '{meta_key}'."
                    )

            if task_dir is None:
                errors.append(
                    f"Line {idx}: Missing 'task_dir' in '{meta_key}' (or legacy '{answer_key}')"
                )
                continue

            if not isinstance(task_dir, str):
                errors.append(
                    f"Line {idx}: 'task_dir' must be a string, got {type(task_dir).__name__}"
                )
                continue

            # 检查 task_dir 路径是否存在
            task_dir_path = Path(task_dir).expanduser()
            if not task_dir_path.exists():
                warnings.append(
                    f"Line {idx}: task_dir does not exist: {task_dir}"
                )
            elif not task_dir_path.is_dir():
                warnings.append(
                    f"Line {idx}: task_dir is not a directory: {task_dir}"
                )

            # 检查 task_dir 中是否有必要的文件（可选）
            if task_dir_path.exists() and task_dir_path.is_dir():
                test_py = task_dir_path / "test.py"
                if not test_py.exists():
                    warnings.append(
                        f"Line {idx}: task_dir does not contain 'test.py': {task_dir}"
                    )

            valid_count += 1

    # 打印结果
    print(f"\n验证完成:")
    print(f"  有效样本数: {valid_count}")
    print(f"  错误数: {len(errors)}")
    print(f"  警告数: {len(warnings)}")

    if errors:
        print(f"\n错误列表:")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... 还有 {len(errors) - 10} 个错误")

    if warnings:
        print(f"\n警告列表:")
        for warning in warnings[:10]:  # 只显示前10个警告
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... 还有 {len(warnings) - 10} 个警告")

    if errors:
        print(f"\n❌ 数据验证失败，请修复上述错误后重试。")
        return False
    elif warnings:
        print(f"\n⚠️  数据验证通过，但有警告。建议修复警告后再使用。")
        return True
    else:
        print(f"\n✅ 数据验证通过！")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate kernel training data format")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data file (JSONL format)",
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

    args = parser.parse_args()

    success = validate_kernel_data(
        data_path=args.data_path,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        meta_key=args.meta_key,
    )

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
