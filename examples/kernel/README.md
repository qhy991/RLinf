# Kernel 训练数据准备指南

本指南说明如何准备用于 RLinf kernel 训练的数据，确保每条样本的 `meta.task_dir` 指向 kernelbench 任务目录。

## 数据格式要求

### 输入数据格式（JSONL）

每条样本必须是一个 JSON 对象，包含以下字段：

```json
{
  "prompt": "你的 prompt 文本",
  "answer": "__global__ void ...",
  "meta": {
    "task_dir": "/path/to/robust-kbench/tasks/kernelbench/level_2/task_7"
  }
}
```

### 完整的 meta 字段示例

`meta` 字段可以包含以下字段：

```json
{
  "task_dir": "/path/to/robust-kbench/tasks/kernelbench/level_2/task_7",
  "cuda_code_path": "/path/to/your/kernel.cu",  // 可选
  "round": 0,  // 可选
  "branch": 0,  // 可选
  "iter": 0,  // 可选
  "op_atol": 1e-3,  // 可选，覆盖配置中的默认值
  "op_rtol": 1e-3,  // 可选
  "warmup_time": 25,  // 可选
  "rep_time": 100,  // 可选
  "eval_type": "kernelbench",  // 可选
  "timeout": 600,  // 可选
  "num_correct_trials": 5,  // 可选
  "correctness_first": true,  // 可选
  "skip_torch_eval": false,  // 可选
  "isolate_execution": true,  // 可选
  "backward": false,  // 可选
  "enable_profile": false  // 可选
}
```

**注意**：`meta.task_dir` 是**必需**的字段，其他字段都是可选的。

## 使用数据准备脚本

我们提供了一个示例脚本来帮助你准备数据。

**方式 A：直接从 kernelbench 任务自动生成（推荐）**
```bash
python examples/kernel/prepare_kernel_data.py \
    --from_kernelbench_tasks \
    --output_data /path/to/output_data.jsonl \
    --robust_kbench_root /path/to/robust-kbench
```

如需自定义 prompt 模板：
```bash
python examples/kernel/prepare_kernel_data.py \
    --from_kernelbench_tasks \
    --output_data /path/to/output_data.jsonl \
    --robust_kbench_root /path/to/robust-kbench \
    --prompt_template_path /path/to/prompt_template.txt
```

**方式 B：从已有数据补充 meta**

```bash
python examples/kernel/prepare_kernel_data.py \
    --input_data /path/to/your/raw_data.jsonl \
    --output_data /path/to/output_data.jsonl \
    --robust_kbench_root /path/to/robust-kbench
```

### 脚本功能

脚本会：
1. 读取原始数据文件（JSONL 格式）
2. 为每条样本构建包含 `task_dir` 的 `meta` 字典
3. 验证 `task_dir` 路径是否存在
4. 输出处理后的数据文件

### 自定义数据映射

如果你的数据格式不同，你需要修改 `prepare_kernel_data.py` 脚本中的逻辑：

- 如果数据中有 `task_id` 字段，脚本会尝试构建 `task_dir` 路径
- 如果数据中已经有 `task_dir` 字段，脚本会直接使用
- 你需要根据你的实际数据格式来调整映射逻辑

## 手动准备数据示例

如果你需要手动准备数据，可以参考以下 Python 代码：

```python
import json

# 示例：准备一条样本
sample = {
    "prompt": "Write a CUDA kernel for matrix multiplication...",
    "answer": "__global__ void ...",
    "meta": {
        "task_dir": "/path/to/robust-kbench/tasks/kernelbench/level_2/task_7"
    },
}

# 写入 JSONL 文件
with open("kernel_data.jsonl", "w") as f:
    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

## 验证数据

在训练前，建议验证数据格式是否正确：

```python
import json

with open("kernel_data.jsonl", "r") as f:
    for idx, line in enumerate(f, 1):
        item = json.loads(line.strip())
        meta = item.get("meta", {})
        if not isinstance(meta, dict):
            print(f"Error at line {idx}: meta must be a dict")
            continue

        if "task_dir" not in meta:
            print(f"Error at line {idx}: missing 'task_dir' in meta")
            continue

        task_dir = meta["task_dir"]
        if not os.path.exists(task_dir):
            print(f"Warning at line {idx}: task_dir does not exist: {task_dir}")
```

## 配置示例

在 RLinf 配置文件中，确保 `data` 部分正确设置：

```yaml
data:
  type: math  # 或使用自定义的数据集类型
  train_data_paths:
    - /path/to/kernel_train_data.jsonl
  val_data_paths:
    - /path/to/kernel_val_data.jsonl
  prompt_key: prompt
  answer_key: answer
  meta_key: meta
  max_prompt_length: 2048
  apply_chat_template: false  # 根据你的模型调整
```

## 常见问题

### Q: task_dir 应该指向哪里？

A: `task_dir` 应该指向 robust-kbench 中的具体任务目录，例如：
- `/path/to/robust-kbench/tasks/kernelbench/level_2/task_7`
- 每个任务目录应该包含 `test.py`、`reference.py` 等文件

### Q: 如果我的数据中没有 task_dir 怎么办？

A: 你需要：
1. 确保你的数据源包含任务标识信息（task_id、task_name 等）
2. 根据任务标识构建对应的 task_dir 路径
3. 或者修改数据加载逻辑，在运行时动态构建 task_dir

### Q: 可以在运行时动态设置 task_dir 吗？

A: 可以，但需要修改数据加载器。建议在数据准备阶段就包含 task_dir，这样更清晰和高效。

## 相关文档

- [Kernel 评估文档](../docs/source-zh/rst_source/tutorials/extend/kernel_eval.rst)
- [RLinf 数据格式文档](../../docs/)
