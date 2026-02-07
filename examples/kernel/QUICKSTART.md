# KernelBench 训练快速开始指南

本指南提供在 RLinf 上运行 KernelBench 训练的最短路径。

## 前置条件

1. **环境检查**：确保 `robust-kbench` 可导入
   ```bash
   python -c "import robust_kbench; print(robust_kbench.__file__)"
   ```
   
   如果失败，安装它：
   ```bash
   pip install -e /path/to/robust-kbench
   ```
   
   或者将 `robust-kbench` 放在与 `RLinf` 同级的目录。

2. **准备数据集**：确保数据集中的 `answers` 字段包含 `task_dir`

## 快速开始（3 步）

### 1. 准备数据集（JSONL 格式）

每条样本的格式：
```json
{"prompt": "请根据 KernelBench 任务生成 CUDA kernel ...", "answer": {"task_dir": "/path/to/robust-kbench/tasks/kernelbench/level_2/task_7"}}
```

**最小测试数据集**：`examples/kernel/kernelbench_minimal.jsonl`

你可以使用数据准备脚本：
```bash
python examples/kernel/prepare_kernel_data.py \
    --input_data /path/to/raw_data.jsonl \
    --output_data /path/to/kernelbench_train.jsonl \
    --robust_kbench_root /path/to/robust-kbench
```

验证数据格式：
```bash
python examples/kernel/validate_kernel_data.py \
    --data_path /path/to/kernelbench_train.jsonl
```

### 2. 配置 YAML

已提供配置文件：`examples/reasoning/config/kernel/qwen2.5-1.5b-kernelbench-grpo.yaml`

**关键配置项**：

```yaml
data:
  type: math
  prompt_key: prompt
  answer_key: answer
  train_data_paths: ["/path/to/your/kernelbench_train.jsonl"]
  val_data_paths: ["/path/to/your/kernelbench_val.jsonl"]

reward:
  reward_type: kernel
  op_atol: 1e-3
  op_rtol: 1e-3
  warmup_time: 25
  rep_time: 100
  timeout: 600
  num_correct_trials: 5
  correctness_first: true
  skip_torch_eval: false
  eval_type: kernelbench
  isolate_execution: true
  kernel_output_dir: /tmp/rlinf_kernel_eval
```

**需要修改的路径**：
- `data.train_data_paths` 和 `data.val_data_paths`：你的数据集路径
- `rollout.model.model_path`：模型路径
- `actor.tokenizer.tokenizer_model`：tokenizer 路径
- `reward.kernel_output_dir`：生成的 kernel 文件保存目录（可选）

### 3. 运行训练

**方法 1：使用提供的脚本**
```bash
# 修改脚本中的路径（ROBUST_KBENCH_PATH 等）
chmod +x examples/kernel/run_kernelbench_grpo.sh
./examples/kernel/run_kernelbench_grpo.sh
```

**方法 2：直接运行 Python**
```bash
export PYTHONPATH=/path/to/RLinf:/path/to/robust-kbench:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 \
python examples/reasoning/main_grpo.py \
  --config-path examples/reasoning/config/kernel \
  --config-name qwen2.5-1.5b-kernelbench-grpo
```

## 最小验证（1-2 个 step）

为了快速验证流程，可以：

1. **使用最小数据集**：`examples/kernel/kernelbench_minimal.jsonl`（2 条样本）
2. **修改配置**：
   ```yaml
   runner:
     max_steps: 2  # 只跑 2 步
   
   data:
     rollout_batch_size: 2  # 小 batch size
   ```
3. **运行训练**：观察是否能正常完成评估和奖励计算

## 数据格式说明

### 必需的字段

- `prompt`：输入提示文本
- `answer.task_dir`：robust-kbench 任务目录路径（**必需**）

### 可选的字段（覆盖配置默认值）

```json
{
  "answer": {
    "task_dir": "/path/to/task_7",
    "cuda_code_path": "/path/to/kernel.cu",  // 可选
    "round": 1,  // 可选
    "branch": 0,  // 可选
    "iter": 3,  // 可选
    "op_atol": 1e-3,  // 可选
    "op_rtol": 1e-3,  // 可选
    "warmup_time": 25,  // 可选
    "rep_time": 100,  // 可选
    "timeout": 600,  // 可选
    "num_correct_trials": 5,  // 可选
    "correctness_first": true,  // 可选
    "skip_torch_eval": false,  // 可选
    "isolate_execution": true,  // 可选
    "backward": false,  // 可选
    "enable_profile": false  // 可选
  }
}
```

## 常见问题

### Q: 如何找到 task_dir？

A: `task_dir` 应该指向 robust-kbench 中的具体任务目录，例如：
- `/path/to/robust-kbench/tasks/kernelbench/level_2/task_7`
- 每个任务目录包含 `test.py`、`reference.py` 等文件

### Q: 训练时出现 "task_dir is required" 错误？

A: 确保：
1. 数据集中每条样本的 `answer` 是字典格式
2. `answer` 中包含 `task_dir` 字段
3. `task_dir` 路径存在且可访问

### Q: 如何查看生成的 kernel 文件？

A: 生成的 `.cu` 文件保存在 `reward.kernel_output_dir` 指定的目录中，按 `round/branch/iter` 组织。

### Q: 评估很慢怎么办？

A: 可以调整：
- `reward.timeout`：减少超时时间
- `reward.rep_time`：减少性能测试重复次数
- `reward.skip_torch_eval`：跳过 torch 评估（如果不需要）
- `reward.correctness_first`：先做正确性检查，失败则跳过性能测试

## 相关文件

- 数据准备脚本：`examples/kernel/prepare_kernel_data.py`
- 数据验证脚本：`examples/kernel/validate_kernel_data.py`
- 配置文件：`examples/reasoning/config/kernel/qwen2.5-1.5b-kernelbench-grpo.yaml`
- 训练脚本：`examples/kernel/run_kernelbench_grpo.sh`
- 详细文档：`examples/kernel/README.md`
