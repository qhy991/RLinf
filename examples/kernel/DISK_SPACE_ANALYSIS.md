# 磁盘空间问题分析

## 问题现象
在 conda 环境中安装 `transformers` 时出现 `OSError: [Errno 28] No space left on device` 错误。

## 磁盘空间检查结果

### 总体空间
- **总空间**: 718T
- **已用**: 500T  
- **可用**: 219T
- **使用率**: 70%

### 关键目录大小
- **conda 环境** (`/mnt/data/qinhaiyan/miniconda3/envs/rlinf`): 7.6G
- **Qwen3-4B 模型**: 7.6G
- **训练数据**: 385K

### conda 相关目录
- **构建目录** (`conda-bld`): 需要检查
- **包缓存** (`pkgs`): 需要检查

## 可能的原因

### 1. conda 构建缓存占用空间
conda 在安装包时会：
- 下载包到 `pkgs` 目录
- 在 `conda-bld` 目录构建包
- 使用临时目录进行解压和安装

即使总体空间充足，如果这些目录所在的分区空间不足，也会报错。

### 2. 临时目录空间不足
conda 安装过程会使用系统临时目录（`/tmp`），如果 `/tmp` 空间不足也会报错。

### 3. inode 耗尽
虽然磁盘空间充足，但如果文件系统 inode 耗尽，也会报 "No space left" 错误。

## 解决方案

### 方案 1: 使用 .venv 环境（推荐）
`.venv` 环境已经安装了所有必需的依赖（`transformers`, `sglang`, `vllm`），无需在 conda 中安装。

**优点**:
- 避免磁盘空间问题
- 环境已配置好，可直接使用
- 不占用 conda 环境空间

### 方案 2: 清理 conda 缓存
如果必须使用 conda 环境，可以清理缓存：

```bash
# 查看可清理的空间
conda clean --dry-run --all

# 清理所有缓存
conda clean --all -y

# 清理构建缓存
conda clean --builds -y

# 清理包缓存（保留已安装的包）
conda clean --packages -y
```

### 方案 3: 配置 conda 使用其他目录
可以配置 conda 使用空间更大的目录：

```bash
# 设置包缓存目录到空间更大的位置
conda config --add pkgs_dirs /path/to/larger/disk

# 设置构建目录
conda config --set croot /path/to/larger/disk/conda-bld
```

## 当前状态

### 问题解决进展（2026-02-07）

**磁盘空间问题**: ✅ 已解决
- 主分区可用: 219T（充足）
- Inode 使用率: 27%（充足）
- `/mnt/data/qinhaiyan` 用户配额已满，输出目录改为 `/home/qinhaiyan/rlinf_results`

**环境配置**: ✅ 已完成
- `.venv` 已安装: `transformers@4.51.1`, `sglang@0.4.6.post5`, `vllm@0.8.5`, `ray@2.47.0`
- Megatron-LM 已克隆到 `/tmp/Megatron-LM`
- OpenTelemetry 版本: 1.26.0（兼容 vllm）
- 训练脚本已强制使用 `.venv`，清理了所有 conda 环境变量

**训练初始化状态**: ⚠️ 部分成功
- ✅ vLLM Rollout Worker: 已成功初始化
- ✅ HF to Megatron checkpoint 转换: 成功
- ✅ CUDA Graph 编译: 成功（使用缓存）
- ❌ MegatronActor: `model_provider_func()` 兼容性问题

### 当前错误

```
TypeError: MegatronModelManager.model_provider_func() got an unexpected keyword argument 'config'
```

这是 RLinf 的 MegatronModelManager 与 Megatron-LM 训练代码之间的接口不兼容问题。

### 已修复的配置项

1. `apply_rope_fusion: False`（无需 Transformer Engine）
2. `skip_train: false`（Megatron 训练参数）
3. `output_dir: /home/qinhaiyan/rlinf_results`（磁盘配额问题）

### 运行命令

```bash
./examples/kernel/run_qwen3_4b_training.sh
```

### 后续建议

1. 检查 RLinf 是否有针对此 Megatron-LM 版本的修复
2. 或使用 RLinf 推荐的 Docker 镜像
3. 或联系 RLinf 维护者确认 Megatron-LM 版本兼容性

## 检查命令

```bash
# 检查磁盘空间
df -h /mnt/data/qinhaiyan

# 检查 inode
df -i /mnt/data/qinhaiyan

# 检查 conda 缓存大小
du -sh /mnt/data/qinhaiyan/miniconda3/pkgs
du -sh /mnt/data/qinhaiyan/miniconda3/conda-bld

# 检查临时目录
df -h /tmp
```
