#!/bin/bash
# 运行 Qwen3-4B KernelBench GRPO 训练脚本

set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$SCRIPT_DIR"))  # examples/kernel -> examples -> RLinf
CONFIG_PATH="${REPO_PATH}/examples/reasoning/config/kernel"
MEGATRON_PATH=/opt/Megatron-LM

# 强制使用 .venv（已安装 transformers 和 sglang）
# conda 环境磁盘空间不足，无法安装 transformers
if [ -f "${REPO_PATH}/.venv/bin/activate" ]; then
    # 彻底清理 conda 环境变量（防止 .zshrc 自动激活的影响）
    unset CONDA_DEFAULT_ENV
    unset CONDA_PREFIX
    unset CONDA_PROMPT_MODIFIER
    unset CONDA_PYTHON_EXE
    unset CONDA_SHLVL
    unset _CONDA_ROOT
    # 取消激活 conda（如果已激活）
    if command -v conda &> /dev/null; then
        conda deactivate 2>/dev/null || true
    fi
    # 从 PATH 中移除所有 conda 相关路径
    export PATH=$(echo $PATH | tr ':' '\n' | grep -v "/mnt/data/qinhaiyan/miniconda3" | grep -v "conda" | tr '\n' ':' | sed 's/:$//' | sed 's/^://')
    # 激活 .venv
    source ${REPO_PATH}/.venv/bin/activate
    echo "已激活虚拟环境: ${REPO_PATH}/.venv"
    echo "Python 路径: $(which python)"
    echo "Python 版本: $(python --version)"
    # 验证确实使用的是 .venv 的 Python
    if [[ "$(which python)" != *".venv"* ]]; then
        echo "❌ 错误: Python 路径不包含 .venv，当前路径: $(which python)"
        exit 1
    fi
else
    echo "错误: 找不到 .venv 环境"
    exit 1
fi

PYTHON_CMD="python"

# 设置 PYTHONPATH，包含 RLinf 和 robust-kbench（RLinf 必须在最前面）
ROBUST_KBENCH_PATH=/home/qinhaiyan/robust-kbench
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${ROBUST_KBENCH_PATH}:$PYTHONPATH

# 验证环境
echo "验证环境..."
${PYTHON_CMD} -c "import rlinf; print('✓ RLinf available')" || {
    echo "❌ RLinf 无法导入，请检查安装"
    exit 1
}

# 检查 robust-kbench 是否可导入
echo "检查 robust-kbench 环境..."
${PYTHON_CMD} -c "import sys; sys.path.insert(0, '${ROBUST_KBENCH_PATH}'); import robust_kbench; print(f'robust-kbench 路径: {robust_kbench.__file__}')" || {
    echo "警告: 无法直接导入 robust-kbench，将在运行时尝试"
}

# 配置名称
CONFIG_NAME="qwen3-4b-kernelbench-grpo"

echo "=========================================="
echo "开始 KernelBench GRPO 训练"
echo "模型: Qwen3-4B"
echo "配置: ${CONFIG_NAME}"
echo "=========================================="

# 运行训练
CUDA_VISIBLE_DEVICES=0 \
${PYTHON_CMD} ${REPO_PATH}/examples/reasoning/main_grpo.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME}
