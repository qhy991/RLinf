#!/bin/bash
# 运行 KernelBench GRPO 训练脚本
# 使用方法: ./run_kernelbench_grpo.sh [config_name]

set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_DIR")
CONFIG_PATH="${REPO_PATH}/examples/reasoning/config/kernel"
MEGATRON_PATH=/opt/Megatron-LM

# 设置 PYTHONPATH，包含 RLinf 和 robust-kbench
# 请根据实际情况修改 robust-kbench 的路径
ROBUST_KBENCH_PATH=/path/to/robust-kbench
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${ROBUST_KBENCH_PATH}:$PYTHONPATH

# 检查 robust-kbench 是否可导入
echo "检查 robust-kbench 环境..."
python -c "import robust_kbench; print(f'robust-kbench 路径: {robust_kbench.__file__}')" || {
    echo "错误: 无法导入 robust-kbench"
    echo "请确保:"
    echo "  1. robust-kbench 已安装: pip install -e /path/to/robust-kbench"
    echo "  2. 或者将 robust-kbench 放在与 RLinf 同级的目录"
    echo "  3. 或者修改脚本中的 ROBUST_KBENCH_PATH 变量"
    exit 1
}

# 配置名称，默认为 qwen2.5-1.5b-kernelbench-grpo
if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-kernelbench-grpo"
else
    CONFIG_NAME=$1
fi

# 运行训练
python ${REPO_PATH}/examples/reasoning/main_grpo.py \
    --config-path ${CONFIG_PATH} \
    --config-name ${CONFIG_NAME}
