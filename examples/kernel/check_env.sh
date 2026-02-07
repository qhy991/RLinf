#!/bin/bash
# 环境检查脚本：验证 KernelBench 训练所需的环境

set -e

echo "=========================================="
echo "KernelBench 训练环境检查"
echo "=========================================="
echo ""

# 1. 检查 Python 环境
echo "[1/4] 检查 Python 环境..."
python --version || {
    echo "❌ Python 未安装或不在 PATH 中"
    exit 1
}
echo "✅ Python 环境正常"
echo ""

# 2. 检查 robust-kbench 是否可导入
echo "[2/4] 检查 robust-kbench..."
if python -c "import robust_kbench; print(f'✅ robust-kbench 路径: {robust_kbench.__file__}')" 2>/dev/null; then
    echo "✅ robust-kbench 可正常导入"
else
    echo "❌ 无法导入 robust-kbench"
    echo ""
    echo "请执行以下操作之一："
    echo "  1. pip install -e /path/to/robust-kbench"
    echo "  2. 将 robust-kbench 放在与 RLinf 同级的目录"
    echo "  3. 设置 PYTHONPATH: export PYTHONPATH=/path/to/robust-kbench:\$PYTHONPATH"
    exit 1
fi
echo ""

# 3. 检查 CUDA 环境
echo "[3/4] 检查 CUDA 环境..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo "✅ CUDA 环境正常"
else
    echo "⚠️  未检测到 nvidia-smi，可能没有 GPU 或 CUDA 未安装"
fi
echo ""

# 4. 检查 RLinf 路径
echo "[4/4] 检查 RLinf 路径..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname "$SCRIPT_DIR")
if [ -f "${REPO_PATH}/rlinf/__init__.py" ]; then
    echo "✅ RLinf 路径: ${REPO_PATH}"
else
    echo "⚠️  未找到 RLinf 主目录，请检查路径"
fi
echo ""

echo "=========================================="
echo "环境检查完成！"
echo "=========================================="
