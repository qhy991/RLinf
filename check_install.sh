#!/bin/bash
# RLinf 安装进度检查脚本

echo "=== RLinf 安装进度检查 ==="
echo ""

# 检查安装进程
echo "1. 安装进程状态:"
if ps aux | grep -E "install.sh|uv sync" | grep -v grep > /dev/null; then
    echo "   ✓ 安装进程正在运行"
    ps aux | grep -E "install.sh|uv sync" | grep -v grep | head -2
else
    echo "   ✗ 未发现安装进程（可能已完成或未启动）"
fi

echo ""

# 检查虚拟环境
echo "2. 虚拟环境状态:"
if [ -d ".venv" ]; then
    echo "   ✓ 虚拟环境目录已创建"
    if [ -f ".venv/.lock" ]; then
        echo "   ⚠ 检测到锁定文件，安装可能正在进行中"
    fi
    if [ -d ".venv/bin" ]; then
        echo "   ✓ 虚拟环境已初始化"
        echo "   Python 版本: $(.venv/bin/python --version 2>/dev/null || echo '未就绪')"
    else
        echo "   ⏳ 虚拟环境正在初始化..."
    fi
else
    echo "   ✗ 虚拟环境尚未创建"
fi

echo ""

# 检查已安装的包
echo "3. 已安装的包数量:"
if [ -d ".venv/lib" ]; then
    PKG_COUNT=$(find .venv/lib/python*/site-packages -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo "   已安装包: $PKG_COUNT"
else
    echo "   尚未开始安装包"
fi

echo ""
echo "=== 提示 ==="
echo "要查看实时安装日志，可以运行:"
echo "  tail -f /proc/\$(pgrep -f 'install.sh' | head -1)/fd/1 2>/dev/null"
echo ""
echo "安装完成后，激活环境使用:"
echo "  source .venv/bin/activate"
