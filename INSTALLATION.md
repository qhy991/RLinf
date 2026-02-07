# RLinf 安装说明

## 安装方式

本项目使用 **uv** 和虚拟环境进行安装，**不需要 conda**。

## 当前安装状态

- ✅ Python 3.11.12（通过 uv 管理）
- ✅ uv 已安装
- ⏳ RLinf 依赖安装中（后台运行）

## 安装完成后使用

### 激活虚拟环境

```bash
cd /home/qinhaiyan/RLinf
source .venv/bin/activate
```

### 验证安装

```bash
python --version  # 应该显示 Python 3.11.12
python -c "import rlinf; print('RLinf 安装成功')"
```

### 运行示例

参考 [RLinf 文档](https://rlinf.readthedocs.io/en/latest/) 和 `examples/` 目录中的示例。

## 检查安装进度

使用提供的检查脚本：

```bash
cd /home/qinhaiyan/RLinf
bash check_install.sh
```

## 注意事项

- 安装过程可能需要 30 分钟到 1 小时，取决于网络速度和编译时间
- 虚拟环境位于 `RLinf/.venv/` 目录
- 使用 `uv` 管理 Python 版本和依赖，无需 conda
