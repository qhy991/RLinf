#!/bin/bash
# 使用 Personal Access Token 推送代码
# 使用方法: ./push_with_token.sh

echo "请确保你已经创建了 GitHub Personal Access Token"
echo "如果没有，请访问: https://github.com/settings/tokens"
echo ""
echo "推送代码到 codex/kernel-eval 分支..."
echo "当提示输入密码时，请粘贴你的 Personal Access Token（不是 GitHub 密码）"
echo ""

git push origin codex/kernel-eval
