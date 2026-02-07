# Git 认证设置指南

## 问题
GitHub 已不支持密码认证，需要使用 Personal Access Token (PAT) 或 SSH 密钥。

## 方法 1: 使用 Personal Access Token (推荐，快速)

### 步骤 1: 创建 Token

1. 访问 GitHub Token 设置页面：
   ```
   https://github.com/settings/tokens
   ```

2. 点击 **"Generate new token"** → **"Generate new token (classic)"**

3. 填写信息：
   - **Note**: 输入描述，如 "RLinf Push Token"
   - **Expiration**: 选择过期时间（建议 90 天或自定义）
   - **Select scopes**: 勾选 **`repo`** (完整仓库访问权限)

4. 点击 **"Generate token"**

5. **重要**: 复制生成的 token（只显示一次！）
   - Token 格式类似：`ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### 步骤 2: 使用 Token 推送

**方式 A: 在推送时输入 token 作为密码**

```bash
cd /home/qinhaiyan/RLinf
git push origin codex/kernel-eval
# Username: qhy991
# Password: <粘贴你的 token，不是 GitHub 密码>
```

**方式 B: 使用 Git Credential Helper 保存 token**

```bash
# 配置 Git 保存凭据（可选，避免每次输入）
git config --global credential.helper store

# 然后推送，输入一次 token 后会自动保存
git push origin codex/kernel-eval
```

**方式 C: 在 URL 中直接使用 token（不推荐，安全性较低）**

```bash
# 临时使用（不保存到配置）
git push https://qhy991:YOUR_TOKEN@ghfast.top/github.com/qhy991/RLinf.git codex/kernel-eval
```

## 方法 2: 使用 SSH 密钥（长期方案）

### 步骤 1: 检查是否已有 SSH 密钥

```bash
ls -la ~/.ssh/id_*.pub
```

如果有 `id_rsa.pub` 或 `id_ed25519.pub`，跳到步骤 3。

### 步骤 2: 生成 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# 按 Enter 使用默认路径
# 可以设置密码或直接按 Enter（空密码）
```

### 步骤 3: 添加 SSH 密钥到 GitHub

1. 复制公钥内容：
   ```bash
   cat ~/.ssh/id_ed25519.pub
   # 或
   cat ~/.ssh/id_rsa.pub
   ```

2. 访问 GitHub SSH 设置：
   ```
   https://github.com/settings/keys
   ```

3. 点击 **"New SSH key"**
   - **Title**: 输入描述，如 "My Dev Machine"
   - **Key**: 粘贴刚才复制的公钥内容
   - 点击 **"Add SSH key"**

### 步骤 4: 修改 Git Remote 为 SSH

```bash
cd /home/qinhaiyan/RLinf
git remote set-url origin git@github.com:qhy991/RLinf.git
git push origin codex/kernel-eval
```

## 方法 3: 使用 GitHub CLI (gh)

如果安装了 GitHub CLI：

```bash
# 登录
gh auth login

# 然后正常推送
git push origin codex/kernel-eval
```

## 推荐方案

- **快速方案**: 使用方法 1（Personal Access Token），配置 credential helper 保存一次即可
- **长期方案**: 使用方法 2（SSH 密钥），更安全且方便

## 故障排除

### Token 无效
- 检查 token 是否过期
- 确认 token 有 `repo` 权限
- 重新生成新 token

### SSH 连接失败
```bash
# 测试 SSH 连接
ssh -T git@github.com

# 如果失败，检查 ~/.ssh/config
```

### 仍然提示密码认证
- 确认 remote URL 是否正确
- 清除已保存的凭据：`git credential-cache exit` 或 `git credential reject`
