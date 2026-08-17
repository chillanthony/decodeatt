# Git Push HTTP 403 排查流程（不使用 GitHub CLI）

适用错误：

```text
RPC failed; HTTP 403 curl 22 The requested URL returned error: 403
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
```

HTTP 403 表示服务器拒绝了请求。应优先检查远程地址、缓存凭据、Personal Access Token（PAT）权限、代理和仓库规则，而不是修改 `http.postBuffer`。

## 1. 检查远程地址

```bash
git remote -v
git remote get-url origin
```

GitHub HTTPS 地址应类似：

```text
https://github.com/OWNER/REPOSITORY.git
```

如果 URL 中直接包含 token，例如 `https://TOKEN@github.com/...`，应立即从远程地址中删除 token，并在 GitHub 上撤销该 token：

```bash
git remote set-url origin https://github.com/OWNER/REPOSITORY.git
```

## 2. 检查 credential helper

```bash
git config --show-origin --get-all credential.helper
git config --show-origin --get-regexp '^credential\.'
```

常见 helper：

- `store`：凭据明文保存在 `~/.git-credentials`。
- `cache`：凭据临时缓存在内存中。
- `osxkeychain`：使用 macOS 钥匙串。
- `manager` 或 `manager-core`：使用 Git Credential Manager。

不要为了检查凭据而运行并公开 `git credential fill` 的结果，因为输出可能包含 token。

## 3. 清除旧的 GitHub 凭据

通用方法：

```bash
printf "protocol=https\nhost=github.com\n\n" | git credential reject
```

如果使用内存缓存：

```bash
git credential-cache exit
```

如果启用了 `credential.useHttpPath`，可针对仓库路径再清除一次：

```bash
printf "protocol=https\nhost=github.com\npath=OWNER/REPOSITORY.git\n\n" |
git credential reject
```

## 4. 使用 PAT 重新认证

```bash
GIT_TERMINAL_PROMPT=1 git push origin HEAD
```

出现提示时输入：

```text
Username: GitHub 用户名
Password: Personal Access Token
```

`Password` 必须填写 PAT，不能填写 GitHub 登录密码。输入 token 时终端通常不会显示字符，这是正常现象。

### Fine-grained PAT

创建 token 时确认：

1. `Resource owner` 是仓库所属账号或组织。
2. `Repository access` 包含目标仓库。
3. `Contents` 权限为 `Read and write`。
4. 如果提交修改了 `.github/workflows/`，为 `Workflows` 授予 `Read and write`。

### Classic PAT

- 私有仓库通常需要 `repo` scope。
- 修改 `.github/workflows/` 时还需要 `workflow` scope。
- 如果组织启用了 SSO，需要在 token 页面完成组织授权。

## 5. 分别测试读取和写入权限

测试读取：

```bash
git ls-remote origin
```

测试推送请求：

```bash
git push --dry-run origin HEAD
```

结果判断：

- `git ls-remote` 也返回 403：凭据无效、远程地址错误或账号没有仓库访问权限。
- `git ls-remote` 成功，但 dry-run 返回 403：PAT 没有写权限，或账号不是仓库协作者。
- dry-run 成功，但实际 push 失败：继续检查分支保护、仓库规则、大文件、Git LFS 或代理。

对于公开仓库，`git ls-remote` 成功不能证明认证成功，因为公开仓库允许匿名读取。

## 6. 检查代理

检查 Git 配置中的代理：

```bash
git config --show-origin --get-regexp '^http\.'
```

检查代理环境变量：

```bash
env | grep -i proxy
```

不要公开包含用户名、密码或 token 的代理地址。如果发现已经失效的全局代理配置，可删除：

```bash
git config --global --unset-all http.proxy
git config --global --unset-all https.proxy
```

某项不存在时，`--unset-all` 返回非零状态是正常的。

## 7. 检查待推送的大文件

先确认当前分支是否有 upstream：

```bash
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
```

检查待推送 blob 的总大小：

```bash
git rev-list --objects '@{upstream}'..HEAD |
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize)' |
awk '$1=="blob"{sum+=$3} END{printf "%.1f MB\n",sum/1048576}'
```

列出最大的待推送文件对象：

```bash
git rev-list --objects '@{upstream}'..HEAD |
git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
awk '$1=="blob"{print $3 "\t" $4}' |
sort -nr |
head -20
```

如果错误输出明确提到 Git LFS，再检查：

```bash
git lfs env
git lfs status
git lfs ls-files
```

Git LFS 返回 403 通常表示 LFS 端点使用了错误凭据、账号没有权限，或存储/带宽受到限制。

## 8. 保存完整错误信息

```bash
git push origin HEAD 2>&1 | tee git-push-error.log
```

重点查看 `RPC failed` 之前的 `remote:` 行，那里通常包含真正原因：

- `Permission denied`：账号或 PAT 没有写权限。
- `protected branch`：目标分支受保护，需要通过 pull request。
- `secret scanning` 或 `repository rule violations`：提交违反仓库规则。
- `file exceeds 100 MB`：提交包含 GitHub 不接受的普通 Git 大文件。
- `Git LFS ... 403`：普通 Git 与 LFS 的认证或配额情况不同。

## 推荐执行顺序

```bash
git remote -v
git config --show-origin --get-all credential.helper
printf "protocol=https\nhost=github.com\n\n" | git credential reject
GIT_TERMINAL_PROMPT=1 git push origin HEAD
```

如果仍然失败，再执行读取/写入权限测试、代理检查和大文件检查，并保留完整的 `remote:` 输出。
