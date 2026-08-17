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

### 最大对象不大时检查总 push 包

如果最大对象是 `1110993` 字节，它约为 `1.06 MiB`，远低于 GitHub 的单对象硬限制 `100 MiB`，因此基本可以排除“单个文件过大”。但大量小文件或大量历史对象仍可能让单次 push 过大；GitHub 对单次 push 强制执行 `2 GiB` 上限。

检查待推送提交数：

```bash
git rev-list --count '@{upstream}'..HEAD
```

检查待推送对象数：

```bash
git rev-list --objects '@{upstream}'..HEAD | wc -l
```

估算压缩后的 push 包大小：

```bash
git rev-list --objects '@{upstream}'..HEAD |
cut -d' ' -f1 |
git pack-objects --stdout 2>/dev/null |
wc -c |
awk '{printf "%.2f MiB\n", $1/1048576}'
```

结果判断：

- 接近或超过 `2048 MiB`：命中 GitHub 单次 push 的 `2 GiB` 限制，需要拆分推送。
- 达到几百 MiB：可能是代理、网络或网关在上传阶段中断，可换网络或改用 SSH 验证。
- 只有几十 MiB：基本可以排除大小问题，应重新检查代理、HTTP credential 和服务器返回的 `remote:` 信息。
- 如果输出明确出现 `Git LFS`：继续检查 LFS 状态和认证。

GitHub 官方说明：

- [Repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)
- [Troubleshooting the 2 GiB push limit](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit)

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

## 9. dry-run 成功但实际 push 失败

如果待推送压缩包只有约 `0.11 MiB`、对象数也很少，同时下面的 dry-run 可以成功：

```bash
git push --dry-run origin HEAD
```

那么可以排除普通 Git 大文件和基础写权限问题。dry-run 不会真正上传对象，因此实际 push 仍可能在 Git LFS、HTTP 请求体上传、代理转发或仓库内容检查阶段失败。

### 9.1 强制使用 HTTP/1.1

先进行一次临时测试：

```bash
git -c http.version=HTTP/1.1 push origin HEAD
```

如果成功，将配置应用到当前仓库：

```bash
git config http.version HTTP/1.1
```

### 9.2 检查 Git LFS

```bash
git lfs status
git lfs push --dry-run origin HEAD
```

如果提示没有 `git lfs` 命令，说明当前环境没有使用 Git LFS 客户端，可以跳过。如果 LFS 明确返回 403，查看最近日志：

```bash
git lfs logs last
```

### 9.3 检查工作流文件权限

```bash
git diff --name-only '@{upstream}'..HEAD |
grep '^\.github/workflows/'
```

如果有输出，确认 PAT 具有相应权限：

- Classic PAT：需要 `workflow` scope。
- Fine-grained PAT：需要 `Workflows: Read and write`。

### 9.4 临时绕过代理

macOS 或 Linux：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
git -c http.proxy= -c http.version=HTTP/1.1 push origin HEAD
```

如果这样能够成功，问题来自代理、VPN 或其他网络中间层。

### 9.5 生成精简诊断日志

```bash
GIT_TRACE=1 \
GIT_TRACE_CURL=1 \
GIT_TRACE_CURL_NO_DATA=1 \
git push origin HEAD 2>push-trace.log
```

提取关键行：

```bash
grep -Ei 'HTTP/|server:|via:|x-github|remote:|error|fatal|lfs' \
push-trace.log
```

分享日志前必须删除包含 `Authorization`、cookie、用户名、代理凭据或 token 的内容。

### 9.6 改用 SSH 绕过 Git Smart HTTP

如果已经在 GitHub 配置 SSH key：

```bash
ssh -T git@github.com
git remote set-url origin git@github.com:OWNER/REPOSITORY.git
git push origin HEAD
```

如果 SSH 的 22 端口不可用，可通过 443 端口连接：

```bash
ssh -T -p 443 git@ssh.github.com
git remote set-url origin \
ssh://git@ssh.github.com:443/OWNER/REPOSITORY.git
git push origin HEAD
```

这种情况下的推荐尝试顺序是：HTTP/1.1、LFS 检查、工作流权限、绕过代理、诊断日志，最后切换到 SSH。
