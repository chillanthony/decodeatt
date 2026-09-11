# 远程环境升级 + vLLM 安装

commit `f673667` 之后，远程机器按本文档操作。

## 0. 先看一眼 Python 版本

wheel 文件名带 `cp310` / `cp311` / `cp312`，下面所有命令都要按它替换。

```bash
cd <repo>
git pull
.venv/bin/python -V
```

## 1. 拉依赖

```bash
uv sync --locked --extra efficiency
```

期望结果：`torch 2.11.0+cu128` / `transformers 5.17.0` / `vllm 0.25.1`。

### 1a. 如果报 `failed to write to the distribution cache`

典型报错：

```
failed to write to the distribution cache
error decoding response body
request or response body error
error reading a body from connection
end of file before message length reached
```

这是**传输层被截断** —— torch 的 cu128 wheel 有 820,214,272 字节（约 782 MB），
镜像响应体的 `Content-Length` 还没读完，连接就被对端关了。不是缓存脏、不是配置错，
所以 `uv cache clean` 之后一定会复发。

`uv` 默认单次读超时只有 30 秒、HTTP 重试只有 3 次，对 800MB 的包不够。提高这两个：

```bash
UV_HTTP_TIMEOUT=900 \
UV_HTTP_RETRIES=50 \
UV_HTTP_CONNECT_TIMEOUT=60 \
UV_CONCURRENT_DOWNLOADS=1 \
uv sync --locked --extra efficiency
```

> `UV_CONCURRENT_DOWNLOADS=1` 也是关键：默认会并发拉多个包，带宽被摊薄后
> 大文件更容易超时。设成 1 让 torch 独占带宽。

### 1b. 仍然失败 → 断点续传拉 wheel（推荐）

先清掉可能存在的残包：

```bash
uv cache clean torch
```

然后循环续传，直到字节数对得上为止。这个脚本的核心是 `curl -C -`：
每次从断点接着下，断了自动重来，**不会前功尽弃**。

```bash
cat > fetch_torch.sh <<'EOF'
URL='https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl'
OUT=torch-2.11.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
EXPECT=820214272

# Linux: stat -c%s   /   macOS: stat -f%z
sizeof() { stat -c%s "$1" 2>/dev/null || echo 0; }

while [ "$(sizeof "$OUT")" -lt "$EXPECT" ]; do
  echo "have $(sizeof "$OUT") / $EXPECT"
  curl -L -C - --retry 50 --retry-delay 3 --retry-all-errors -o "$OUT" "$URL" || true
done
echo "done: $(sizeof "$OUT") bytes"
EOF

bash fetch_torch.sh
```

> 把 `cp311` 换成上一步 `python -V` 得到的版本。
> 下载前可以先确认镜像支持断点续传：`curl -sL -o /dev/null -w "%{http_code}\n" -r 0-1023 "$URL"`
> 返回 `206` 才说明续传有效（已验证返回 `206`）。

校验大小必须完全等于 `820214272`：

```bash
ls -l torch-2.11.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
```

### 1c. 本地安装 torch，再让 uv 跳过它

```bash
uv pip install ./torch-2.11.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
uv sync --locked --extra efficiency --no-install-package torch
```

## 2. 验证

```bash
.venv/bin/python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
```

期望输出：`2.11.0+cu128 True 0.25.1`。

`cuda.is_available()` 为 `False` 说明驱动/设备没挂上，先解决这个再往下走。

## 3. 打 R-KV 端口的 vLLM patch

```bash
bash scripts/apply_rkv.sh
uv pip install -e vllm-src
```

> **注意**：`scripts/apply_rkv.sh` 与 `vllm-src/` 目前尚未提交到本仓库
> （需从上游 https://github.com/zefan-cai/r-kv 取 vLLM 端口部分，落到 `rkv-efficiency/`）。
> 这一步在两个文件就位前跑不了。

## 4. 之后

按 `note/knowledge/rkvrepro.md` 的「实现流程」：

```bash
cd vLLM/benchmark
./prepare_data.sh
python eval.py --n 200 --label fullkv
VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 python eval.py --n 200 --label rkv_b256_buf128
```

---

## 为什么锁定这些版本

| 组件 | 版本 | 原因 |
|---|---|---|
| torch | `2.11.0` (cu128) | vLLM 0.25.1 精确要求 `torch==2.11.0`；也是 cu128 索引上的最高版本，匹配 CUDA 12.8 / 驱动 550，无需升驱动 |
| transformers | `>=5.5.3,<6` | vLLM 0.25.1 的下限；实际锁到 5.17.0 |
| vllm | `0.25.1` | R-KV 上游 vLLM 端口所 pin 的版本 |

**不要**升到 cu129 / cu130 —— 那需要驱动 >= 575 / 580。

torch 索引用的是上海交大镜像而非 `download.pytorch.org`：后者在国内机器上
`failed to fetch`。镜像内容与官方一致（sha256 完全相同），已核对。
