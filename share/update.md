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

### 1b. 每次都只下到 50–200MB 就断 → 分块并行下载（推荐）

这是本机实测到的最终形态：**单条 TCP 流在这个镜像上活不过 50–200 MB**，
无论 `uv` 还是 `curl -C -` 都一样 —— 续传也一样，因为每段新连接同样会断，
永远凑不齐 782 MB。

根因不在工具，在链路：单流有硬上限。解法是**把它拆成许多条独立的小流**，
每条只负责 8 MiB，断了只重来这一小块。这样一次 782 MB 的失败被摊成
~100 个小任务，每个都能廉价重试。

镜像支持 `Range` 请求（实测 `206`），所以分块可行。

```bash
cat > fetch_big.py <<'PY'
#!/usr/bin/env python3
"""分块并行下载，支持断点续传。失败就重跑，已完成的块不会重下。

    python3 fetch_big.py <url> <out-file> <total-bytes> [sha256] [jobs]
"""
import hashlib, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

CHUNK = 8 << 20          # 每块 8 MiB
MAX_ATTEMPTS = 500
READ_TIMEOUT = 120


def sizeof(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def fetch_chunk(unit):
    url, path, start, end = unit
    want = end - start + 1
    for attempt in range(1, MAX_ATTEMPTS + 1):
        have = sizeof(path)
        if have >= want:
            return None
        req = urllib.request.Request(
            url, headers={"Range": "bytes=%d-%d" % (start + have, end)})
        try:
            resp = urllib.request.urlopen(req, timeout=READ_TIMEOUT)
            try:
                fh = open(path, "ab" if have else "wb")
                try:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        fh.write(block)
                finally:
                    fh.close()
            finally:
                resp.close()
        except Exception as exc:
            time.sleep(2)
            if attempt % 20 == 0:
                sys.stderr.write("  chunk@%d: %d tries, %d/%d bytes (%s)\n"
                                 % (start, attempt, sizeof(path), want,
                                    type(exc).__name__))
                sys.stderr.flush()
            continue
    return "chunk@%d gave up at %d/%d bytes" % (start, sizeof(path), want)


def sha256_of(path):
    h = hashlib.sha256()
    fh = open(path, "rb")
    try:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            h.update(block)
    finally:
        fh.close()
    return h.hexdigest()


def main():
    if len(sys.argv) < 4:
        sys.stderr.write(__doc__)
        return 2
    url, out, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sha = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
    jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 8

    parts = out + ".parts"
    if not os.path.isdir(parts):
        os.makedirs(parts)

    units = [(url, os.path.join(parts, str(i)), start,
              min(start + CHUNK - 1, total - 1))
             for i, start in enumerate(range(0, total, CHUNK))]
    print("%d chunks of %d MiB, %d parallel" % (len(units), CHUNK >> 20, jobs))

    round_no = 0
    while True:
        round_no += 1
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            failures = [e for e in pool.map(fetch_chunk, units) if e]
        have = sum(sizeof(u[1]) for u in units)
        print("round %d: %d / %d bytes (%.1f%%)"
              % (round_no, have, total, 100.0 * have / total))
        sys.stdout.flush()
        if have >= total:
            break
        if failures and round_no > MAX_ATTEMPTS:
            sys.stderr.write("aborting: %s\n" % failures[0])
            return 1

    dst = open(out, "wb")
    try:
        for unit in units:
            src = open(unit[1], "rb")
            try:
                while True:
                    block = src.read(1 << 20)
                    if not block:
                        break
                    dst.write(block)
            finally:
                src.close()
    finally:
        dst.close()

    got = sizeof(out)
    if got != total:
        sys.stderr.write("SIZE MISMATCH: %d != %d\n" % (got, total))
        return 1
    if sha:
        actual = sha256_of(out)
        if actual != sha:
            sys.stderr.write("SHA256 MISMATCH\n  got  %s\n  want %s\n"
                             % (actual, sha))
            return 1
    for unit in units:
        os.remove(unit[1])
    os.rmdir(parts)
    print("OK  %s  %d bytes" % (out, got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
PY
```

用法（`cp311` 按第 0 步的 `python -V` 替换）：

```bash
python3 fetch_big.py \
  'https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl' \
  torch-2.11.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
  820214272 \
  c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad \
  8
```

第 4 个参数是 `uv.lock` 里记的 sha256，脚本会自己校验 —— 不用再手工 `ls -l`
对字节数。第 5 个是并发数，掉速就调到 4。

**中断了就重跑同一条命令**：已下完的块直接跳过，只有没下完的那块会接着下。

> macOS 上把 `python3` 换成 `.venv/bin/python` 也行，脚本只用标准库，
> 3.8 以上都能跑。

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
