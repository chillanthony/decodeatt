# 使用 uv 构建持久化环境

在与训练任务相同的 Linux 镜像中执行：

```bash
cd /你的/decodeatt/代码目录

mkdir -p /home/ma-user/work/.venvs
export UV_PROJECT_ENVIRONMENT=/home/ma-user/work/.venvs/decodeatt
export UV_CACHE_DIR=/home/ma-user/work/.cache/uv

uv sync --frozen --python 3.11
```

验证环境：

```bash
/home/ma-user/work/.venvs/decodeatt/bin/python -c \
  'import torch, transformers; print(torch.__version__, torch.cuda.is_available())'
```

验证成功后直接运行：

```bash
bash scripts/jiqun/smoke_test.sh
```

第一次执行 `uv sync` 会比较久，之后依赖和缓存都保存在 `/home/ma-user/work`，通常不需要重新下载。
