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

## 查看 ma-user 所有的一级文件和文件夹大小

```bash
find /ma-user/work -mindepth 1 -maxdepth 1 -user ma-user -exec du -shx {} + 2>/dev/null | sort -h
```

该命令会筛选 `/ma-user/work` 下所有者为 `ma-user` 的一级文件和文件夹，并按大小升序显示。如果服务器的实际持久化路径是 `/home/ma-user/work`，需要将命令中的路径相应替换。

> 注意：文件所有者是 `ma-user` 不一定代表它会计入当前容器的容量配额，配额通常按文件系统或项目计算。
