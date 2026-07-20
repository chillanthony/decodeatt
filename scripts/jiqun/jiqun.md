# 集群环境配置

## 在用户主目录中构建 uv 环境

在项目根目录中执行：

```bash
cd /你的/decodeatt/代码目录

mkdir -p "$HOME/.venvs/decodeatt" "$HOME/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/decodeatt"
export UV_CACHE_DIR="$HOME/.cache/uv"
( =打开安装了uv的conda环境)
uv sync --frozen --python 3.11
```

验证环境：

```bash
"$HOME/.venvs/decodeatt/bin/python" -c \
  'import torch, transformers; print(torch.__version__, torch.cuda.is_available())'
```

验证成功后运行 smoke test

## 查看主目录中除 work 外的文件和文件夹大小

```bash
find "$HOME" -mindepth 1 -maxdepth 1 ! -name work -exec du -sh {} + 2>/dev/null | sort -h
```

该命令会显示主目录下除 `work` 外所有一级文件和文件夹的大小，并按大小升序排列。
