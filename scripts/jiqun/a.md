# Hugging Face 元数据下载失败排障

最可能的问题是：**远程机器无法直连公网，而脚本现在主动清除了所有代理变量**。之前官方源的代理返回 `504`，只说明代理访问官方源失败，不代表机器可以不通过代理访问镜像。

可以使用下面两条命令进行对比测试。

## 使用当前代理访问

```bash
curl -vL https://hf-mirror.com/api/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## 清除代理后直接访问

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    curl -vkL https://hf-mirror.com/api/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

## 如何判断

- 第一条成功、第二条失败：机器必须使用代理，应删除脚本里的 `env -u`。
- 只有加 `-k` 才成功：属于 SSL 证书问题，需要给 `curl` 增加 `-k`，并给 `wget` 增加 `--no-check-certificate`。
- 两条都返回 `502` 或 `504`：镜像或代理链路异常。
- 返回 `403` 或 `429`：存在访问限制或限流。

结合当前脚本改动，**强制清除代理导致无法联网**是最可能的原因。
