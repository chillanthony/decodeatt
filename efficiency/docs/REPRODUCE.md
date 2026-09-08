# Reproducing / regenerating the R-KV patch

The wiring patch [`patch/rkv-vllm-0.25.1.patch`](../patch/rkv-vllm-0.25.1.patch)
is generated against a pinned upstream vLLM commit. Anyone can rebuild the
patched tree, verify the patch applies cleanly, or regenerate it after editing
the wiring.

## Pinned reference

| | |
| --- | --- |
| Repo | `https://github.com/vllm-project/vllm.git` |
| Tag | `v0.25.1` |
| Commit | `752a3a504485790a2e8491cacbb35c137339ad34` |

## Build the patched tree

```bash
bash efficiency/scripts/build.sh
```

By default the script builds in `/home/ma-user/work/decodeatt-vllm-src` and
creates `/home/ma-user/.venvs/rkv-fast`; remove the existing build tree before
rebuilding. Both are outside this repository and are never committed.

## Verify the patch applies cleanly (non-destructive)

```bash
cd /home/ma-user/work/decodeatt-vllm-src
git stash                                   # park any local edits
git apply --check /path/to/decodeatt/efficiency/patch/rkv-vllm-0.25.1.patch
git stash pop
```

`git apply --check` exits 0 when the patch applies cleanly to the pristine
v0.25.1 tree.

## Regenerate the patch after editing the wiring

The source of truth for the *algorithm* is `rkv/`; the source of truth for the
*wiring* is the patch. If you change the wiring, edit the files inside
`vllm-src/` and regenerate:

```bash
cd /home/ma-user/work/decodeatt-vllm-src
# ... make wiring edits to the 13 tracked files ...
git diff > /path/to/decodeatt/efficiency/patch/rkv-vllm-0.25.1.patch
```

Do **not** hand-edit the patch file. Do **not** rely on edits inside the external
build tree persisting; they are invisible to this repository. Always fold
wiring changes back into the patch.

## The 13 wired files

```
vllm/envs.py
vllm/config/vllm.py
vllm/v1/request.py
vllm/v1/core/sched/output.py
vllm/v1/core/sched/scheduler.py
vllm/v1/outputs.py
vllm/v1/worker/gpu_input_batch.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/attention/backend.py
vllm/v1/attention/backends/flash_attn.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/single_type_kv_cache_manager.py
```
