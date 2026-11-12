"""Single-GPU maximum throughput of FullKV and R-KV, following the R-KV paper (Table 1).

Every row generates exactly ``--gen-len`` tokens (EOS is ignored), so there is
no long-tail idling: the number is the decoder's throughput at the largest
batch that fits, not the end-to-end time of an evaluation.  Prompts are the
first ``--num-prompts`` dataset problems, cycled to fill the batch.

Per strategy:
1. probe: run ``--probe-batch`` rows and read the peak memory per row;
2. estimate the largest batch: (GPU memory x ``--mem-fraction`` - weights) / memory per row;
3. measure at that batch; on CUDA out-of-memory retry at 80% until it fits.
``--batch-sizes`` skips 1-2 and measures the given sizes instead.

``--mode full`` generates the whole length in every run.  ``--mode fast`` (default)
generates part of it and extrapolates from per-step wall times, which depend
on the batch and the cache length but not on which tokens are generated:

* eviction strategies: after the first compaction, cache length, memory and
  step time repeat every ``evict_every`` steps.  Probe and measurement run
  ``budget + 4 * evict_every`` steps (so an out-of-memory shows up there), and
  the remaining steps are charged at the mean step time of the last two cycles.
* FullKV: step time grows about linearly with cache length.  The probe still
  runs the full length (memory peaks at the end), the measurement runs
  ``--sample-len`` steps (default a quarter), a line is fitted to the step times
  of its second half and integrated over the remaining steps.  The KV buffers are
  reserved for the full length during the measurement, so a batch whose cache
  does not fit fails there; attention temporaries that grow with length are
  covered only by the full-length probe.

    python script/speedtest.py                                   # fullkv and rkv@1024, 8192 tokens per row
    python script/speedtest.py --mode full                       # no extrapolation (validation)

Strategy settings come from ``config/strategies/<name>.yaml`` as in ``script/eval.py``;
``--budget`` replaces the arm budget and the attention backend follows ``--attn auto``.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kvbench.config import load_run_config, resolve_settings  # noqa: E402
from kvbench.datasets import load_problems  # noqa: E402
from kvbench.evaluator import _arm_generate_kwargs  # noqa: E402
from kvbench.models import build_input_ids, format_prompt, load_causal_lm, pad_input_ids  # noqa: E402
from kveviction import growable_cache  # noqa: E402
from kveviction.backends import TransformersExecutionBackend  # noqa: E402
from kveviction.runner_token import generate_token_evict_batch  # noqa: E402
from kveviction.strategies.token import get_policy  # noqa: E402


class _NoEosTokenizer:
    """Hides EOS so every row runs to ``max_new``; skips decoding the output."""

    def __init__(self, tokenizer):
        self.eos_token_id = None
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    def decode(self, ids, skip_special_tokens=True):
        return ""


class _StepClock(TransformersExecutionBackend):
    """Timestamps every decode call.  The runner syncs with the GPU once per
    step before the next forward, so consecutive stamps are whole step times."""

    def __init__(self, model):
        super().__init__(model)
        self.stamps: list[float] = []

    def decode(self, input_ids, **kwargs):
        self.stamps.append(time.perf_counter())
        return super().decode(input_ids, **kwargs)


def _settings(args, strategy: str):
    cfg = load_run_config(args.model_config, args.data_config, str(ROOT / f"config/strategies/{strategy}.yaml"))
    arm = strategy if strategy == "fullkv" else f"{strategy}@{args.budget}"
    cli = SimpleNamespace(
        model=args.model, arms=arm, policy_param=[], only_ids="", debug_dir=None,
        debug_topk=None, greedy=False, attn=args.attn,
    )
    settings = resolve_settings(cfg, cli)
    groups, _ = settings.attn_groups()
    (attn, arms), = groups.items()
    return settings, attn, arms[0]


def _free():
    gc.collect()
    torch.cuda.empty_cache()


class _Bench:
    def __init__(self, model, tokenizer, prompts, arm, gen, args):
        self.model, self.tokenizer, self.prompts, self.arm, self.gen = model, tokenizer, prompts, arm, gen
        self.gen_len, self.mode = args.gen_len, args.mode
        self.evicts = not get_policy(arm.backend).never_evict
        period = gen["evict_every"]
        self.evict_steps = min(args.gen_len, arm.budget + 4 * period) if self.evicts else args.gen_len
        self.sample_steps = min(args.gen_len, args.sample_len or args.gen_len // 4)

    def _generate(self, size: int, steps: int, reserve: bool = False):
        """Generate ``steps`` tokens per row; with ``reserve`` the KV buffers are
        allocated for the full ``gen_len`` on the first growth, so a batch whose
        full-length cache does not fit fails here instead of near the end."""
        rows = [self.prompts[i % len(self.prompts)] for i in range(size)]
        ids, mask, _ = pad_input_ids(self.tokenizer, rows, self.model.device)
        clock = _StepClock(self.model)
        kwargs = _arm_generate_kwargs(self.arm, {**self.gen, "max_new": steps, "execution_backend": clock})
        grown = growable_cache._grown_capacity
        if reserve:
            # The largest buffer a full run can end with (growth adds up to 1/8 plus a chunk).
            full = int(ids.shape[1]) + self.gen_len
            full = grown(full, full)
            growable_cache._grown_capacity = lambda needed, current: max(grown(needed, current), full)
        try:
            result = generate_token_evict_batch(
                self.model, self.tokenizer, ids, batch_size=size, input_attention_mask=mask,
                seeds=list(range(size)), **kwargs,
            )
        except torch.cuda.OutOfMemoryError:
            result = None
        finally:
            growable_cache._grown_capacity = grown
        return result, [b - a for a, b in zip(clock.stamps, clock.stamps[1:])]

    def run(self, size: int, purpose: str) -> dict | None:
        """One run at ``size`` rows; ``None`` on CUDA out-of-memory.

        ``purpose`` is ``probe`` (memory per row) or ``measure`` (throughput);
        in fast mode the number of generated steps depends on both.
        """
        steps = self.gen_len
        if self.mode == "fast":
            steps = self.evict_steps if self.evicts else (self.gen_len if purpose == "probe" else self.sample_steps)
        result, step_sec = self._generate(size, steps, reserve=not self.evicts and steps < self.gen_len)
        if result is None:
            _free()
            return None
        cand = result["candidates"][0]
        decode = cand["decode_sec"] or 1e-9
        elapsed = result["elapsed_sec"]
        predicted = steps < self.gen_len
        if predicted:
            elapsed += self._remaining_sec(step_sec, steps)
        row = {
            "arm": self.arm.name, "batch": size, "gen_len": self.gen_len, "measured_steps": steps,
            "predicted": predicted, "tokens": size * self.gen_len, "elapsed_sec": elapsed,
            "peak_memory_bytes": result["peak_memory_bytes"] or 0,
            "forward_share": cand["decode_forward_sec"] / decode,
            "observe_share": cand["attention_observation_sec"] / decode,
            "evict_share": cand["eviction_sec_total"] / decode,
            "other_share": cand["other_decode_sec"] / decode,
        }
        row["tokens_per_sec"] = row["tokens"] / elapsed
        row["ms_per_step"] = 1000 * elapsed / self.gen_len
        del result
        _free()
        return row

    def _remaining_sec(self, step_sec: list[float], done: int) -> float:
        """Predicted wall time of steps ``done .. gen_len - 1`` from the measured step times."""
        remaining = self.gen_len - done
        if self.evicts:
            period = self.gen["evict_every"]
            tail = step_sec[-2 * period:]
            return remaining * sum(tail) / len(tail)
        # FullKV: least-squares line t(i) = a + b * i over the second half, integrated over i = done .. gen_len - 1.
        start = len(step_sec) // 2
        xs = list(range(start + 1, len(step_sec) + 1))
        ys = step_sec[start:]
        n, mx, my = len(xs), sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        a = my - b * mx
        return remaining * a + b * (done + self.gen_len - 1) * remaining / 2


def _print(row: dict, tag: str) -> None:
    shares = "/".join(f"{row[k]:.0%}" for k in ("forward_share", "observe_share", "evict_share", "other_share"))
    note = f"  (measured {row['measured_steps']} steps, rest extrapolated)" if row["predicted"] else ""
    print(
        f"{row['arm']:12s} {tag:7s} {row['batch']:6d} {row['tokens_per_sec']:9.0f} {row['ms_per_step']:8.1f} "
        f"{row['peak_memory_bytes'] / 2 ** 30:8.1f}  {shares}{note}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", default=str(ROOT / "config/models/deepseek_r1_distill_llama8b.yaml"))
    parser.add_argument("--data-config", default=str(ROOT / "config/datasets/math500.yaml"))
    parser.add_argument("--model", default=None, help="Override the model name/path.")
    parser.add_argument("--strategies", default="fullkv rkv")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--gen-len", type=int, default=8192)
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--sample-len", type=int, default=0,
                        help="FullKV steps measured in fast mode (default gen_len / 4).")
    parser.add_argument("--probe-batch", type=int, default=8)
    parser.add_argument("--mem-fraction", type=float, default=0.9, help="Share of GPU memory the estimate may fill.")
    parser.add_argument("--max-batch", type=int, default=1024, help="Upper bound on the estimated batch.")
    parser.add_argument("--batch-sizes", default=None, help="Measure these sizes instead of probing for the maximum.")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--attn", default=None, help="Force an attention backend instead of auto.")
    parser.add_argument("--out", default=None, help="Write all rows as JSON here.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("speedtest needs a CUDA device")
    plans = [(s, *_settings(args, s)) for s in args.strategies.split()]
    plans.sort(key=lambda plan: plan[2])  # one model load per attention backend
    gpu = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.mem_get_info()[1]
    print(f"[speedtest] gpu={gpu} memory={total_mem / 2 ** 30:.0f} GiB gen_len={args.gen_len} "
          f"budget={args.budget} mode={args.mode}", flush=True)
    print(f"{'arm':12s} {'run':7s} {'batch':>6s} {'tok/s':>9s} {'ms/step':>8s} {'peak GiB':>8s}  "
          "forward/observe/evict/other", flush=True)

    rows, model, bench, loaded = [], None, None, None
    for strategy, settings, attn, arm in plans:
        if attn != loaded:
            model = bench = None  # drop every reference so the old weights are freed
            _free()
            model, raw_tokenizer = load_causal_lm(
                settings.model_path, dtype=settings.dtype, device_map=settings.device_map,
                attn_implementation=attn,
            )
            weights = torch.cuda.memory_allocated()
            kw = settings.eval_kwargs
            prompts = [
                build_input_ids(raw_tokenizer, format_prompt(p["question"], kw["prompt_template"]), model.device)
                for p in load_problems(kw["dataset"], args.num_prompts)
            ]
            loaded = attn
        kw = settings.eval_kwargs
        gen = dict(
            recent=kw["recent"], sink=kw["sink"], evict_every=kw["evict_every"],
            obs_window=kw["obs_window"], obs_decay=kw["obs_decay"], max_new=args.gen_len,
            do_sample=kw["do_sample"], temperature=kw["temperature"], top_p=kw["top_p"],
            policy_params=kw["policy_params"], execution_backend=None, cache_manager=None,
            sync_timing=False, cache_impl=kw["cache_impl"],
        )
        bench = _Bench(model, _NoEosTokenizer(raw_tokenizer), prompts, arm, gen, args)
        bench._generate(2, 16)  # warm-up: kernels, cuBLAS handles
        _free()

        if args.batch_sizes:
            for size in sorted(int(x) for x in args.batch_sizes.split()):
                row = bench.run(size, "measure")
                if row is None:
                    print(f"{arm.name:12s} {'fixed':7s} {size:6d}  OOM", flush=True)
                    break
                _print(row, "fixed")
                rows.append({**row, "run": "fixed"})
            continue

        probe = bench.run(args.probe_batch, "probe")
        if probe is None:
            print(f"{arm.name:12s} probe   {args.probe_batch:6d}  OOM; lower --probe-batch or --gen-len", flush=True)
            continue
        _print(probe, "probe")
        rows.append({**probe, "run": "probe"})
        per_row = max(probe["peak_memory_bytes"] - weights, 1) / args.probe_batch
        size = int((total_mem * args.mem_fraction - weights) / per_row)
        size = max(args.probe_batch, min(size, args.max_batch))
        print(f"{arm.name:12s} {'':7s} {per_row / 2 ** 20:.0f} MiB per row -> estimated max batch {size}", flush=True)
        while True:
            row = bench.run(size, "measure")
            if row is not None:
                _print(row, "max")
                rows.append({**row, "run": "max"})
                break
            print(f"{arm.name:12s} {'':7s} batch {size} OOM; retrying at {int(size * 0.8)}", flush=True)
            size = int(size * 0.8)
            if size <= args.probe_batch:
                break

    best = {r["arm"]: r for r in rows if r["run"] in ("max", "fixed")}
    full = best.get("fullkv")
    if full:
        for name, r in best.items():
            if name != "fullkv":
                print(f"[speedtest] {name} vs fullkv: batch {r['batch'] / full['batch']:.1f}x, "
                      f"throughput {r['tokens_per_sec'] / full['tokens_per_sec']:.1f}x", flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"gpu": gpu, "args": vars(args), "rows": rows}, indent=1))
        print(f"[speedtest] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
