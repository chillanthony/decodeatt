"""Offline batched R-KV throughput + accuracy driver for patched vLLM.

Runs the patched vLLM (v0.25.1 + ``vllm/rkv``) the same way the reference
harness does: submit *all* N prompts to ``LLM.generate`` at once, so vLLM can
continuously batch them all and R-KV's KV-compression advantage shows up as
higher sustained decode throughput.

R-KV on/off is driven purely by env vars (``VLLM_V1_R_KV_BUDGET`` /
``VLLM_V1_R_KV_BUFFER``), exactly like upstream:
  * both > 0  -> R-KV compression active
  * either = 0 -> pure upstream vLLM (Full-KV baseline)

Datasets
--------
``--data aime``  (default)  : AIME_2024 questions, scored as integer answers.
                              This is your research model's real workload.
``--data gsm8k``            : the bundled few-shot GSM8K split, scored as the
                              paper does. Use this to reproduce the reference
                              throughput number directly.

Model
-----
Default ``RKV_MODEL`` (or ``--model``) is ``deepseek-ai/DeepSeek-R1-Distill-Llama-8B``
-- the model your grid already runs. Point it at ``Qwen/Qwen2.5-Math-7B-Instruct``
to reproduce the reference paper's exact model.

Usage
-----
    # R-KV budget=256 buffer=128, 30 AIME problems, single GPU:
    VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
      python eval_offline.py --data aime --n 30 --label rkv_b256_buf128 \
      --out /tmp/rkv_aime_b256.json

    # Full-KV production baseline (prefix caching on):
    python eval_offline.py --data aime --n 30 --label fullkv_production

    # Full-KV "constrained" -- the fair A/B (prefix caching off, like R-KV):
    python eval_offline.py --data aime --n 30 --no-prefix --label fullkv_constrained

    # 8-way data parallelism (each rank its own R-KV replica):
    VLLM_V1_R_KV_BUDGET=256 VLLM_V1_R_KV_BUFFER=128 \
      python eval_offline.py --data aime --n 30 --dp 8
"""

import argparse
import json
import os
import re
import time

# The compaction counter is read from each worker via collective_rpc with a
# small closure; vLLM gates callable RPCs behind this flag. Set before importing.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from vllm import LLM, SamplingParams  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_GSM8K = os.path.join(_HERE, "gsm8k_fewshot.jsonl")


def _load_aime(n: int) -> list[str]:
    """Load ``n`` AIME_2024 questions (via the parent repo's dataset loader)."""
    import sys

    # Allow importing kvbench.datasets from the repo root even when this script
    # is run from elsewhere.
    repo_root = os.path.dirname(os.path.dirname(_HERE))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from kvbench.datasets import load_problems

    return [str(p["question"]) for p in load_problems("aime", n)]


def _load_gsm8k(n: int) -> list[str]:
    with open(_GSM8K) as f:
        rows = [json.loads(line) for line in f]
    rows = rows[:n]
    prompts = []
    for r in rows:
        # The bundled few-shot file stores the whole prompt in
        # ``request.messages``; pass the message list so vLLM applies the chat
        # template exactly as the reference did.
        prompts.append(r["request"]["messages"])
    return prompts


def _extract_gold(answer: str):
    m = re.search(r"####\s*([\-0-9\.,/]+)", answer)
    return m.group(1).replace(",", "").rstrip(".") if m else None


_PATS = [
    r"\\boxed\{([^{}]*)\}",
    r"[Tt]he final answer is[:\s]*\$?\\?\(?\$?([\-0-9\.,/]+)",
    r"[Tt]he answer is[:\s]*\$?([\-0-9\.,/]+)",
]


def _extract_pred(text: str):
    for pat in _PATS:
        found = re.findall(pat, text)
        if found:
            return found[-1].replace(",", "").replace("$", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def _to_num(x):
    if x is None:
        return None
    x = x.strip()
    try:
        if "/" in x:
            a, b = x.split("/")
            return float(a) / float(b)
        return float(x)
    except Exception:
        return None


def _read_compactions(llm: LLM):
    def _get(self):
        mr = getattr(self, "model_runner", None)
        comp = getattr(mr, "rkv_compactor", None)
        return int(getattr(comp, "_n_compactions", 0)) if comp is not None else 0

    try:
        counts = llm.collective_rpc(_get)
        return max(counts) if counts else 0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="aime", choices=["aime", "gsm8k"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=32768,
                    help="decode budget per request")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--label", default="")
    ap.add_argument("--model", default=os.environ.get(
        "RKV_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"))
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--dp", type=int, default=1, help="data_parallel_size")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    ap.add_argument("--no-prefix", action="store_true",
                    help="disable prefix caching (fair Full-KV A/B; R-KV forces this)")
    ap.add_argument("--eager", action="store_true",
                    help="force eager (default: PIECEWISE cudagraph when R-KV on)")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="generate exactly --max-tokens per request (KV stress test)")
    ap.add_argument("--stats", action="store_true",
                    help="enable vLLM engine stat logging (achieved concurrency)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    budget = int(os.environ.get("VLLM_V1_R_KV_BUDGET", "0"))
    buffer = int(os.environ.get("VLLM_V1_R_KV_BUFFER", "0"))
    rkv_on = budget > 0 and buffer > 0
    label = args.label or (f"rkv_b{budget}_buf{buffer}" if rkv_on else "fullkv")

    if args.data == "aime":
        prompts = _load_aime(args.n)
        golds = [None] * len(prompts)  # answers not needed; score as numeric
    else:
        prompts = _load_gsm8k(args.n)
        with open(_GSM8K) as f:
            rows = [json.loads(line) for line in f][: args.n]
        golds = [_to_num(_extract_gold(r["answer"])) for r in rows]
    n = len(prompts)

    kw = {}
    if args.no_prefix:
        kw["enable_prefix_caching"] = False
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        enforce_eager=args.eager,
        gpu_memory_utilization=args.mem_frac,
        max_model_len=args.max_model_len,
        disable_log_stats=not args.stats,
        seed=0,
        block_size=16,
        **kw,
    )
    tok = llm.get_tokenizer()
    in_toks = sum(len(tok(p).input_ids if isinstance(p, str)
                   else tok.apply_chat_template(p, tokenize=True))
                  for p in prompts)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
        stop=None if args.ignore_eos else ["\nProblem"],
    )

    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0

    correct = 0
    scored = 0
    out_toks = 0
    for i, (d, o) in enumerate(zip(prompts, outs)):
        out_toks += len(o.outputs[0].token_ids)
        if golds[i] is None:
            continue
        scored += 1
        g = golds[i]
        p = _to_num(_extract_pred(o.outputs[0].text))
        ok = g is not None and p is not None and abs(g - p) < 1e-4
        correct += int(ok)
        if i < 5:
            print(f"[{i}] gold={g} pred={p} ok={ok} ntok={len(o.outputs[0].token_ids)}")

    compactions = _read_compactions(llm) if rkv_on else 0
    res = {
        "label": label,
        "budget": budget,
        "buffer": buffer,
        "model": args.model,
        "data": args.data,
        "n": n,
        "tp": args.tp,
        "dp": args.dp,
        "max_tokens": args.max_tokens,
        "accuracy": round(correct / scored, 4) if scored else None,
        "scored": scored,
        "correct": correct,
        "wall_s": round(dt, 1),
        "out_tokens": out_toks,
        "in_tokens": in_toks,
        "decode_tok_s": round(out_toks / dt, 1),
        "total_tok_s": round((out_toks + in_toks) / dt, 1),
        "avg_gen_len": round(out_toks / n, 1),
        "compactions": compactions,
        "per_gpu_decode_tok_s": round(out_toks / dt / args.dp, 1),
    }
    print(
        f"\n=== {label} ===\n"
        f"accuracy    : {correct}/{scored} = {(correct / scored if scored else 0):.3f}\n"
        f"avg_tokens  : {out_toks / n:.0f}\n"
        f"wall_time   : {dt:.1f}s\n"
        f"decode_tput : {out_toks / dt:.1f} tok/s (offline batched, {n} prompts in flight"
        + (f", {args.dp} DP replicas -> {out_toks / dt / args.dp:.1f}/GPU" if args.dp > 1 else "") + ")\n"
        f"compactions : {compactions}"
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
