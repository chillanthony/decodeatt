"""Teacher-forced attention and KV transition signals for fixed FullKV traces."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from kvbench.diagnostics.gen_traces import _atomic_torch_save, _load_config, _pick
from kvbench.models import load_causal_lm


def _distributed_context() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"invalid distributed context rank={rank}, world_size={world_size}")
    return rank, world_size, local_rank


def _cache_layers(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if isinstance(cache, tuple):
        return list(cache)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"unsupported cache type: {type(cache).__name__}")


def _last_kv(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (keys[..., -1, :].detach().float().clone(), values[..., -1, :].detach().float().clone())
        for keys, values in _cache_layers(cache)
    ]


def attention_entropy_by_layer(attentions) -> torch.Tensor:
    """Mean head entropy for the current query at every layer, in nats."""
    values = []
    for attention in attentions:
        probabilities = attention[0, :, -1, :].detach().float().clamp_min(1e-30)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        values.append(-(probabilities * probabilities.log()).sum(dim=-1).mean())
    if not values:
        raise RuntimeError("model returned no attentions; load it with attn_implementation='eager'")
    return torch.stack(values)


def kv_delta_by_layer(
    previous: list[tuple[torch.Tensor, torch.Tensor]],
    current: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Mean cosine distance of adjacent K and V representations per layer."""
    if len(previous) != len(current):
        raise ValueError(f"cache layer mismatch: {len(previous)} != {len(current)}")
    values = []
    for (previous_key, previous_value), (current_key, current_value) in zip(previous, current):
        key_similarity = torch.nn.functional.cosine_similarity(
            previous_key, current_key, dim=-1, eps=1e-8
        ).mean()
        value_similarity = torch.nn.functional.cosine_similarity(
            previous_value, current_value, dim=-1, eps=1e-8
        ).mean()
        values.append(1.0 - 0.5 * (key_similarity + value_similarity))
    return torch.stack(values)


@torch.no_grad()
def extract_trace_signals(model, trace: dict, *, progress_every: int = 256) -> dict:
    """Replay one generated trajectory through an uncompressed cache."""
    device = next(model.parameters()).device
    prompt_ids = trace["prompt_ids"].reshape(1, -1).to(device)
    generated_ids = trace["gen_ids"].reshape(-1).to(device)
    prompt_length = int(prompt_ids.shape[1])

    output = model(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        use_cache=True,
        output_attentions=False,
    )
    cache = output.past_key_values
    next_logits = output.logits[:, -1, :]
    previous_kv = _last_kv(cache)

    attention_rows = []
    kv_rows = []
    replay_nll = []
    for token_index, token_id in enumerate(generated_ids):
        replay_nll.append(
            -torch.log_softmax(next_logits.float(), dim=-1)[0, int(token_id)].detach().cpu()
        )
        absolute_position = prompt_length + token_index
        output = model(
            input_ids=token_id.reshape(1, 1),
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
            attention_mask=torch.ones(
                1, absolute_position + 1, dtype=torch.long, device=device
            ),
            position_ids=torch.tensor([[absolute_position]], device=device),
            cache_position=torch.tensor([absolute_position], device=device),
        )
        cache = output.past_key_values
        next_logits = output.logits[:, -1, :]
        current_kv = _last_kv(cache)
        attention_rows.append(attention_entropy_by_layer(output.attentions).cpu())
        kv_rows.append(kv_delta_by_layer(previous_kv, current_kv).cpu())
        previous_kv = current_kv
        if progress_every > 0 and (token_index + 1) % progress_every == 0:
            print(
                f"  {trace['id']}: replayed {token_index + 1}/{generated_ids.numel()} tokens",
                flush=True,
            )

    num_layers = len(_cache_layers(cache))
    if attention_rows:
        attention_by_layer = torch.stack(attention_rows).float()
        kv_by_layer = torch.stack(kv_rows).float()
        replay_fullkv_nll = torch.stack(replay_nll).float()
    else:
        attention_by_layer = torch.empty(0, num_layers, dtype=torch.float32)
        kv_by_layer = torch.empty(0, num_layers, dtype=torch.float32)
        replay_fullkv_nll = torch.empty(0, dtype=torch.float32)

    reference_nll = trace.get("fullkv_nll")
    nll_max_abs_error = None
    if reference_nll is not None and reference_nll.numel() == replay_fullkv_nll.numel():
        nll_max_abs_error = float(
            (reference_nll.float() - replay_fullkv_nll).abs().max().item()
        ) if replay_fullkv_nll.numel() else 0.0

    return {
        "schema_version": 1,
        "id": trace["id"],
        "model": trace["model"],
        "source_trace": f"{trace['id']}.pt",
        "prompt_length": prompt_length,
        "num_tokens": int(generated_ids.numel()),
        "alignment": {
            "attention_entropy": "attention produced while processing gen_ids[token_index]",
            "kv_delta": "cosine distance from the preceding token representation",
        },
        "attention_entropy": attention_by_layer.mean(dim=1),
        "attention_entropy_by_layer": attention_by_layer,
        "kv_delta": kv_by_layer.mean(dim=1),
        "kv_delta_by_layer": kv_by_layer,
        "replay_fullkv_nll": replay_fullkv_nll,
        "nll_max_abs_error": nll_max_abs_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/configs/experiments/tideprobe_step1.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trace-dir", default=None)
    parser.add_argument("--signals-dir", default=None)
    parser.add_argument("--progress-every", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    rank, world_size, local_rank = _distributed_context()
    model_name = _pick(args.model, config, ("model", "name"))
    dtype = _pick(args.dtype, config, ("model", "dtype"), "bfloat16")
    trace_dir = Path(
        _pick(args.trace_dir, config, ("diagnostics", "trace_dir"), "/home/ma-user/work/bucket-wulan-green/chenyanbo/trace")
    )
    signals_dir = Path(
        _pick(
            args.signals_dir,
            config,
            ("diagnostics", "signals_dir"),
            "/home/ma-user/work/bucket-wulan-green/chenyanbo/decodeatt/runs/tideprobe_step1/signals",
        )
    )
    trace_paths = sorted(trace_dir.glob("*.pt"))
    if not trace_paths:
        raise SystemExit(f"no trace files found in {trace_dir}")
    assigned = trace_paths[rank::world_size]
    signals_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[rank {rank}/{world_size}] model={model_name} assigned={len(assigned)}/{len(trace_paths)} "
        f"input={trace_dir} output={signals_dir}",
        flush=True,
    )
    if not assigned:
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_map: str | dict = {"": f"cuda:{local_rank}"}
    else:
        device_map = "cpu"
    model, _ = load_causal_lm(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    for local_index, trace_path in enumerate(assigned, start=1):
        output_path = signals_dir / trace_path.name
        if output_path.exists() and not args.overwrite:
            print(f"[rank {rank}] skip existing {output_path}", flush=True)
            continue
        trace = torch.load(trace_path, map_location="cpu", weights_only=False)
        if trace.get("model") != model_name and Path(str(trace.get("model"))).name != Path(model_name).name:
            raise ValueError(
                f"trace/model mismatch for {trace_path}: {trace.get('model')} != {model_name}"
            )
        print(
            f"[rank {rank} {local_index}/{len(assigned)}] extracting {trace['id']} "
            f"({trace['gen_ids'].numel()} tokens)",
            flush=True,
        )
        signals = extract_trace_signals(model, trace, progress_every=args.progress_every)
        _atomic_torch_save(signals, output_path)
        print(f"[rank {rank}] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
