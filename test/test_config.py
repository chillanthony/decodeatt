"""Run configuration: split-YAML loading, schema validation, CLI precedence and arm parsing."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from kvbench.arms import parse_arm, parse_arms
from kvbench.config import ConfigError, arm_attn_backend, load_run_config, resolve_settings

ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides) -> Namespace:
    base = dict(
        model=None, arms=None, policy_param=[], greedy=False, only_ids="",
        debug_dir=None, debug_topk=None, dataset=None, n=None, max_new=None,
        recent=None, sink=None, evict_every=None, obs_window=None, obs_decay=None,
        temperature=None, top_p=None, seed=None, batch_size=None,
        num_return_sequences=None, seed_offset=None, problem_batch_size=None,
        prompt_bucket_size=None, dtype=None, device_map=None, attn=None,
        fast_attn=None, out=None, log_mode=None,
    )
    base.update(overrides)
    return Namespace(**base)


def _write(tmp_path: Path, name: str, data: dict) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _configs(tmp_path, *, model=None, data=None, strategy=None):
    return (
        _write(tmp_path, "model.yaml", model or {"model": {"name": "m"}}),
        _write(tmp_path, "data.yaml", data or {"data": {"dataset": "sample", "n": 2}}),
        _write(tmp_path, "strategy.yaml", strategy or {"arms": [{"policy": "rkv", "budget": 64}]}),
    )


@pytest.mark.parametrize("data", sorted((ROOT / "config/datasets").glob("*.yaml")), ids=lambda p: p.stem)
@pytest.mark.parametrize("strategy", sorted((ROOT / "config/strategies").glob("*.yaml")), ids=lambda p: p.stem)
def test_repository_configs_validate(data, strategy):
    model = ROOT / "config/models/deepseek_r1_distill_llama8b.yaml"
    cfg = load_run_config(str(model), str(data), str(strategy))
    settings = resolve_settings(cfg, _args())          # works without --arms
    assert settings.model_name == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    assert [arm.policy for arm in settings.arms] == [strategy.stem]


@pytest.mark.parametrize("files, message", [
    (dict(data={"data": {"dataset": "sample", "nn": 2}}), r"unknown key\(s\) \['nn'\] in data"),
    (dict(data={"generation": {"max_new": "lots"}}), r"generation.max_new must be int"),
    (dict(data={"generation": {"seed": True}}), r"generation.seed must be int, got bool"),
    (dict(data={"eviction": {"recent": 4}}), r"unknown section"),
    (dict(model={"model": {"name": "m", "dtyp": "bf16"}}), r"unknown key\(s\) \['dtyp'\] in model"),
    (dict(strategy={"policy_defaults": {"rkvv": {}}}), r"unknown policy 'rkvv'"),
    (dict(strategy={"arms": [{"policy": "rkv", "budget": 64, "bugdet": 1}]}), r"unknown key\(s\) \['bugdet'\] in arms\[0\]"),
    (dict(strategy={"arms": [{"policy": "window", "budget": 64, "anchor_mode": "anchr"}]}), r"anchor_mode"),
    (dict(strategy={"eviction": {"obs_decay": "high"}}), r"eviction.obs_decay must be int/float"),
    (dict(strategy={"arms": [{"policy": "rkv", "budget": 64, "anchor_mode": "anchor", "anchor_frac": 0.1}]}), r"arms\[0\]: anchor modes need a slot-level"),
])
def test_invalid_configs_are_rejected(tmp_path, files, message):
    with pytest.raises(ConfigError, match=message):
        load_run_config(*_configs(tmp_path, **files))


def test_unknown_policy_param_override_is_rejected(tmp_path):
    cfg = load_run_config(*_configs(tmp_path))
    with pytest.raises(ConfigError, match="unknown policy 'rk'"):
        resolve_settings(cfg, _args(policy_param=["rk.alpha=4"]))


def test_cli_overrides_yaml_and_defaults_fill_gaps(tmp_path):
    cfg = load_run_config(*_configs(
        tmp_path,
        data={"data": {"dataset": "aime", "n": 5}, "generation": {"max_new": 900, "do_sample": True}},
        strategy={
            "arms": [{"policy": "rkv", "budget": 64}],
            "policy_defaults": {"rkv": {"alpha": 8, "lambda": 0.1}},
            "eviction": {"recent": 8},
        },
    ))
    settings = resolve_settings(cfg, _args(n=3, greedy=True, policy_param=["rkv.alpha=4"]))
    kw = settings.eval_kwargs
    assert (kw["dataset"], kw["n"], kw["max_new"], kw["recent"]) == ("aime", 3, 900, 8)
    assert kw["sink"] == 8 and kw["evict_every"] == 128      # defaults
    assert kw["do_sample"] is False                          # --greedy wins
    assert settings.arms[0].params == {"alpha": 4, "lambda": 0.1}
    assert settings.out_path == Path("result/eval.json")
    groups, vllm_arms = settings.attn_groups()
    assert list(groups) == ["eager"] and not vllm_arms


def test_vllm_section_is_accepted_in_model_config(tmp_path):
    cfg = load_run_config(*_configs(
        tmp_path,
        model={"model": {"name": "m"}, "vllm": {"gpu_memory_utilization": 0.8}},
        strategy={"arms": [{"policy": "fullkv", "engine": "vllm"}]},
    ))
    settings = resolve_settings(cfg, _args())
    assert settings.cfg["vllm"] == {"gpu_memory_utilization": 0.8}
    assert settings.resolved()["vllm"] == {"gpu_memory_utilization": 0.8}
    _, vllm_arms = settings.attn_groups()
    assert [arm.id for arm in vllm_arms] == ["fullkv"]
    with pytest.raises(ConfigError, match="'vllm' must be a mapping"):
        load_run_config(*_configs(tmp_path, model={"model": {"name": "m"}, "vllm": "fast"}))


def test_split_runs_require_exactly_one_arm(tmp_path):
    cfg = load_run_config(*_configs(tmp_path))
    with pytest.raises(ConfigError, match="exactly one strategy arm"):
        resolve_settings(cfg, _args(arms="rkv@64,fullkv"))


def test_auto_attention_backend_selection():
    for policy in ("fullkv", "streamingllm", "random"):
        assert arm_attn_backend(policy, "auto", "sdpa") == "sdpa"
    for policy in ("window", "snapkv", "h2o", "rkv"):
        assert arm_attn_backend(policy, "auto", "sdpa") == "eager"
    # fast_observe moves only the cache-aware policies (attention recomputed from hidden states).
    for policy in ("rkv", "snapkv", "h2o"):
        assert arm_attn_backend(policy, "auto", "sdpa", fast_observe=True) == "sdpa"
    assert arm_attn_backend("window", "auto", "sdpa", fast_observe=True) == "eager"
    assert arm_attn_backend("rkv", "eager", "sdpa", fast_observe=True) == "eager"


def test_shorthand_arms_attach_policy_defaults_and_overrides():
    arms = parse_arms(
        "rkv@1024,rkv@2048,fullkv",
        policy_defaults={"rkv": {"lambda": 0.1, "alpha": 8}},
        policy_overrides={"rkv": {"lambda": 0.2}},
    )
    assert [arm.name for arm in arms] == ["rkv@1024", "rkv@2048", "fullkv"]
    assert [arm.budget for arm in arms[:2]] == [1024, 2048]
    assert arms[0].params == arms[1].params == {"lambda": 0.2, "alpha": 8}
    assert arms[2].is_full and not arms[0].is_full
    assert parse_arm("fullkv").backend == "fullkv"


def test_structured_arm_parsing_allows_same_policy_with_different_params():
    arms = parse_arms(
        [
            {"id": "rkv_b1024_lam01", "policy": "rkv", "budget": 1024},
            {
                "id": "rkv_b1024_lam02",
                "policy": "rkv",
                "budget": 1024,
                "params": {"lambda": 0.2},
            },
        ],
        policy_defaults={"rkv": {"lambda": 0.1, "alpha": 8}},
    )
    assert [arm.name for arm in arms] == ["rkv_b1024_lam01", "rkv_b1024_lam02"]
    assert [arm.backend for arm in arms] == ["rkv", "rkv"]
    assert arms[0].params == {"lambda": 0.1, "alpha": 8}
    assert arms[1].params == {"lambda": 0.2, "alpha": 8}


def test_structured_arm_selects_execution_engine():
    arms = parse_arms([
        {"id": "fullkv_hf", "policy": "fullkv", "engine": "hf"},
        {"id": "fullkv_vllm", "policy": "fullkv", "engine": "vllm"},
    ])
    assert [arm.engine for arm in arms] == ["hf", "vllm"]


def test_anchor_arm_parsing_and_default_ids():
    arms = parse_arms([
        {"policy": "window", "budget": 512, "anchor_mode": "anchor", "anchor_frac": 0.05},
        {"policy": "window", "budget": 512, "anchor_mode": "lowent", "anchor_frac": 0.05},
        {"policy": "window", "budget": 512, "anchor_mode": "sig", "anchor_frac": 0.1,
         "sig": {"weights": [1, 0, 0, 0], "bias": 0}},
        {"policy": "window", "budget": 512},
    ])
    assert [arm.id for arm in arms] == [
        "window@512+anchor0.05", "window@512+lowent0.05", "window@512+sig0.1", "window@512",
    ]
    assert arms[2].sig == ((1.0, 0.0, 0.0, 0.0), 0.0)
    assert arms[3].anchor_frac == 0.0 and arms[3].sig is None


@pytest.mark.parametrize("item, message", [
    ({"policy": "window", "budget": 64, "anchor_mode": "anchor"}, "requires anchor_frac > 0"),
    ({"policy": "window", "budget": 64, "anchor_frac": 0.1}, "anchor_mode is 'none'"),
    ({"policy": "rkv", "budget": 64, "anchor_mode": "anchor", "anchor_frac": 0.1}, "slot-level"),
    ({"policy": "fullkv", "anchor_mode": "anchor", "anchor_frac": 0.1}, "slot-level"),
    ({"policy": "window", "budget": 64, "anchor_mode": "sig", "anchor_frac": 0.1}, "requires sig"),
    ({"policy": "random", "budget": 64, "anchor_mode": "sig", "anchor_frac": 0.1,
      "sig": {"weights": [1, 0, 0, 0], "bias": 0}}, "never observes"),
    ({"policy": "window", "budget": 64, "anchor_mode": "anchor", "anchor_frac": 0.1,
      "sig": {"weights": [1, 0, 0, 0], "bias": 0}}, "only used with anchor_mode 'sig'"),
])
def test_invalid_anchor_arms_are_rejected(item, message):
    with pytest.raises(ValueError, match=message):
        parse_arms([item])


def test_local_dir_is_used_when_it_exists_else_the_hf_id(tmp_path, monkeypatch):
    from kvbench.config import resolve_model_path

    (tmp_path / "m").mkdir()
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path))
    assert resolve_model_path("org/m", "${MODEL_ROOT}/m") == str(tmp_path / "m")
    assert resolve_model_path("org/m", "${MODEL_ROOT}/missing") == "org/m"
    assert resolve_model_path("org/m", None) == "org/m"
    monkeypatch.delenv("MODEL_ROOT")
    assert resolve_model_path("org/m", "${MODEL_ROOT}/m") == "org/m"   # unset variable

    monkeypatch.setenv("MODEL_ROOT", str(tmp_path))
    cfg = load_run_config(*_configs(tmp_path, model={"model": {"name": "org/m", "local_dir": "${MODEL_ROOT}/m"}}))
    settings = resolve_settings(cfg, _args())
    assert (settings.model_name, settings.model_path) == ("org/m", str(tmp_path / "m"))
    assert settings.eval_kwargs["model_id"] == "org/m"
    assert settings.resolved()["model"]["load_from"] == str(tmp_path / "m")
    # --model replaces the YAML model, including its local_dir.
    settings = resolve_settings(cfg, _args(model="other/x"))
    assert (settings.model_name, settings.model_path) == ("other/x", "other/x")
