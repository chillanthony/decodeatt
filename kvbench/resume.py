"""Resume partially written result files.

Every result file written by ``run_generation_eval`` carries a ``run_config``
fingerprint (model, attention backend, dataset, arms, generation and eviction
settings) and a ``complete`` flag.  A rerun with the same output path picks up
the problems already in the file and only generates the rest.  A file written
by a different configuration is never extended: mixing two configurations in
one file would silently corrupt the comparison, so loading raises instead.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any


class ResumeMismatch(ValueError):
    """An existing result file was produced by a different run configuration."""


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def run_fingerprint(model, *, arms, policy_params: dict | None, only_ids, **settings) -> dict:
    """Everything that changes the generated records, in JSON-comparable form."""
    config = getattr(model, "config", None)
    try:
        dtype = str(next(model.parameters()).dtype)
    except (AttributeError, StopIteration, TypeError):
        dtype = None
    resolved_arms = []
    for arm in arms:
        entry = asdict(arm)
        entry["params"] = arm.params or (policy_params or {}).get(arm.backend, {})
        resolved_arms.append(entry)
    return _jsonable({
        "model": getattr(config, "_name_or_path", None),
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "dtype": dtype,
        "arms": resolved_arms,
        "only_ids": sorted(only_ids) if only_ids else None,
        **settings,
    })


def _diff(old: dict, new: dict) -> list[str]:
    keys = sorted(set(old) | set(new))
    return [f"{key}: {old.get(key)!r} -> {new.get(key)!r}" for key in keys if old.get(key) != new.get(key)]


def load_resumable(path: Path, fingerprint: dict) -> tuple[list[dict], dict | None]:
    """Return ``(records, complete_payload)`` from an existing result file.

    ``complete_payload`` is the file's payload when the run already finished,
    else None.  A missing file yields ``([], None)``.
    """
    if not path.exists():
        return [], None
    payload = json.loads(path.read_text())
    old = payload.get("run_config")
    if old is None:
        raise ResumeMismatch(
            f"{path} has no run_config (written before resume support or by another tool); "
            "pass --overwrite or choose another --out"
        )
    if old != fingerprint:
        changes = "; ".join(_diff(old, fingerprint))
        raise ResumeMismatch(
            f"{path} was written by a different configuration ({changes}); "
            "pass --overwrite or choose another --out"
        )
    if payload.get("complete"):
        return payload.get("records", []), payload
    return payload.get("records", []), None


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write via a temp file + rename so readers never see a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


__all__ = ["ResumeMismatch", "atomic_write_json", "load_resumable", "run_fingerprint"]
