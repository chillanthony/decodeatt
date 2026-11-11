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


# Recorded for reference but not compared: the same model loaded from a local
# directory on one machine and from the HF cache on another is the same run.
_UNCOMPARED = ("model_path",)


def run_fingerprint(
    model, *, arms, policy_params: dict | None, only_ids, model_id: str | None = None, **settings
) -> dict:
    """Everything that changes the generated records, in JSON-comparable form.

    ``model`` is identified by ``model_id`` (the HF id) when given; the path it
    was actually loaded from is kept as ``model_path``.
    """
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
        "model": model_id or getattr(config, "_name_or_path", None),
        "model_path": getattr(config, "_name_or_path", None),
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
    old_cmp = {k: v for k, v in old.items() if k not in _UNCOMPARED}
    new_cmp = {k: v for k, v in fingerprint.items() if k not in _UNCOMPARED}
    if old_cmp != new_cmp:
        changes = "; ".join(_diff(old_cmp, new_cmp))
        raise ResumeMismatch(
            f"{path} was written by a different configuration ({changes}); "
            "pass --overwrite or choose another --out"
        )
    if payload.get("complete"):
        return payload.get("records", []), payload
    return payload.get("records", []), None


def atomic_write_json(path: Path, payload: dict) -> None:
    """Write via a temp file + rename so readers never see a half-written file.

    Object-storage mounts (e.g. an OBS bucket under ModelArts) can refuse the
    rename with EPERM/ENOTSUP; there the file is written in place instead,
    which loses atomicity but keeps the run going.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    try:
        os.replace(tmp, path)
    except OSError:
        path.write_text(text)
        tmp.unlink(missing_ok=True)


__all__ = ["ResumeMismatch", "atomic_write_json", "load_resumable", "run_fingerprint"]
