"""Stable result and timing contracts for all execution backends."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimingSink:
    enabled: bool = True
    values: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        if self.enabled:
            self.values[name] = self.values.get(name, 0.0) + float(seconds)


@dataclass
class BackendResult:
    text: str
    gen_ids: list[int]
    elapsed_sec: float
    prefill_sec: float
    decode_sec: float
    batch_tokens_per_sec: float
    final_cache_len: int
    n_evict: int = 0
    metrics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "text": self.text, "gen_ids": self.gen_ids,
            "elapsed_sec": self.elapsed_sec, "prefill_sec": self.prefill_sec,
            "decode_sec": self.decode_sec, "batch_tokens_per_sec": self.batch_tokens_per_sec,
            "final_cache_len": self.final_cache_len, "n_evict": self.n_evict,
            **self.metrics,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "BackendResult":
        known = {field for field in cls.__dataclass_fields__ if field != "metrics"}
        payload = {key: value[key] for key in known if key in value}
        payload.setdefault("metrics", {key: item for key, item in value.items() if key not in known})
        return cls(**payload)


__all__ = ["BackendResult", "TimingSink"]
