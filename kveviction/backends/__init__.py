"""Execution backends for generation-time KV management.

The first backend preserves the current HuggingFace calling convention.  A
serving backend can implement the same semantic prefill/decode boundary
without making policies depend on a particular model or cache class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ExecutionBackend(Protocol):
    """Minimal model execution boundary used by the reference runner."""

    def prefill(self, input_ids, **kwargs: Any):
        ...

    def decode(self, input_ids, **kwargs: Any):
        ...

    def admit(self, request_id: str, **kwargs: Any) -> None:
        ...

    def release(self, request_id: str) -> None:
        ...


@dataclass
class TransformersExecutionBackend:
    """Adapter for a callable HuggingFace-style causal language model."""

    model: Any

    def __post_init__(self):
        self._active_requests: set[str] = set()

    def prefill(self, input_ids, **kwargs: Any):
        return self.model(input_ids=input_ids, **kwargs)

    def decode(self, input_ids, **kwargs: Any):
        return self.model(input_ids=input_ids, **kwargs)

    def admit(self, request_id: str, **kwargs: Any) -> None:
        self._active_requests.add(str(request_id))

    def release(self, request_id: str) -> None:
        self._active_requests.discard(str(request_id))

    @property
    def active_requests(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_requests))


__all__ = ["ExecutionBackend", "TransformersExecutionBackend"]
