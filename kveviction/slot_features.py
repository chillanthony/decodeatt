"""Per-slot feature buffers kept alongside the KV cache during decoding.

The runner owns one ``SlotFeatures`` per generation call.  Buffers are
preallocated to ``capacity`` (prompt + max new tokens) and indexed by cache
slot, so appends are O(1) and compaction is a single gather per buffer.
Tensors are ``[capacity]`` for a single request and ``[B, capacity]`` for a
static batch; every operation indexes the trailing (slot) dimension.

Which buffers exist depends on what the policy and anchor mode need:
attention statistics for ``sig`` anchors are only tracked when requested.
"""
from __future__ import annotations

import torch

from kveviction.strategies.base import SelectionContext

# Buffers every request carries.  ``imp``/``cum_imp``/``win_imp`` are the
# decayed-max, cumulative and decayed-sum attention importance.
_BASE = ("slot_pos", "slot_ids", "ent", "imp", "cum_imp", "win_imp")
# Attention-received statistics for signature anchors.
_SIG = ("att_sum", "att_cnt", "att_max")


class SlotFeatures:
    def __init__(self, shape, device, *, track_sig: bool = False, track_valid: bool = False):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.track_sig = track_sig
        self.slot_pos = torch.zeros(shape, dtype=torch.long, device=device)
        self.slot_ids = torch.full(shape, -1, dtype=torch.long, device=device)
        for name in _BASE[2:] + (_SIG if track_sig else ()):
            setattr(self, name, torch.zeros(shape, device=device))
        self.valid = torch.zeros(shape, dtype=torch.bool, device=device) if track_valid else None

    @property
    def _names(self):
        names = _BASE + (_SIG if self.track_sig else ())
        return names + (("valid",) if self.valid is not None else ())

    def append(self, n: int, *, pos, token, entropy, valid=None) -> None:
        """Record a freshly decoded token in slot ``n`` (attention stats start at 0)."""
        self.slot_pos[..., n] = pos
        self.slot_ids[..., n] = token
        self.ent[..., n] = entropy
        self.imp[..., n] = 0.0
        self.cum_imp[..., n] = 0.0
        self.win_imp[..., n] = 0.0
        if self.track_sig:
            self.att_sum[..., n] = 0.0
            self.att_cnt[..., n] = 0.0
            self.att_max[..., n] = 0.0
        if self.valid is not None:
            self.valid[..., n] = valid

    def observe(self, n: int, row: torch.Tensor, decay: float) -> None:
        """Fold one attention-importance row over slots ``[:n]`` into the buffers."""
        self.imp[..., :n] = torch.maximum(self.imp[..., :n] * decay, row)
        self.cum_imp[..., :n] += row
        self.win_imp[..., :n] = self.win_imp[..., :n] * decay + row
        if self.track_sig:
            self.att_sum[..., :n] += row
            self.att_cnt[..., :n] += 1.0
            self.att_max[..., :n] = torch.maximum(self.att_max[..., :n], row)

    def compact(self, idx: torch.Tensor, before_n: int) -> int:
        """Keep slots ``idx`` (``[k]`` or ``[B, k]``) at the front of every buffer."""
        k = int(idx.shape[-1])
        for name in self._names:
            buf = getattr(self, name)
            if buf.ndim == 1:
                buf[:k] = buf[idx]
            else:
                buf[:, :k] = torch.gather(buf[:, :before_n], 1, idx)
        return k

    def sig_stats(self, n: int, row=...):
        """``(cumulative_mean, concentration)`` over slots ``[:n]``, or ``(None, None)``."""
        if not self.track_sig:
            return None, None
        cum = self.att_sum[row, :n] / self.att_cnt[row, :n].clamp(min=1)
        con = self.att_max[row, :n] / (cum + 1e-9)
        return cum, con

    def context(self, n: int, *, budget: int, recent: int, sink: int, params: dict,
                row=..., slots=None, generator=None) -> SelectionContext:
        """Build the selection context for slots ``[:n]`` (optionally one batch row/subset)."""
        def take(buf):
            values = buf[row, :n]
            return values if slots is None else values[slots]

        cum, con = self.sig_stats(n, row)
        if slots is not None and cum is not None:
            cum, con = cum[slots], con[slots]
        return SelectionContext(
            importance=take(self.imp),
            cumulative_attention=take(self.cum_imp),
            window_attention=take(self.win_imp),
            slot_pos=take(self.slot_pos),
            budget=budget,
            recent=recent,
            sink=sink,
            params=params,
            entropy=take(self.ent),
            cumulative_mean=cum,
            concentration=con,
            generator=generator,
        )


__all__ = ["SlotFeatures"]
