from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class SequenceStore:
    def __init__(self, seq_len: int, feature_dim: int) -> None:
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self._store: dict[str, deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.seq_len)
        )

    def add_event(self, entity_id: str, features: list[float]) -> None:
        if len(features) != self.feature_dim:
            raise ValueError(
                f"feature length mismatch: got {len(features)}, expected {self.feature_dim}"
            )
        self._store[entity_id].append(np.asarray(features, dtype=np.float32))

    def get_sequence(
        self, entity_id: str, require_full_sequence: bool, pad_short_sequences: bool
    ) -> np.ndarray:
        if require_full_sequence and pad_short_sequences:
            raise ValueError(
                "invalid sequence mode: require_full_sequence and pad_short_sequences "
                "cannot both be true"
            )

        events = list(self._store.get(entity_id, []))
        if len(events) > self.seq_len:
            events = events[-self.seq_len :]

        if len(events) < self.seq_len and require_full_sequence:
            raise ValueError(
                "insufficient history for notebook-style inference: "
                f"got {len(events)} events, need {self.seq_len}"
            )

        if len(events) < self.seq_len and pad_short_sequences:
            pad_count = self.seq_len - len(events)
            pad = [np.zeros(self.feature_dim, dtype=np.float32) for _ in range(pad_count)]
            events = pad + events

        if len(events) < self.seq_len:
            raise ValueError(
                "sequence too short and padding is disabled: "
                f"got {len(events)} events, need {self.seq_len}"
            )

        return np.stack(events, axis=0)
