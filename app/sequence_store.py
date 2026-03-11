from __future__ import annotations

from collections import defaultdict, deque
from typing import Protocol

import numpy as np


class SequenceStoreBackend(Protocol):
    def add_event(self, entity_id: str, features: list[float]) -> None:
        ...

    def get_sequence(
        self, entity_id: str, require_full_sequence: bool, pad_short_sequences: bool
    ) -> np.ndarray:
        ...


def _validate_mode(require_full_sequence: bool, pad_short_sequences: bool) -> None:
    if require_full_sequence and pad_short_sequences:
        raise ValueError(
            "invalid sequence mode: require_full_sequence and pad_short_sequences "
            "cannot both be true"
        )


def _finalize_sequence(
    events: list[np.ndarray],
    entity_id: str,
    seq_len: int,
    feature_dim: int,
    require_full_sequence: bool,
    pad_short_sequences: bool,
) -> np.ndarray:
    _validate_mode(require_full_sequence, pad_short_sequences)

    if len(events) > seq_len:
        events = events[-seq_len:]

    if len(events) < seq_len and require_full_sequence:
        raise ValueError(
            "insufficient history for notebook-style inference: "
            f"got {len(events)} events, need {seq_len}"
        )

    if len(events) < seq_len and pad_short_sequences:
        pad_count = seq_len - len(events)
        pad = [np.zeros(feature_dim, dtype=np.float32) for _ in range(pad_count)]
        events = pad + events

    if len(events) < seq_len:
        raise ValueError(
            "sequence too short and padding is disabled: "
            f"got {len(events)} events, need {seq_len}"
        )

    return np.stack(events, axis=0).astype(np.float32, copy=False)


class InMemorySequenceStore:
    def __init__(self, seq_len: int, feature_dim: int) -> None:
        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
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
        events = list(self._store.get(entity_id, []))
        return _finalize_sequence(
            events=events,
            entity_id=entity_id,
            seq_len=self.seq_len,
            feature_dim=self.feature_dim,
            require_full_sequence=require_full_sequence,
            pad_short_sequences=pad_short_sequences,
        )


class RedisSequenceStore:
    def __init__(
        self,
        seq_len: int,
        feature_dim: int,
        redis_url: str,
        redis_key_prefix: str = "fraud:seq",
        redis_ttl_seconds: int = 0,
    ) -> None:
        try:
            import redis
        except ImportError as exc:
            raise ValueError(
                "Redis backend requested but 'redis' package is not installed."
            ) from exc

        if not redis_url:
            raise ValueError("REDIS_URL must be set when using Redis backend.")

        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
        self.redis_key_prefix = redis_key_prefix or "fraud:seq"
        self.redis_ttl_seconds = max(0, int(redis_ttl_seconds))
        self._client = redis.Redis.from_url(redis_url, decode_responses=False)

    def _key(self, entity_id: str) -> str:
        return f"{self.redis_key_prefix}:{entity_id}"

    def add_event(self, entity_id: str, features: list[float]) -> None:
        if len(features) != self.feature_dim:
            raise ValueError(
                f"feature length mismatch: got {len(features)}, expected {self.feature_dim}"
            )

        arr = np.asarray(features, dtype=np.float32)
        key = self._key(entity_id)
        payload = arr.tobytes(order="C")

        pipe = self._client.pipeline(transaction=False)
        pipe.rpush(key, payload)
        pipe.ltrim(key, -self.seq_len, -1)
        if self.redis_ttl_seconds > 0:
            pipe.expire(key, self.redis_ttl_seconds)
        pipe.execute()

    def get_sequence(
        self, entity_id: str, require_full_sequence: bool, pad_short_sequences: bool
    ) -> np.ndarray:
        key = self._key(entity_id)
        values = self._client.lrange(key, 0, -1)

        events: list[np.ndarray] = []
        for raw in values:
            arr = np.frombuffer(raw, dtype=np.float32)
            if arr.shape != (self.feature_dim,):
                raise ValueError(
                    "corrupted Redis sequence payload: "
                    f"got {arr.shape}, expected ({self.feature_dim},)"
                )
            events.append(arr)

        return _finalize_sequence(
            events=events,
            entity_id=entity_id,
            seq_len=self.seq_len,
            feature_dim=self.feature_dim,
            require_full_sequence=require_full_sequence,
            pad_short_sequences=pad_short_sequences,
        )


# Backward-compatible alias for existing imports.
class SequenceStore(InMemorySequenceStore):
    pass


def create_sequence_store(
    backend: str,
    seq_len: int,
    feature_dim: int,
    redis_url: str | None = None,
    redis_key_prefix: str = "fraud:seq",
    redis_ttl_seconds: int = 0,
) -> SequenceStoreBackend:
    normalized = (backend or "memory").strip().lower()
    if normalized in {"memory", "in-memory", "inmemory"}:
        return InMemorySequenceStore(seq_len=seq_len, feature_dim=feature_dim)
    if normalized == "redis":
        return RedisSequenceStore(
            seq_len=seq_len,
            feature_dim=feature_dim,
            redis_url=redis_url or "",
            redis_key_prefix=redis_key_prefix,
            redis_ttl_seconds=redis_ttl_seconds,
        )
    raise ValueError(
        f"unsupported sequence store backend '{backend}'. Use 'memory' or 'redis'."
    )
