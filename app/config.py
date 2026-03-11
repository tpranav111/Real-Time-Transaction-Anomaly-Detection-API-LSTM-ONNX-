from __future__ import annotations

import os


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings:
    model_path: str = os.getenv("MODEL_PATH", "models/fraud_gru.onnx")
    model_version: str = os.getenv("MODEL_VERSION", "gru-v1")
    mapper_path: str | None = os.getenv("MAPPER_PATH")
    default_seq_len: int = int(os.getenv("DEFAULT_SEQ_LEN", "7"))
    default_feature_dim: int = int(os.getenv("DEFAULT_FEATURE_DIM", "220"))
    model_input_layout: str = os.getenv("MODEL_INPUT_LAYOUT", "auto")
    require_full_sequence: bool = _parse_bool(
        os.getenv("REQUIRE_FULL_SEQUENCE"), default=True
    )
    pad_short_sequences: bool = _parse_bool(
        os.getenv("PAD_SHORT_SEQUENCES"), default=False
    )
    approve_threshold: float = float(os.getenv("APPROVE_THRESHOLD", "0.25"))
    challenge_threshold: float = float(os.getenv("CHALLENGE_THRESHOLD", "0.60"))
    decline_threshold: float = float(os.getenv("DECLINE_THRESHOLD", "0.85"))

    # Sequence state backend configuration
    sequence_store_backend: str = os.getenv("SEQUENCE_STORE_BACKEND", "memory")
    redis_url: str | None = os.getenv("REDIS_URL")
    redis_key_prefix: str = os.getenv("REDIS_KEY_PREFIX", "fraud:seq")
    redis_ttl_seconds: int = int(os.getenv("REDIS_TTL_SECONDS", "0"))


settings = Settings()
