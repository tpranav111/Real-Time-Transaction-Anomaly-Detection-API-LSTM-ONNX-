from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class RawTransaction(BaseModel):
    merchant_state: str | None = None
    zip_code: int | str | None = None
    merchant_name: str
    merchant_city: str
    mcc: int | str
    use_chip: str | None = None
    errors: str | None = None
    year: int | None = None
    month: int | None = None
    day: int | None = None
    time: str | None = Field(default=None, description="HH:MM")
    amount: str | float


class ScoreRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction id")
    entity_id: str = Field(..., description="Card/account/device key used for sequence state")
    event_time: datetime
    features: list[float] | None = Field(
        default=None, description="Precomputed feature vector for current event"
    )
    raw_transaction: RawTransaction | None = Field(
        default=None,
        description="Raw transaction fields for reference notebook-style mapper preprocessing",
    )
    amount: float | None = None
    channel: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_mode(self) -> "ScoreRequest":
        if self.features is None and self.raw_transaction is None:
            raise ValueError("provide either features or raw_transaction")
        return self


class ScoreResponse(BaseModel):
    transaction_id: str
    entity_id: str
    fraud_score: float
    decision: str
    reasons: list[str]
    model_version: str
    latency_ms: float


class ModelMetaResponse(BaseModel):
    model_version: str
    input_name: str
    output_name: str
    input_layout: str
    expected_seq_len: int
    expected_feature_dim: int
    expected_batch_dim: int | None

