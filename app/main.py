from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.transaction_preprocessing import TransactionFeatureMapper
from app.model_runtime import FraudModel
from app.policy import decide
from app.schemas import ModelMetaResponse, ScoreRequest, ScoreResponse
from app.sequence_store import SequenceStore

app = FastAPI(title="Fraud Scoring API", version="1.0.0")

model: FraudModel | None = None
sequence_store: SequenceStore | None = None
feature_mapper: TransactionFeatureMapper | None = None


@app.on_event("startup")
def startup() -> None:
    global model, sequence_store, feature_mapper
    model = FraudModel(settings.model_path)
    sequence_store = SequenceStore(
        seq_len=model.signature.expected_seq_len,
        feature_dim=model.signature.expected_feature_dim,
    )
    feature_mapper = TransactionFeatureMapper(settings.mapper_path)


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict[str, str]:
    if model is None or sequence_store is None:
        raise HTTPException(status_code=503, detail="model not initialized")
    return {"status": "ready"}


@app.get("/v1/model/meta", response_model=ModelMetaResponse)
def model_meta() -> ModelMetaResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="model not initialized")

    sig = model.signature
    return ModelMetaResponse(
        model_version=settings.model_version,
        input_name=sig.input_name,
        output_name=sig.output_name,
        input_layout=sig.input_layout,
        expected_seq_len=sig.expected_seq_len,
        expected_feature_dim=sig.expected_feature_dim,
        expected_batch_dim=sig.expected_batch_dim,
    )


@app.post("/v1/score", response_model=ScoreResponse)
def score(payload: ScoreRequest) -> ScoreResponse:
    if model is None or sequence_store is None:
        raise HTTPException(status_code=503, detail="model not initialized")

    start = time.perf_counter()
    try:
        if payload.features is not None:
            event_features = payload.features
        elif payload.raw_transaction is not None and feature_mapper is not None:
            mapped = feature_mapper.transform_raw_transaction(
                raw_transaction=payload.raw_transaction.model_dump(),
                event_time=payload.event_time,
                expected_feature_dim=model.signature.expected_feature_dim,
            )
            event_features = mapped.features
        else:
            raise ValueError("unable to derive feature vector from request")

        sequence_store.add_event(payload.entity_id, event_features)
        seq = sequence_store.get_sequence(
            payload.entity_id,
            require_full_sequence=settings.require_full_sequence,
            pad_short_sequences=settings.pad_short_sequences,
        )
        fraud_score = model.predict_score(seq)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

    decision, reasons = decide(fraud_score, amount=payload.amount)
    latency_ms = (time.perf_counter() - start) * 1000

    return ScoreResponse(
        transaction_id=payload.transaction_id,
        entity_id=payload.entity_id,
        fraud_score=fraud_score,
        decision=decision,
        reasons=reasons,
        model_version=settings.model_version,
        latency_ms=round(latency_ms, 2),
    )

