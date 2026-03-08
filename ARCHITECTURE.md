# Real-World Financial Fraud Detection Architecture (GRU-Centered, FastAPI Inference)

## 1) Design Goal
- Detect fraud during payment authorization with low latency and high recall.
- Use a GRU sequence model inspired by reference `ai-on-z-fraud-detection` approach:
  - recurrent model over transaction history
  - ONNX model artifact for serving
  - sequence-based tensor contract

## 2) Core Architecture (Online Path)
1. Channel ingress
- Card-present / card-not-present transaction request enters `Payment Gateway`.
- Gateway publishes normalized event to `Kafka` topic `txn.auth.requested`.

2. Real-time enrichment
- `Feature Enricher` joins:
  - customer profile
  - merchant and device intelligence
  - geo/IP/risk lists
  - historical aggregates (velocity, burst, spend deltas)

3. Sequence builder (critical for GRU)
- `Sequence Service` maintains last `N` events per entity (card/account/device), with strict event ordering.
- Emits `N x F` tensor candidate for model input.
- For reference-style GRU alignment, preserve the model's exact ONNX input signature at runtime.

4. Inference (FastAPI)
- `Fraud Scoring API` (FastAPI + ONNX Runtime):
  - validates request schema
  - performs sequence padding/truncation
  - runs GRU ONNX inference
  - computes calibrated fraud probability

5. Decision engine
- `Risk Policy Engine` fuses:
  - ML score
  - hard rules (e.g., sanction lists, impossible travel, AML flags)
  - contextual controls (MFA step-up, soft decline, manual review)
- Returns final action: `APPROVE`, `CHALLENGE`, `DECLINE`, or `REVIEW`.

6. Response and audit
- Decision returned to payment switch within SLA budget.
- Full feature snapshot + model version + decision reason persisted to immutable audit store.

## 3) Control Plane and MLOps
1. Offline training pipeline
- Sources: authorization logs, disputes/chargebacks, confirmed fraud labels.
- Label delay handling:
  - temporal joins
  - delayed-positive relabeling
  - survival-window aware evaluation
- Trains GRU and challenger models; exports ONNX.

2. Model registry and promotion
- Every model version includes:
  - ONNX artifact
  - feature schema hash
  - calibration artifact
  - threshold profile by segment
- Promotion path: `dev -> staging -> shadow -> canary -> production`.

3. Monitoring and feedback
- Real-time:
  - p50/p95/p99 latency
  - error rate
  - score distribution drift
  - feature null/shape anomalies
- Delayed:
  - fraud capture rate
  - false positive rate
  - approval rate impact
  - cost per prevented fraud dollar

## 4) Data and Feature Architecture
1. Dual feature stores
- `Online Feature Store` (low-latency reads for inference).
- `Offline Feature Store` (training parity and reproducibility).

2. Feature governance
- versioned features, freshness SLAs, lineage, and backfill controls.
- strict schema contracts to prevent training-serving skew.

3. Sequence state management
- short-lived online state in Redis/Aerospike (TTL by use case).
- deterministic ordering key: event_time + sequence_number.

## 5) Latency Budget (Authorization Use Case)
- Target end-to-end: `<= 80 ms` (typical issuer budget may vary).
- Suggested split:
  - enrichment: 20-30 ms
  - sequence fetch/build: 5-10 ms
  - FastAPI model inference: 10-20 ms
  - policy fusion + response marshalling: 5-10 ms
  - network overhead: remaining buffer

## 6) Reliability and Security
- Active-active deployment across zones/regions.
- Idempotency key on scoring request to avoid duplicate decisions.
- Graceful degradation:
  - if model unavailable -> rules-only fallback profile.
- Security:
  - mTLS between services
  - encryption in transit and at rest
  - PCI/PII tokenization
  - signed model artifacts and SBOM checks.

## 7) FastAPI Inference Interface (Recommended)
1. `POST /v1/score`
- Input: entity id, event timestamp, current transaction features, optional metadata.
- Output:
  - `fraud_score` (0-1)
  - `decision`
  - `reasons`
  - `model_version`
  - `latency_ms`

2. `POST /v1/score/batch`
- Async/batch processing for non-blocking channels and retrospective scoring.

3. `GET /health/live` and `GET /health/ready`
- Liveness and readiness for orchestration.

4. `GET /v1/model/meta`
- Model input signature, active version, sequence/feature dimensions.

## 8) Practical GRU Alignment Notes
- upstream repo documents:
  - 2-layer GRU/LSTM design for fraud detection
  - ONNX export path for serving
  - sequence-driven input tensor
- The LSTM notebook specifically uses:
  - `seq_length = 7` (6 prior + current event)
  - transformed feature width of `220`
  - time-major training/eval tensors
  - ONNX export spec example `(7, 16, 220)`
- In production, lock your FastAPI preprocessor to the exact ONNX input schema in the deployed artifact (including layout assumptions), and reject non-conformant payloads to avoid silent score corruption.

## 9) Evolution Roadmap
1. Phase 1
- GRU + rules in synchronous authorization path.

2. Phase 2
- Add graph-based risk features (entity-link patterns).

3. Phase 3
- Multi-model ensemble:
  - GRU (behavioral sequence)
  - GBDT (tabular risk)
  - graph model (ring/collusion detection)
- Use a calibrated meta-model for final score fusion.

