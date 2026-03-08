from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnxruntime as ort

from app.config import settings


@dataclass
class ModelSignature:
    input_name: str
    output_name: str
    input_layout: str
    expected_batch_dim: int | None
    expected_seq_len: int
    expected_feature_dim: int
    input_shape: tuple[int | None, int | None, int | None]


class FraudModel:
    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        model_input = self.session.get_inputs()[0]
        model_output = self.session.get_outputs()[0]
        shape = self._normalize_shape(model_input.shape)
        if len(shape) != 3:
            raise ValueError(f"model input rank must be 3, got {shape}")

        input_layout = self._resolve_layout(shape)
        expected_feature_dim = shape[2] or settings.default_feature_dim
        if input_layout == "time_major":
            expected_seq_len = shape[0] or settings.default_seq_len
            expected_batch_dim = shape[1]
        else:
            expected_seq_len = shape[1] or settings.default_seq_len
            expected_batch_dim = shape[0]

        self.signature = ModelSignature(
            input_name=model_input.name,
            output_name=model_output.name,
            input_layout=input_layout,
            expected_batch_dim=expected_batch_dim,
            expected_seq_len=expected_seq_len,
            expected_feature_dim=expected_feature_dim,
            input_shape=shape,
        )

    def predict_score(self, seq: np.ndarray) -> float:
        if seq.shape != (
            self.signature.expected_seq_len,
            self.signature.expected_feature_dim,
        ):
            raise ValueError(
                "sequence shape mismatch: "
                f"got {seq.shape}, expected "
                f"({self.signature.expected_seq_len}, {self.signature.expected_feature_dim})"
            )

        model_input = self._build_model_input(seq)

        output = self.session.run(
            [self.signature.output_name],
            {self.signature.input_name: model_input},
        )[0]

        score = self._extract_score(np.asarray(output))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _normalize_shape(shape: list[Any]) -> tuple[int | None, int | None, int | None]:
        normalized: list[int | None] = []
        for dim in shape:
            normalized.append(dim if isinstance(dim, int) else None)
        return tuple(normalized)  # type: ignore[return-value]

    @staticmethod
    def _resolve_layout(shape: tuple[int | None, int | None, int | None]) -> str:
        forced = settings.model_input_layout.strip().lower()
        if forced in {"time_major", "batch_major"}:
            return forced
        if forced != "auto":
            raise ValueError("MODEL_INPUT_LAYOUT must be one of auto|time_major|batch_major")

        # Reference notebook ONNX contract uses (seq, batch, feature), e.g. (7, 16, 220).
        if shape[0] == settings.default_seq_len:
            return "time_major"
        if shape[1] == settings.default_seq_len:
            return "batch_major"
        if shape[0] is not None and shape[0] <= 16 and shape[1] not in {1, None}:
            return "time_major"
        return "batch_major"

    def _build_model_input(self, seq: np.ndarray) -> np.ndarray:
        batch_dim = self.signature.expected_batch_dim or 1
        if self.signature.input_layout == "time_major":
            # [seq, batch, feature] as used in reference notebook.
            tensor = np.repeat(seq[:, np.newaxis, :], batch_dim, axis=1)
        else:
            # [batch, seq, feature]
            tensor = np.repeat(seq[np.newaxis, :, :], batch_dim, axis=0)
        return tensor.astype(np.float32)

    def _extract_score(self, output: np.ndarray) -> float:
        if output.ndim == 3:
            if self.signature.input_layout == "time_major":
                return float(output[-1, 0, 0])
            return float(output[0, -1, 0])
        if output.ndim == 2:
            return float(output[0, 0])
        if output.ndim == 1:
            return float(output[0])
        return float(output.reshape(-1)[0])

