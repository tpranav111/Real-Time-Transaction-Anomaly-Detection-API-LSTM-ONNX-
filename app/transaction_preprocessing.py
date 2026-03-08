from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd


# Notebook-compatible helper functions used by reference mapper artifacts.
def timeEncoder(X: pd.DataFrame) -> pd.DataFrame:  # noqa: N802
    X_hm = X["Time"].str.split(":", expand=True)
    d = pd.to_datetime(
        dict(
            year=X["Year"],
            month=X["Month"],
            day=X["Day"],
            hour=X_hm[0],
            minute=X_hm[1],
        )
    ).astype(int)
    return pd.DataFrame(d)


def amtEncoder(X: pd.Series) -> pd.DataFrame:  # noqa: N802
    amt = (
        X.apply(lambda x: str(x)[1:])
        .astype(float)
        .map(lambda value: max(1, value))
        .map(math.log)
    )
    return pd.DataFrame(amt)


def decimalEncoder(X: pd.Series, length: int = 5) -> pd.DataFrame:  # noqa: N802
    dnew = pd.DataFrame()
    values = np.asarray(X).reshape(-1)
    work = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).astype(int)
    for i in range(length):
        dnew[i] = np.mod(work, 10)
        work = np.floor_divide(work, 10)
    return dnew


def fraudEncoder(X: pd.Series) -> np.ndarray:  # noqa: N802
    return np.where(X == "Yes", 1, 0).astype(int)


def time_encoder(X: pd.DataFrame) -> pd.DataFrame:
    return timeEncoder(X)


def amt_encoder(X: pd.Series) -> pd.DataFrame:
    return amtEncoder(X)


def decimal_encoder(X: pd.Series, length: int = 5) -> pd.DataFrame:
    return decimalEncoder(X, length=length)


def fraud_encoder(X: pd.Series) -> np.ndarray:
    return fraudEncoder(X)


def _register_legacy_pickle_symbols() -> None:
    # Training script pickles may reference functions under __main__.
    # Register equivalent symbols so joblib can resolve them at load time.
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    setattr(main_mod, "time_encoder", time_encoder)
    setattr(main_mod, "amt_encoder", amt_encoder)
    setattr(main_mod, "decimal_encoder", decimal_encoder)
    setattr(main_mod, "fraud_encoder", fraud_encoder)
    setattr(main_mod, "timeEncoder", timeEncoder)
    setattr(main_mod, "amtEncoder", amtEncoder)
    setattr(main_mod, "decimalEncoder", decimalEncoder)
    setattr(main_mod, "fraudEncoder", fraudEncoder)


@dataclass
class MapperResult:
    features: list[float]
    source: str


class TransactionFeatureMapper:
    def __init__(self, mapper_path: str | None) -> None:
        self.mapper_path = mapper_path
        if mapper_path:
            _register_legacy_pickle_symbols()
            self.mapper = joblib.load(mapper_path)
        else:
            self.mapper = None

    @property
    def enabled(self) -> bool:
        return self.mapper is not None

    def transform_raw_transaction(
        self,
        raw_transaction: dict[str, Any],
        event_time: datetime,
        expected_feature_dim: int,
    ) -> MapperResult:
        if self.mapper is None:
            raise ValueError(
                "raw_transaction received but MAPPER_PATH is not configured"
            )

        row = self._to_mapper_row(raw_transaction, event_time)
        mapped = self.mapper.transform(pd.DataFrame([row]))
        vector = self._extract_feature_vector(mapped, expected_feature_dim)
        return MapperResult(features=vector.tolist(), source="fitted_mapper")

    @staticmethod
    def _extract_feature_vector(mapped: Any, expected_feature_dim: int) -> np.ndarray:
        if isinstance(mapped, pd.DataFrame):
            if "Is Fraud?" in mapped.columns:
                mapped = mapped.drop(["Is Fraud?"], axis=1)
            values = mapped.to_numpy(dtype=np.float32)
            vector = values.reshape(-1)
        else:
            values = np.asarray(mapped, dtype=np.float32)
            vector = values.reshape(-1)
            if vector.size == expected_feature_dim + 1:
                vector = vector[1:]

        if vector.size != expected_feature_dim:
            raise ValueError(
                "mapped feature size mismatch: "
                f"got {vector.size}, expected {expected_feature_dim}"
            )
        return vector.astype(np.float32)

    @staticmethod
    def _to_mapper_row(raw_transaction: dict[str, Any], event_time: datetime) -> dict[str, Any]:
        def nan_if_missing(value: Any) -> Any:
            if value is None:
                return np.nan
            if isinstance(value, str) and value.strip() == "":
                return np.nan
            return value

        tx_time = raw_transaction.get("time")
        if tx_time is None:
            tx_time = event_time.strftime("%H:%M")

        zip_code = raw_transaction.get("zip_code")
        if zip_code is None or zip_code == "":
            zip_code = np.nan
        else:
            # Keep Zip as object/string to avoid SimpleImputer float->int casting issues.
            zip_code = str(zip_code)

        amount = raw_transaction.get("amount")
        if amount is None:
            raise ValueError("raw_transaction.amount is required for mapper preprocessing")
        amount_str = str(amount)
        if not amount_str.startswith("$"):
            amount_str = f"${amount_str}"

        return {
            "Is Fraud?": "No",
            "Merchant State": nan_if_missing(raw_transaction.get("merchant_state")),
            "Zip": zip_code,
            "Merchant Name": str(raw_transaction.get("merchant_name")),
            "Merchant City": raw_transaction.get("merchant_city"),
            "MCC": raw_transaction.get("mcc"),
            "Use Chip": nan_if_missing(raw_transaction.get("use_chip")),
            "Errors?": nan_if_missing(raw_transaction.get("errors")),
            "Year": raw_transaction.get("year", event_time.year),
            "Month": raw_transaction.get("month", event_time.month),
            "Day": raw_transaction.get("day", event_time.day),
            "Time": tx_time,
            "Amount": amount_str,
        }

