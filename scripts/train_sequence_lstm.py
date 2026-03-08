from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelBinarizer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn_pandas import DataFrameMapper


def export_onnx_with_tf_function(
    model: tf.keras.Model,
    onnx_path: Path,
    batch_size: int,
    seq_length: int,
    input_size: int,
) -> None:
    signature = [tf.TensorSpec((batch_size, seq_length, input_size), tf.float32, name="input")]

    @tf.function(input_signature=signature)
    def serving_fn(inputs):
        return model(inputs, training=False)

    tf2onnx.convert.from_function(
        serving_fn,
        input_signature=signature,
        output_path=str(onnx_path),
    )

    if not onnx_path.exists() or onnx_path.stat().st_size <= 0:
        raise RuntimeError(f"ONNX export failed: file not created at {onnx_path}")


def time_encoder(X: pd.DataFrame) -> pd.DataFrame:
    X_hm = X["Time"].astype(str).str.split(":", expand=True)
    d = pd.to_datetime(
        dict(
            year=X["Year"],
            month=X["Month"],
            day=X["Day"],
            hour=X_hm[0],
            minute=X_hm[1],
        )
    ).astype("int64")
    return pd.DataFrame(d)


def amt_encoder(X: pd.Series) -> pd.DataFrame:
    amt = (
        X.apply(lambda x: str(x)[1:])
        .astype(float)
        .map(lambda value: max(1, value))
        .map(math.log)
    )
    return pd.DataFrame(amt)


def decimal_encoder(X: pd.Series, length: int = 5) -> pd.DataFrame:
    dnew = pd.DataFrame()
    values = np.asarray(X).reshape(-1)
    work = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).astype(int)
    for i in range(length):
        dnew[i] = np.mod(work, 10)
        work = np.floor_divide(work, 10)
    return dnew


def fraud_encoder(X: pd.Series) -> np.ndarray:
    return np.where(X == "Yes", 1, 0).astype(int)


def build_mapper() -> DataFrameMapper:
    return DataFrameMapper(
        [
            ("Is Fraud?", FunctionTransformer(fraud_encoder)),
            (
                ["Merchant State"],
                [
                    SimpleImputer(strategy="constant"),
                    FunctionTransformer(np.ravel),
                    LabelEncoder(),
                    FunctionTransformer(decimal_encoder),
                    OneHotEncoder(),
                ],
            ),
            (
                ["Zip"],
                [
                    SimpleImputer(strategy="constant"),
                    FunctionTransformer(np.ravel),
                    FunctionTransformer(decimal_encoder),
                    OneHotEncoder(),
                ],
            ),
            (
                "Merchant Name",
                [
                    LabelEncoder(),
                    FunctionTransformer(decimal_encoder),
                    OneHotEncoder(),
                ],
            ),
            (
                "Merchant City",
                [
                    LabelEncoder(),
                    FunctionTransformer(decimal_encoder),
                    OneHotEncoder(),
                ],
            ),
            (
                "MCC",
                [LabelEncoder(), FunctionTransformer(decimal_encoder), OneHotEncoder()],
            ),
            (["Use Chip"], [SimpleImputer(strategy="constant"), LabelBinarizer()]),
            (["Errors?"], [SimpleImputer(strategy="constant"), LabelBinarizer()]),
            (
                ["Year", "Month", "Day", "Time"],
                [FunctionTransformer(time_encoder), MinMaxScaler()],
            ),
            ("Amount", [FunctionTransformer(amt_encoder), MinMaxScaler()]),
        ],
        input_df=True,
        df_out=True,
    )


def prepare_dataframe(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["Merchant Name"] = df["Merchant Name"].astype(str)
    df.sort_values(by=["User", "Card"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def split_indices(df: pd.DataFrame, seq_length: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = df[["User", "Card"]].drop_duplicates()
    first_indices = np.array(first.index)

    drop_list = np.concatenate([np.arange(x, x + seq_length - 1) for x in first_indices])
    index_list = np.setdiff1d(df.index.values, drop_list)

    total = index_list.shape[0]
    train_length = total // 2
    validate_length = (total - train_length) * 3 // 5

    rng = np.random.default_rng(seed)
    train_indices = rng.choice(index_list, train_length, replace=False)
    tv_list = np.setdiff1d(index_list, train_indices)
    validate_indices = rng.choice(tv_list, validate_length, replace=False)
    test_indices = np.setdiff1d(tv_list, validate_indices)
    return train_indices, validate_indices, test_indices


def gen_training_batch(
    df: pd.DataFrame,
    mapper: DataFrameMapper,
    index_list: np.ndarray,
    batch_size: int,
    seq_length: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    train_df = df.loc[index_list]
    non_fraud_indices = train_df[train_df["Is Fraud?"] == "No"].index.values
    fraud_indices = train_df[train_df["Is Fraud?"] == "Yes"].index.values

    if fraud_indices.size == 0:
        raise ValueError("training split has no fraud samples")

    fraud_size = fraud_indices.shape[0]

    while True:
        replace_flag = non_fraud_indices.shape[0] < fraud_size
        sampled_non_fraud = rng.choice(
            non_fraud_indices, fraud_size, replace=replace_flag
        )
        indices = np.concatenate((fraud_indices, sampled_non_fraud))
        rng.shuffle(indices)

        rows = indices.shape[0]
        index_array = np.zeros((rows, seq_length), dtype=int)
        for i in range(seq_length):
            index_array[:, i] = indices + 1 - seq_length + i

        full_df = mapper.transform(df.loc[index_array.flatten()])
        target_buffer = (
            full_df["Is Fraud?"].to_numpy(dtype=np.float32).reshape(rows, seq_length, 1)
        )
        data_buffer = (
            full_df.drop(["Is Fraud?"], axis=1)
            .to_numpy(dtype=np.float32)
            .reshape(rows, seq_length, -1)
        )

        batch_ptr = 0
        while (batch_ptr + batch_size) <= rows:
            data = data_buffer[batch_ptr : batch_ptr + batch_size]
            targets = target_buffer[batch_ptr : batch_ptr + batch_size]
            batch_ptr += batch_size

            yield data, targets


def gen_eval_batch(
    df: pd.DataFrame,
    mapper: DataFrameMapper,
    indices: np.ndarray,
    batch_size: int,
    seq_length: int,
):
    rows = indices.shape[0]
    index_array = np.zeros((rows, seq_length), dtype=int)
    for i in range(seq_length):
        index_array[:, i] = indices + 1 - seq_length + i

    count = 0
    while count + batch_size <= rows:
        full_df = mapper.transform(df.loc[index_array[count : count + batch_size].flatten()])
        data = (
            full_df.drop(["Is Fraud?"], axis=1)
            .to_numpy(dtype=np.float32)
            .reshape(batch_size, seq_length, -1)
        )
        targets = (
            full_df["Is Fraud?"].to_numpy(dtype=np.float32).reshape(batch_size, seq_length, 1)
        )
        count += batch_size
        yield data, targets


class TP(tf.keras.metrics.TruePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):  # type: ignore[override]
        super().update_state(y_true[:, -1, :], y_pred[:, -1, :], sample_weight)


class FP(tf.keras.metrics.FalsePositives):
    def update_state(self, y_true, y_pred, sample_weight=None):  # type: ignore[override]
        super().update_state(y_true[:, -1, :], y_pred[:, -1, :], sample_weight)


class FN(tf.keras.metrics.FalseNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):  # type: ignore[override]
        super().update_state(y_true[:, -1, :], y_pred[:, -1, :], sample_weight)


class TN(tf.keras.metrics.TrueNegatives):
    def update_state(self, y_true, y_pred, sample_weight=None):  # type: ignore[override]
        super().update_state(y_true[:, -1, :], y_pred[:, -1, :], sample_weight)


def build_model(
    model_type: str,
    input_size: int,
    units: tuple[int, int],
    batch_size: int,
    seq_length: int,
) -> tf.keras.Model:
    _ = batch_size
    rnn_layer = tf.keras.layers.LSTM if model_type == "lstm" else tf.keras.layers.GRU

    # Keras 3 no longer accepts time_major/batch_size in layer constructors.
    # We train in batch-major layout [batch, seq, feature] and export ONNX accordingly.
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape=(seq_length, input_size)),
            rnn_layer(
                units[0],
                return_sequences=True,
            ),
            rnn_layer(units[1], return_sequences=True),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    metrics = [
        "accuracy",
        TP(name="TP"),
        FP(name="FP"),
        FN(name="FN"),
        TN(name="TN"),
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.TrueNegatives(name="tn"),
    ]
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)
    return model


def save_indices(path: Path, values: np.ndarray) -> None:
    np.savetxt(path, values.astype(int), fmt="%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Notebook-aligned training pipeline for reference-style fraud RNN model."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to card_transaction.v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/sequence_lstm_static",
        help="Output directory for mapper/model/onnx artifacts.",
    )
    parser.add_argument(
        "--model-type",
        choices=["lstm", "gru"],
        default="lstm",
        help="RNN cell type for training.",
    )
    parser.add_argument("--seq-length", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--onnx-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--unit-1", type=int, default=200)
    parser.add_argument("--unit-2", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=1111)
    parser.add_argument("--sample-seed", type=int, default=98765)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run validation/test evaluation after training.",
    )
    parser.add_argument(
        "--onnx-name",
        default="fraud_rnn_static.onnx",
        help="ONNX filename written into output-dir.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip saving keras/weights/onnx artifacts and exit after training/eval.",
    )
    parser.add_argument(
        "--skip-keras-save",
        action="store_true",
        help="Skip writing .keras/.weights.h5 artifacts and only export ONNX.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {data_path}")
    df = prepare_dataframe(data_path)
    print(f"Rows loaded: {df.shape[0]}")

    train_indices, validate_indices, test_indices = split_indices(
        df, seq_length=args.seq_length, seed=args.split_seed
    )
    print(
        "Split sizes "
        f"train={train_indices.size} validate={validate_indices.size} test={test_indices.size}"
    )

    mapper = build_mapper()
    print("Fitting mapper on full dataset (notebook parity).")
    mapper.fit(df)

    mapper_path = output_dir / "fitted_mapper.pkl"
    joblib.dump(mapper, mapper_path)
    print(f"Saved mapper: {mapper_path}")

    mapped_sample = mapper.transform(df[:100])
    mapped_size = mapped_sample.shape[-1]
    input_size = mapped_size - 1
    print(f"Mapped size including label={mapped_size} => input_size={input_size}")

    model = build_model(
        model_type=args.model_type,
        input_size=input_size,
        units=(args.unit_1, args.unit_2),
        batch_size=args.batch_size,
        seq_length=args.seq_length,
    )
    model.summary()

    train_gen = gen_training_batch(
        df=df,
        mapper=mapper,
        index_list=train_indices,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        seed=args.sample_seed,
    )
    model.fit(
        train_gen,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        verbose=1,
    )

    if args.evaluate:
        validate_steps = validate_indices.size // args.batch_size
        test_steps = test_indices.size // args.batch_size

        if validate_steps > 0:
            print("Validation evaluation")
            val_gen = gen_eval_batch(
                df=df,
                mapper=mapper,
                indices=validate_indices,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
            )
            model.evaluate(val_gen, steps=validate_steps, verbose=1)

        if test_steps > 0:
            print("Test evaluation")
            test_gen = gen_eval_batch(
                df=df,
                mapper=mapper,
                indices=test_indices,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
            )
            model.evaluate(test_gen, steps=test_steps, verbose=1)

    if args.skip_artifacts:
        print("Skipping artifact writes (--skip-artifacts).")
        return

    onnx_path = output_dir / args.onnx_name
    if args.skip_onnx:
        print("Skipping ONNX export (--skip-onnx).")
    else:
        export_onnx_with_tf_function(
            model=model,
            onnx_path=onnx_path,
            batch_size=args.onnx_batch_size,
            seq_length=args.seq_length,
            input_size=input_size,
        )
        print(f"Saved ONNX model: {onnx_path}")

    keras_path = output_dir / "model.keras"
    weights_path = output_dir / "model.weights.h5"
    if args.skip_keras_save:
        print("Skipping Keras artifact save (--skip-keras-save).")
    else:
        model.save(str(keras_path))
        model.save_weights(str(weights_path))
        print(f"Saved keras model: {keras_path}")
        print(f"Saved weights: {weights_path}")

    save_indices(output_dir / "train.indices", train_indices)
    save_indices(output_dir / "validate.indices", validate_indices)
    save_indices(output_dir / "test.indices", test_indices)

    metadata = {
        "data_path": str(data_path),
        "seq_length": args.seq_length,
        "input_size": int(input_size),
        "batch_size": args.batch_size,
        "onnx_batch_size": args.onnx_batch_size,
        "model_type": args.model_type,
        "units": [args.unit_1, args.unit_2],
        "split_sizes": {
            "train": int(train_indices.size),
            "validate": int(validate_indices.size),
            "test": int(test_indices.size),
        },
        "artifacts": {
            "mapper": str(mapper_path),
            "keras_model": "" if args.skip_keras_save else str(keras_path),
            "keras_weights": "" if args.skip_keras_save else str(weights_path),
            "onnx_model": "" if args.skip_onnx else str(onnx_path),
        },
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="ascii")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

