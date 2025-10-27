"""Utility script for training collaborative-filtering recommender models.

This module reads an interaction dataset, builds a sparse user-item matrix and
trains either an Alternating Least Squares (ALS) model provided by the
``implicit`` package or a ``LightFM`` hybrid model. The resulting model and the
index mapping dictionaries are persisted to disk so that the inference service
can reuse them without recomputing the factors.

Typical usage from the project root::

    python SystemCode/backend/app/dataservice/DataScript/train_recommender.py \
        --input-path data/interactions.csv \
        --output-dir SystemCode/backend/app/dataservice/models \
        --model-type als --rank 128 --regularization 0.08 --epochs 30

Both models expose the common hyper-parameters that the team requested—``rank``
(a.k.a. latent factors), ``regularization`` and ``epochs``—so they can be tuned
without modifying the source code. Additional LightFM specific parameters such
as ``loss`` and ``learning-rate`` are also configurable through the CLI.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from scipy import sparse


@dataclass(slots=True)
class TrainingArtifacts:
    """Container describing files produced by a training run."""

    model_path: Path
    user_mapping_path: Path
    item_mapping_path: Path
    metrics_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ALS or LightFM recommender models from interaction data."
    )

    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the interaction CSV file. Must contain user and item columns.",
    )
    parser.add_argument(
        "--user-col",
        default="user_id",
        help="Column name representing the user identifier.",
    )
    parser.add_argument(
        "--item-col",
        default="item_id",
        help="Column name representing the item identifier.",
    )
    parser.add_argument(
        "--rating-col",
        default=None,
        help="Optional column with explicit feedback scores. Defaults to implicit ones.",
    )
    parser.add_argument(
        "--model-type",
        choices=("als", "lightfm"),
        default="als",
        help="Choose between implicit ALS or LightFM models.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="Number of latent factors used by the model.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=0.1,
        help="L2 regularization strength for the model (user/item alpha for LightFM).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs/iterations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate used by LightFM (ignored for ALS).",
    )
    parser.add_argument(
        "--loss",
        choices=("warp", "bpr", "warp-kos", "logistic"),
        default="warp",
        help="LightFM loss function (ignored for ALS).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Worker threads used for model fitting and evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of interactions reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible train/test splitting and model initialisation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cut-off value for precision@k evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="SystemCode/backend/app/dataservice/models",
        help="Directory where the trained model and metadata should be stored.",
    )
    parser.add_argument(
        "--model-filename",
        default="als_model.pkl",
        help="Filename used for the persisted model (defaults to als_model.pkl).",
    )
    parser.add_argument(
        "--gcs-bucket",
        default=None,
        help="Optional Google Cloud Storage bucket name for syncing artefacts.",
    )
    parser.add_argument(
        "--gcs-credentials",
        default=None,
        help="Service account JSON used when uploading to GCS (optional).",
    )

    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_interactions(
    *,
    input_path: Path,
    user_col: str,
    item_col: str,
    rating_col: str | None,
) -> Tuple[pd.DataFrame, str | None]:
    df = pd.read_csv(input_path)
    missing_columns = {
        col
        for col in (user_col, item_col)
        if col not in df.columns
    }
    if missing_columns:
        raise ValueError(
            f"Missing required columns {sorted(missing_columns)} in {input_path}."
        )

    df = df.dropna(subset=[user_col, item_col])
    if df.empty:
        raise ValueError("The interaction file contains no valid user-item pairs.")

    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)

    if rating_col and rating_col in df.columns:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce").fillna(0.0)
        df = df[df[rating_col] > 0]
    else:
        rating_col = None

    logging.info(
        "Loaded %d interactions involving %d users and %d items.",
        len(df),
        df[user_col].nunique(),
        df[item_col].nunique(),
    )

    cleaned_df = df[[user_col, item_col]] if rating_col is None else df[[user_col, item_col, rating_col]]
    return cleaned_df, rating_col


def build_mappings(
    df: pd.DataFrame, user_col: str, item_col: str
) -> Tuple[Dict[str, int], Dict[str, int]]:
    user2idx = {user_id: idx for idx, user_id in enumerate(df[user_col].unique())}
    item2idx = {item_id: idx for idx, item_id in enumerate(df[item_col].unique())}

    logging.info("Constructed %d user and %d item indices.", len(user2idx), len(item2idx))
    return user2idx, item2idx


def build_sparse_matrix(
    df: pd.DataFrame,
    user2idx: Mapping[str, int],
    item2idx: Mapping[str, int],
    user_col: str,
    item_col: str,
    rating_col: str | None,
) -> sparse.csr_matrix:
    rows = df[user_col].map(user2idx.get).to_numpy()
    cols = df[item_col].map(item2idx.get).to_numpy()
    if rating_col:
        data = df[rating_col].to_numpy(dtype=np.float32)
    else:
        data = np.ones_like(rows, dtype=np.float32)

    interaction_matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(len(user2idx), len(item2idx)), dtype=np.float32
    ).tocsr()

    logging.info(
        "Built sparse matrix with shape %s and %d stored interactions.",
        interaction_matrix.shape,
        interaction_matrix.nnz,
    )

    return interaction_matrix


def split_train_test(
    interactions: sparse.csr_matrix,
    test_size: float,
    random_state: int,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be within the interval (0, 1).")

    coo = interactions.tocoo()
    rng = np.random.default_rng(random_state)
    test_mask = rng.random(coo.nnz) < test_size

    train = sparse.coo_matrix(
        (coo.data[~test_mask], (coo.row[~test_mask], coo.col[~test_mask])),
        shape=interactions.shape,
    ).tocsr()

    test = sparse.coo_matrix(
        (coo.data[test_mask], (coo.row[test_mask], coo.col[test_mask])),
        shape=interactions.shape,
    ).tocsr()

    if test.nnz == 0:
        logging.warning(
            "Test split ended up empty; reducing the hold-out fraction to keep at least one interaction."
        )
        min_test = max(1, int(np.ceil(coo.nnz * 0.05)))
        test_indices = rng.choice(coo.nnz, size=min_test, replace=False)
        mask = np.zeros(coo.nnz, dtype=bool)
        mask[test_indices] = True
        train = sparse.coo_matrix(
            (coo.data[~mask], (coo.row[~mask], coo.col[~mask])),
            shape=interactions.shape,
        ).tocsr()
        test = sparse.coo_matrix(
            (coo.data[mask], (coo.row[mask], coo.col[mask])),
            shape=interactions.shape,
        ).tocsr()

    logging.info(
        "Train interactions: %d, Test interactions: %d",
        train.nnz,
        test.nnz,
    )
    return train, test


def train_lightfm(
    train: sparse.csr_matrix,
    test: sparse.csr_matrix,
    *,
    rank: int,
    regularization: float,
    epochs: int,
    learning_rate: float,
    loss: str,
    num_threads: int,
    k: int,
) -> Tuple[object, Dict[str, float]]:
    try:
        from lightfm import LightFM
        from lightfm.evaluation import auc_score, precision_at_k
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "LightFM is not installed. Install the 'lightfm' extra requirements to train this model."
        ) from exc

    model = LightFM(
        no_components=rank,
        loss=loss,
        learning_rate=learning_rate,
        item_alpha=regularization,
        user_alpha=regularization,
        random_state=42,
    )

    logging.info(
        "Training LightFM model with %d components, loss=%s for %d epochs.",
        rank,
        loss,
        epochs,
    )
    model.fit(train, epochs=epochs, num_threads=num_threads)

    metrics: Dict[str, float] = {}
    if test.nnz > 0:
        precision = precision_at_k(
            model,
            test,
            train_interactions=train,
            k=k,
            num_threads=num_threads,
        ).mean()
        auc = auc_score(
            model,
            test,
            train_interactions=train,
            num_threads=num_threads,
        ).mean()
        metrics["precision_at_k"] = float(precision)
        metrics["auc_score"] = float(auc)
        logging.info("LightFM precision@%d = %.4f | AUC = %.4f", k, precision, auc)
    else:
        logging.warning("Skipping evaluation because the test split is empty.")

    return model, metrics


def train_als(
    train: sparse.csr_matrix,
    test: sparse.csr_matrix,
    *,
    rank: int,
    regularization: float,
    epochs: int,
    num_threads: int,
    k: int,
) -> Tuple[object, Dict[str, float]]:
    try:
        from implicit.als import AlternatingLeastSquares
        from implicit.evaluation import mean_average_precision_at_k, precision_at_k
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "implicit is not installed. Install the 'implicit' extra requirements to train this model."
        ) from exc

    model = AlternatingLeastSquares(
        factors=rank,
        regularization=regularization,
        iterations=epochs,
        num_threads=num_threads,
    )

    logging.info(
        "Training ALS model with %d factors for %d iterations.", rank, epochs
    )
    model.fit(train.T.tocsr())

    metrics: Dict[str, float] = {}
    if test.nnz > 0:
        # implicit expects item-user matrices for evaluation functions
        item_user_train = train.T.tocsr()
        item_user_test = test.T.tocsr()

        precision = precision_at_k(
            model,
            item_user_train,
            item_user_test,
            K=k,
        ).mean()
        map_k = mean_average_precision_at_k(
            model,
            item_user_train,
            item_user_test,
            K=k,
        ).mean()
        metrics["precision_at_k"] = float(precision)
        metrics["map_at_k"] = float(map_k)
        logging.info("ALS precision@%d = %.4f | MAP@%d = %.4f", k, precision, k, map_k)
    else:
        logging.warning("Skipping evaluation because the test split is empty.")

    return model, metrics


def save_json(data: Mapping[str, object], path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logging.info("Wrote %s", path)


def sync_to_gcs(files: Iterable[Path], bucket_name: str, credentials: str | None) -> None:
    try:
        from google.cloud import storage
    except Exception as exc:  # pragma: no cover - optional dependency
        logging.error("google-cloud-storage is not available: %s", exc)
        return

    try:
        if credentials:
            client = storage.Client.from_service_account_json(credentials)
        else:
            client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as exc:  # pragma: no cover - runtime configuration issue
        logging.error("Failed to initialise GCS client: %s", exc)
        return

    for file_path in files:
        try:
            blob = bucket.blob(file_path.name)
            blob.upload_from_filename(file_path)
            logging.info(
                "Uploaded %s to gs://%s/%s", file_path, bucket_name, blob.name
            )
        except Exception as exc:  # pragma: no cover - runtime configuration issue
            logging.error("Failed to upload %s: %s", file_path, exc)


def persist_artifacts(
    *,
    model: object,
    output_dir: Path,
    model_filename: str,
    user2idx: Mapping[str, int],
    item2idx: Mapping[str, int],
    metrics: Mapping[str, float],
    config: Mapping[str, object],
) -> TrainingArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / model_filename
    dump(model, model_path)
    logging.info("Persisted model to %s", model_path)

    user_mapping_path = output_dir / "user2idx.json"
    item_mapping_path = output_dir / "item2idx.json"
    metrics_path = output_dir / "metrics.json"

    save_json({k: int(v) for k, v in user2idx.items()}, user_mapping_path)
    save_json({k: int(v) for k, v in item2idx.items()}, item_mapping_path)

    save_json(
        {
            "metrics": metrics,
            "config": config,
        },
        metrics_path,
    )

    return TrainingArtifacts(
        model_path=model_path,
        user_mapping_path=user_mapping_path,
        item_mapping_path=item_mapping_path,
        metrics_path=metrics_path,
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    df, rating_col = load_interactions(
        input_path=input_path,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
    )
    user2idx, item2idx = build_mappings(df, args.user_col, args.item_col)
    interactions = build_sparse_matrix(
        df,
        user2idx=user2idx,
        item2idx=item2idx,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=rating_col,
    )
    train, test = split_train_test(
        interactions, test_size=args.test_size, random_state=args.random_state
    )

    if args.model_type == "lightfm":
        model, metrics = train_lightfm(
            train,
            test,
            rank=args.rank,
            regularization=args.regularization,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            loss=args.loss,
            num_threads=args.num_threads,
            k=args.k,
        )
    else:
        model, metrics = train_als(
            train,
            test,
            rank=args.rank,
            regularization=args.regularization,
            epochs=args.epochs,
            num_threads=args.num_threads,
            k=args.k,
        )

    config = {
        "model_type": args.model_type,
        "rank": args.rank,
        "regularization": args.regularization,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "loss": args.loss,
        "num_threads": args.num_threads,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "k": args.k,
        "input_path": str(input_path),
        "rating_col": rating_col,
    }

    artifacts = persist_artifacts(
        model=model,
        output_dir=output_dir,
        model_filename=args.model_filename,
        user2idx=user2idx,
        item2idx=item2idx,
        metrics=metrics,
        config=config,
    )

    if args.gcs_bucket:
        logging.info(
            "Synchronising artefacts to bucket %s", args.gcs_bucket
        )
        sync_to_gcs(
            (
                artifacts.model_path,
                artifacts.user_mapping_path,
                artifacts.item_mapping_path,
                artifacts.metrics_path,
            ),
            bucket_name=args.gcs_bucket,
            credentials=args.gcs_credentials,
        )

    logging.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
