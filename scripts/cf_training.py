"""Collaborative-filtering preprocessing utility.

The script answers the question "What prepares data for the CF models?"
by orchestrating three responsibilities:

* Connect to the Postgres database specified by ``DATABASE_URL`` or
  ``CLOUD_DATABASE_URL`` and aggregate historical recommendation
  interactions into a tabular format.
* Produce deterministic index mappings for the distinct users and
  property listings so that matrix factorisation libraries (ALS,
  LightFM, etc.) can consume the interactions consistently across
  training runs and deployments.
* Persist the generated ``user2idx`` and ``item2idx`` dictionaries as
  JSON files that the serving layer can load when translating between
  external identifiers and matrix coordinates.

Usage::

    python scripts/cf_training.py --output-dir SystemCode/backend/models/cf

Optional arguments allow custom output directories and overriding the
aggregation SQL used to load interactions. The resulting mapping files
(``user2idx.json`` and ``item2idx.json``) can be reused inside the API
service when serving model predictions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

try:
    from scipy.sparse import coo_matrix
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "scipy is required to build the interaction matrix. "
        "Install it with `pip install scipy`."
    ) from exc

DEFAULT_AGGREGATION_SQL = """
WITH expanded AS (
    SELECT
        e.device_id AS user_id,
        (payload ->> 'property_id')::BIGINT AS property_id,
        COALESCE((payload ->> 'weighted_score')::DOUBLE PRECISION, 1.0) AS weight
    FROM recommendations AS r
    JOIN enquiries AS e ON e.eid = r.eid
    CROSS JOIN LATERAL jsonb_array_elements(
        COALESCE(r.recommandation_result::jsonb, '[]'::jsonb)
    ) AS payload
    WHERE e.device_id IS NOT NULL
)
SELECT
    user_id,
    property_id,
    COUNT(*) AS interaction_count,
    SUM(weight) AS total_weight,
    AVG(weight) AS mean_weight
FROM expanded
WHERE property_id IS NOT NULL
GROUP BY user_id, property_id
ORDER BY interaction_count DESC, total_weight DESC;
"""


def resolve_database_url() -> str:
    """Return the configured database connection string.

    The function checks ``DATABASE_URL`` first followed by
    ``CLOUD_DATABASE_URL``. A ``RuntimeError`` is raised when neither is
    provided.
    """

    for env_key in ("DATABASE_URL", "CLOUD_DATABASE_URL"):
        value = os.getenv(env_key)
        if value:
            return value
    raise RuntimeError(
        "Neither DATABASE_URL nor CLOUD_DATABASE_URL is set. "
        "Please provide a valid SQLAlchemy connection string."
    )


def get_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for the provided URL."""

    return create_engine(database_url, pool_pre_ping=True, future=True)


def load_interactions(engine: Engine, query: str) -> pd.DataFrame:
    """Fetch the aggregated interaction table as a DataFrame."""

    with engine.connect() as connection:
        df = pd.read_sql_query(text(query), connection)
    if df.empty:
        return df

    df = df.dropna(subset=["user_id", "property_id"]).copy()
    df["user_key"] = df["user_id"].astype(str)
    df["item_key"] = df["property_id"].astype(str)

    if "interaction_count" not in df.columns:
        df["interaction_count"] = 1.0

    if "total_weight" not in df.columns:
        df["total_weight"] = df["interaction_count"].astype(float)

    return df


def build_index_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create deterministic user/item index mappings from the dataframe."""

    user_keys = sorted(df["user_key"].unique())
    item_keys = sorted(df["item_key"].unique())

    user2idx = {key: idx for idx, key in enumerate(user_keys)}
    item2idx = {key: idx for idx, key in enumerate(item_keys)}

    return user2idx, item2idx


def build_interaction_matrix(
    df: pd.DataFrame,
    user2idx: Dict[str, int],
    item2idx: Dict[str, int],
    *,
    weight_column: str = "total_weight",
) -> coo_matrix:
    """Construct a COO sparse matrix based on the supplied dataframe."""

    if weight_column not in df.columns:
        raise ValueError(f"Column '{weight_column}' not found in dataframe")

    weights = df[weight_column].fillna(0).astype(float)
    user_indices = df["user_key"].map(user2idx)
    item_indices = df["item_key"].map(item2idx)

    return coo_matrix(
        (weights, (user_indices.to_numpy(), item_indices.to_numpy())),
        shape=(len(user2idx), len(item2idx)),
    )


def save_mappings(user2idx: Dict[str, int], item2idx: Dict[str, int], output_dir: Path) -> None:
    """Persist mapping dictionaries into JSON files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    user_path = output_dir / "user2idx.json"
    item_path = output_dir / "item2idx.json"

    with user_path.open("w", encoding="utf-8") as user_file:
        json.dump(user2idx, user_file, ensure_ascii=False, indent=2)

    with item_path.open("w", encoding="utf-8") as item_file:
        json.dump(item2idx, item_file, ensure_ascii=False, indent=2)

    print(f"Saved user mapping to {user_path}")
    print(f"Saved item mapping to {item_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training utility."""

    parser = argparse.ArgumentParser(description="Collaborative filtering preprocessing")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("SystemCode/backend/models/cf"),
        help="Directory where mapping files will be stored.",
    )
    parser.add_argument(
        "--sql-file",
        type=Path,
        default=None,
        help="Optional path to a .sql file overriding the default aggregation query.",
    )
    parser.add_argument(
        "--weight-column",
        type=str,
        default="total_weight",
        help="Dataframe column to use as interaction strength when building the matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    database_url = resolve_database_url()
    engine = get_engine(database_url)

    if args.sql_file:
        aggregation_sql = args.sql_file.read_text(encoding="utf-8")
    else:
        aggregation_sql = os.getenv("CF_AGG_SQL", DEFAULT_AGGREGATION_SQL)

    interactions = load_interactions(engine, aggregation_sql)
    engine.dispose()

    if interactions.empty:
        print("No interactions were retrieved. Nothing to do.")
        return

    user2idx, item2idx = build_index_mappings(interactions)
    interaction_matrix = build_interaction_matrix(
        interactions, user2idx, item2idx, weight_column=args.weight_column
    )

    print(
        "Constructed sparse matrix with shape %s and %d non-zero entries"
        % (interaction_matrix.shape, interaction_matrix.nnz)
    )

    save_mappings(user2idx, item2idx, args.output_dir)

    print("Preprocessing complete. You can now train ALS/LightFM models using the matrix.")


if __name__ == "__main__":
    main()
