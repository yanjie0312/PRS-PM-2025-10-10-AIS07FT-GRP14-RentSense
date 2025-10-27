"""Utilities to prepare collaborative-filtering training data.

本模块主要完成两个步骤：

1. **初始化异步数据库连接** —— 参照 ``app.dataservice.sql_api.func``
   创建 ``AsyncEngine`` 与 ``AsyncSession``，这样离线脚本能复用线上
   的数据库配置。
2. **执行聚合 SQL** —— 通过 ``AGGREGATION_SQL`` 将推荐结果拆解并按
   ``user_id``、``listing_id`` 求出偏好强度，供协同过滤训练阶段消费。

This module exposes a small async API that can be reused by
training scripts and scheduled jobs. The async engine setup mirrors
``app.dataservice.sql_api.func`` so it works in the same runtime
environment.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, Mapping, Sequence

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.dataservice.sql_api.envconfig import get_database_url_async

# --- Database bootstrap ----------------------------------------------------

DATABASE_URL_ASYNC = get_database_url_async()

async_engine = create_async_engine(
    DATABASE_URL_ASYNC,
    echo=False,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# --- Aggregation query -----------------------------------------------------

AGGREGATION_SQL = text(
    """
    WITH recommendation_items AS (
        SELECT
            e.device_id AS user_id,
            (item ->> 'property_id')::INTEGER AS listing_id,
            (
                COALESCE(e.importance_rent, 0) * COALESCE((item ->> 'costScore')::DOUBLE PRECISION, 0) +
                COALESCE(e.importance_location, 0) * COALESCE((item ->> 'commuteScore')::DOUBLE PRECISION, 0) +
                COALESCE(e.importance_facility, 0) * COALESCE((item ->> 'neighborhoodScore')::DOUBLE PRECISION, 0)
            ) AS strength
        FROM recommendations AS r
        INNER JOIN enquiries AS e ON e.eid = r.eid
        CROSS JOIN LATERAL jsonb_array_elements(r.recommandation_result::jsonb) AS item
        WHERE r.recommandation_result IS NOT NULL
          AND e.device_id IS NOT NULL
          AND item ? 'property_id'
    )
    SELECT
        user_id,
        listing_id,
        AVG(strength) AS strength
    FROM recommendation_items
    WHERE user_id IS NOT NULL
      AND listing_id IS NOT NULL
    GROUP BY user_id, listing_id
    ORDER BY user_id, listing_id
    """
)

# --- Public API ------------------------------------------------------------


async def fetch_user_item_strength(
    *,
    session: AsyncSession | None = None,
) -> Sequence[Mapping[str, float | int | str]]:
    """Return aggregated user-listing strengths.

    Parameters
    ----------
    session:
        Optional existing :class:`AsyncSession`. When provided the caller
        is responsible for session lifecycle management.
    """

    close_session = False
    if session is None:
        session = AsyncSessionLocal()
        close_session = True

    try:
        result = await session.execute(AGGREGATION_SQL)
        return result.mappings().all()
    finally:
        if close_session:
            await session.close()


def rows_to_dataframe(rows: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Convert query rows to a :class:`pandas.DataFrame`.

    The helper normalises the expected columns so downstream training
    code receives a predictable schema. When *rows* is empty an empty
    DataFrame with ``user_id``, ``listing_id`` and ``strength`` columns
    is returned.
    """

    records = list(rows)
    if not records:
        return pd.DataFrame(columns=["user_id", "listing_id", "strength"])

    return pd.DataFrame.from_records(records, columns=["user_id", "listing_id", "strength"])


async def _main() -> None:
    """Simple CLI entry point for manual runs."""

    rows = await fetch_user_item_strength()
    df = rows_to_dataframe(rows)
    print(df.head())


if __name__ == "__main__":
    asyncio.run(_main())
