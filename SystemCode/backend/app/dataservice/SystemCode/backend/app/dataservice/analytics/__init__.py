"""Convenience re-exports for collaborative-filtering analytics helpers.

Importing the package exposes the public helpers at the package root so that
offline jobs can simply ``from app.dataservice.analytics import
fetch_user_item_strength`` without needing to know the internal module
structure.
"""

from .cf_aggregations import fetch_user_item_strength, rows_to_dataframe

__all__ = (
    "fetch_user_item_strength",
    "rows_to_dataframe",
)
