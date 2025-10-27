import math
import os
import sys
import types

os.environ.setdefault("CLOUD_DATABASE_URL", "sqlite+aiosqlite:///:memory:")

stub_api = types.ModuleType("app.dataservice.sql_api.api")


async def _dummy_fetch_recommend_properties_async(*args, **kwargs):
    return []


def _dummy_fetch_recommend_properties(*args, **kwargs):
    return []


stub_api.fetch_recommend_properties_async = _dummy_fetch_recommend_properties_async
stub_api.fetch_recommend_properties = _dummy_fetch_recommend_properties
sys.modules.setdefault("app.dataservice.sql_api.api", stub_api)

from app.models import EnquiryForm, Property
from app.services.recommendation_service import multi_objective_optimization_ranking


def test_multi_objective_ranking_filters_invalid_properties_and_orders_by_priority():
    enquiry = EnquiryForm(
        device_id="test-device",
        min_monthly_rent=1000,
        max_monthly_rent=2000,
        school_id=1,
        importance_rent=3,
        importance_location=2,
        importance_facility=1,
    )

    property_a = Property(
        property_id=1,
        costScore=0.9,
        commuteScore=0.4,
        neighborhoodScore=0.6,
    )

    property_b = Property(
        property_id=2,
        costScore=0.6,
        commuteScore=0.9,
        neighborhoodScore=0.5,
    )

    property_c = Property(
        property_id=3,
        costScore=0.7,
        commuteScore=0.5,
        neighborhoodScore=0.9,
    )

    invalid_property = Property(
        property_id=99,
        costScore=0.0,
        commuteScore=0.5,
        neighborhoodScore=0.5,
    )

    ranked = multi_objective_optimization_ranking(
        enquiry=enquiry,
        propertyList=[property_a, property_b, property_c, invalid_property],
    )

    assert [p.property_id for p in ranked] == [1, 3, 2]
    assert all(p.property_id != 99 for p in ranked)

    top_property = ranked[0]
    assert math.isclose(top_property.costScore, 1.0)
    assert top_property.commuteScore < 1.0
    assert top_property.neighborhoodScore < 1.0
