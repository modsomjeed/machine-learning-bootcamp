from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Int64, String
from feast.value_type import ValueType

customer = Entity(name="customer", join_keys=["customer_id"], value_type=ValueType.STRING)

customer_source = FileSource(
    name="customer_snapshot",
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp",
)

customer_features = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=0),
    schema=[
        Field(name="orders_90d", dtype=Int64),
        Field(name="plan", dtype=String),
    ],
    online=True,
    offline=False,
    source=customer_source,
)
