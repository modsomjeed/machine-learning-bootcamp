"""Two June columns derived from one row.

Notebook 03 asks you to write the same function. The searches in notebooks 04
and 06 call this copy inside every candidate pipeline.
"""

import pandas as pd


def add_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["orders_per_week"] = out["orders_90d"] / 13
    out["recency_over_45"] = (out["days_since_last_order"] >= 45).astype(int)
    return out
