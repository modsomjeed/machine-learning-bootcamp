"""Two June columns derived from one row.

Notebook 03 asks you to write the same function. Notebooks 04 and 05 call this
copy so the saved model still has the columns after a kernel restart.
"""

import pandas as pd


def add_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["orders_per_week"] = out["orders_90d"] / 13
    out["recency_over_45"] = (out["days_since_last_order"] >= 45).astype(int)
    return out
