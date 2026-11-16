"""Train, register, and serve a churn model from files in this folder."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")
os.environ.setdefault("MLFLOW_LOG_UV_FILES", "false")

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "fresh_and_good_customers.csv"
FEATURES = ROOT / "data" / "features.parquet"
REFERENCE = ROOT / "data" / "reference.parquet"
LIVE = ROOT / "data" / "live.parquet"
FEAST_PARQUET = ROOT / "feature_repo" / "data" / "customer_features.parquet"
MLRUNS = ROOT / "mlruns"
FEATURE_REPO = ROOT / "feature_repo"

INT_FEATURES = (
    "tenure_days",
    "orders_90d",
    "days_since_last_order",
    "app_sessions_90d",
    "support_tickets_90d",
)
FLOAT_FEATURES = (
    "spend_thb_90d",
    "avg_basket_thb_90d",
    "promo_order_share_90d",
)
CATEGORICAL = ("district", "plan", "primary_aisle")
NUMERIC = INT_FEATURES + FLOAT_FEATURES
MODEL_FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"
N_ROWS = 2400
GATE = 0.70
PSI_LIMIT = 0.2
MODEL_NAME = "fresh-and-good-churn"
ALIAS = "champion"
CUSTOMER_ID = re.compile(r"FG(\d{4})")
PIP_REQUIREMENTS = ["numpy==2.5.4", "pandas==2.3.3", "scikit-learn==1.9.1"]


def customer_id(index: int) -> str:
    return f"FG{index:04d}"


def parse_customer_id(value: str) -> str:
    match = CUSTOMER_ID.fullmatch(value.strip())
    if match is None:
        raise ValueError("Use a customer id like FG0010.")
    number = int(match.group(1))
    if number >= N_ROWS:
        raise ValueError(f"Customer ids run from FG0000 through FG{N_ROWS - 1:04d}.")
    return customer_id(number)


def _open_store() -> None:
    uri = MLRUNS.resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)


def population_stability(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.astype(float).to_numpy()
    current = current.astype(float).to_numpy()
    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    ref_counts = np.histogram(reference, bins=cuts)[0].astype(float)
    cur_counts = np.histogram(current, bins=cuts)[0].astype(float)
    ref_share = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1.0), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in NUMERIC:
        score = population_stability(reference[name], current[name])
        rows.append({"column": name, "psi": round(score, 4), "check": score >= PSI_LIMIT})
    return pd.DataFrame(rows)


def prepare() -> None:
    if not CSV.is_file():
        raise FileNotFoundError(CSV)
    raw = pd.read_csv(CSV)
    if len(raw) != N_ROWS:
        raise ValueError(f"Expected {N_ROWS} rows, found {len(raw)}.")
    frame = raw.loc[:, list(MODEL_FEATURES + (TARGET,))].copy()
    frame.insert(0, "customer_id", [customer_id(i) for i in range(len(frame))])
    for name in NUMERIC:
        frame[name] = frame[name].astype("float64")
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), FEATURES)
    reference = frame.loc[:, list(NUMERIC)]
    pq.write_table(pa.Table.from_pandas(reference, preserve_index=False), REFERENCE)
    if not LIVE.is_file():
        pq.write_table(pa.Table.from_pandas(reference, preserve_index=False), LIVE)

    predictors = frame.loc[:, list(MODEL_FEATURES)]
    labels = frame[TARGET].astype(int)
    train_x, test_x, train_y, test_y = train_test_split(
        predictors, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), list(NUMERIC)),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            list(CATEGORICAL),
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )
    pipeline.fit(train_x, train_y)
    scores = pipeline.predict_proba(test_x)[:, 1]
    auc = float(roc_auc_score(test_y, scores))
    if auc < GATE:
        raise RuntimeError(f"roc_auc {auc:.4f} is below {GATE:.2f}.")
    _open_store()
    mlflow.set_experiment("fresh-and-good-capstone")
    with mlflow.start_run() as run:
        mlflow.log_metric("roc_auc", auc)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=train_x.head(2),
            pip_requirements=PIP_REQUIREMENTS,
        )
        run_id = run.info.run_id
    client = MlflowClient()
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except Exception:
        current = None
    if current is None or current.run_id != run_id:
        version = mlflow.register_model(info.model_uri, MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    metrics = {"roc_auc": round(auc, 6), "gate": GATE, "passed": True, "run_id": run_id}
    (ROOT / "data" / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    feast_frame = frame.loc[:, ["customer_id", "orders_90d", "plan"]].copy()
    feast_frame["orders_90d"] = feast_frame["orders_90d"].astype("int64")
    table = pa.table(
        [
            pa.array(feast_frame["customer_id"].astype(str).tolist(), type=pa.string()),
            pa.array([datetime(2026, 6, 30)] * len(feast_frame), type=pa.timestamp("us")),
            pa.array(feast_frame["orders_90d"].to_numpy(), type=pa.int64()),
            pa.array(feast_frame["plan"].astype(str).tolist(), type=pa.string()),
        ],
        names=["customer_id", "event_timestamp", "orders_90d", "plan"],
    )
    FEAST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, FEAST_PARQUET)
    print(json.dumps(metrics))


def score_customer(customer_id_text: str) -> float:
    identity = parse_customer_id(customer_id_text)
    if not FEATURES.is_file():
        raise FileNotFoundError("No feature table yet. Run make prepare first.")
    frame = pq.read_table(FEATURES).to_pandas()
    row = frame.loc[frame["customer_id"] == identity, list(MODEL_FEATURES)]
    if row.empty:
        raise LookupError(f"No feature row for {identity}.")
    _open_store()
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    return float(model.predict_proba(row)[0, 1])


def monitor_table() -> pd.DataFrame:
    if not REFERENCE.is_file() or not LIVE.is_file():
        raise FileNotFoundError("No drift tables yet. Run make prepare first.")
    reference = pq.read_table(REFERENCE).to_pandas()
    current = pq.read_table(LIVE).to_pandas()
    return drift_table(reference, current)


def materialize_feast() -> None:
    import subprocess
    import sys

    if not FEAST_PARQUET.is_file():
        raise FileNotFoundError("No Feast table yet. Run make prepare first.")
    feast = Path(sys.executable).with_name("feast")
    subprocess.run([str(feast), "apply"], cwd=FEATURE_REPO, check=True)
    subprocess.run(
        [
            str(feast),
            "materialize",
            "2026-06-01T00:00:00",
            "2026-07-02T00:00:00",
        ],
        cwd=FEATURE_REPO,
        check=True,
    )


def feast_lookup(customer_id_text: str) -> tuple[int, str]:
    identity = parse_customer_id(customer_id_text)
    registry = FEATURE_REPO / "data" / "registry.db"
    if not registry.is_file():
        raise FileNotFoundError("Feast registry is missing. Run make prepare first.")
    from feast import FeatureStore

    result = FeatureStore(repo_path=str(FEATURE_REPO)).get_online_features(
        features=["customer_features:orders_90d", "customer_features:plan"],
        entity_rows=[{"customer_id": identity}],
    ).to_dict()
    return int(result["orders_90d"][0]), str(result["plan"][0])


def main() -> None:
    prepare()
    materialize_feast()


if __name__ == "__main__":
    main()
