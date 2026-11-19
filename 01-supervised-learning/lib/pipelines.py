import time

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from features import add_row_features
from schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from spend import clip_spend

ENGINEERED_FEATURES = ("orders_per_week", "recency_over_45")
CHURN_NAMES = ("logistic", "forest", "svc")
SPEND_NAMES = ("ridge", "forest", "hist_gradient")


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def make_engineered_preprocessor() -> ColumnTransformer:
    numeric = list(NUMERIC_FEATURES) + list(ENGINEERED_FEATURES)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def make_candidate_pipeline(estimator, task: str = "churn") -> Pipeline:
    """Engineer, scale, select, then fit one model.

    The input columns are the original June fields. `task` is `churn` or `spend`.
    """
    if task == "churn":
        score_func, k = f_classif, 8
    elif task == "spend":
        score_func, k = f_regression, 5
    else:
        raise ValueError("task must be churn or spend.")
    return Pipeline(
        steps=[
            ("add", FunctionTransformer(add_row_features, validate=False)),
            ("prep", make_engineered_preprocessor()),
            ("select", SelectKBest(score_func, k=k)),
            ("model", estimator),
        ]
    )


def make_feature_pipeline(C: float = 1.0) -> Pipeline:
    """Logistic candidate used by the form check. Same steps as the churn search."""
    return make_candidate_pipeline(
        LogisticRegression(C=C, max_iter=500, solver="lbfgs"),
        task="churn",
    )


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("model", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )


def make_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_dummy_spend_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def make_ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def churn_folds() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def spend_folds() -> KFold:
    return KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def candidate_spec(name: str, task: str) -> tuple[object, dict]:
    """Return the estimator and the small grid for one family."""
    if task == "churn" and name == "logistic":
        return (
            LogisticRegression(max_iter=500, solver="lbfgs"),
            {"model__C": [0.1, 1.0, 10.0]},
        )
    if task == "churn" and name == "forest":
        return (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [40, 80],
                "model__max_depth": [3, 8],
                "model__min_samples_leaf": [5, 20],
            },
        )
    if task == "churn" and name == "svc":
        # SVC(probability=True) is deprecated in scikit-learn 1.9 and removed in 1.11.
        # CalibratedClassifierCV is the replacement that still returns probabilities.
        model = CalibratedClassifierCV(
            SVC(kernel="rbf", random_state=RANDOM_STATE),
            ensemble=False,
        )
        return model, {"model__estimator__C": [1.0, 10.0]}
    if task == "spend" and name == "ridge":
        return Ridge(), {"model__alpha": [0.1, 1.0, 10.0, 50.0]}
    if task == "spend" and name == "forest":
        return (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "model__n_estimators": [40, 80],
                "model__max_depth": [3, 8],
                "model__min_samples_leaf": [5, 20],
            },
        )
    if task == "spend" and name == "hist_gradient":
        return (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__learning_rate": [0.05, 0.1], "model__max_depth": [3, 6]},
        )
    raise KeyError(f"No {task} candidate named {name}.")


def fit_grid(estimator, grid: dict, features: pd.DataFrame, labels: pd.Series, task: str):
    """Fit a small grid on train rows. Returns the search and the seconds it took."""
    if task == "churn":
        cv = churn_folds()
        scoring = "roc_auc"
    elif task == "spend":
        cv = spend_folds()
        scoring = "neg_mean_absolute_error"
    else:
        raise ValueError("task must be churn or spend.")
    started = time.perf_counter()
    search = GridSearchCV(
        estimator=make_candidate_pipeline(estimator, task=task),
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=1,
    )
    search.fit(features, labels)
    return search, time.perf_counter() - started


def fit_named(name: str, features: pd.DataFrame, labels: pd.Series, task: str):
    estimator, grid = candidate_spec(name, task)
    return fit_grid(estimator, grid, features, labels, task)


def score_row(name: str, search: GridSearchCV, seconds: float, features, labels, task: str) -> dict:
    index = int(search.best_index_)
    cv_mean = float(search.cv_results_["mean_test_score"][index])
    cv_std = float(search.cv_results_["std_test_score"][index])
    if task == "spend":
        cv_mean = -cv_mean
        predicted = clip_spend(search.predict(features))
        validation = float(mean_absolute_error(labels, predicted))
        extra = float(root_mean_squared_error(labels, predicted))
    elif task == "churn":
        validation = float(roc_auc_score(labels, search.predict_proba(features)[:, 1]))
        extra = None
    else:
        raise ValueError("task must be churn or spend.")
    return {
        "name": name,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "validation": validation,
        "rmse": extra,
        "seconds": float(seconds),
        "params": dict(search.best_params_),
    }


def format_comparison(rows: list[dict], task: str) -> str:
    if task == "churn":
        lines = [f"{'name':<16}{'cv_auc':>10}{'cv_std':>10}{'val_auc':>10}{'seconds':>10}"]
        for row in rows:
            lines.append(
                f"{row['name']:<16}{row['cv_mean']:10.3f}{row['cv_std']:10.3f}"
                f"{row['validation']:10.3f}{row['seconds']:10.1f}"
            )
        return "\n".join(lines)
    if task == "spend":
        lines = [f"{'name':<16}{'cv_mae':>10}{'cv_std':>10}{'val_mae':>10}{'val_rmse':>10}{'seconds':>10}"]
        for row in rows:
            lines.append(
                f"{row['name']:<16}{row['cv_mean']:10.1f}{row['cv_std']:10.1f}"
                f"{row['validation']:10.1f}{row['rmse']:10.1f}{row['seconds']:10.1f}"
            )
        return "\n".join(lines)
    raise ValueError("task must be churn or spend.")


def pick_winner(rows: list[dict], task: str) -> str:
    if task == "churn":
        return max(rows, key=lambda row: row["validation"])["name"]
    if task == "spend":
        return min(rows, key=lambda row: row["validation"])["name"]
    raise ValueError("task must be churn or spend.")
