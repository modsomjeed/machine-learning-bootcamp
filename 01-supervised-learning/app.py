import os
from pathlib import Path

import pandas as pd
import streamlit as st

from paths import churn_bundle_path, spend_bundle_path
from persist import TASK_CHURN, TASK_SPEND, load_bundle
from schema import AISLES, DISTRICTS, FEATURE_COLUMNS, PLANS
from spend import clip_spend
from validate import input_errors

PAGE_STYLE = """
<style>
@import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap");

:root {
  --fg-ink: #163528;
  --fg-leaf: #1f7a4d;
  --fg-leaf-deep: #145c39;
  --fg-mist: #e8f3ec;
  --fg-paper: #f7faf7;
  --fg-line: #c5d9cc;
  --fg-warn: #9a3412;
  --fg-ok: #166534;
}

html, body, [class*="css"] {
  font-family: "Manrope", sans-serif;
  color: var(--fg-ink);
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 50% at 10% -10%, #d7efe0 0%, transparent 55%),
    radial-gradient(ellipse 60% 40% at 100% 0%, #f3e8d4 0%, transparent 45%),
    linear-gradient(180deg, #f4f8f4 0%, #eef5f0 100%);
}

[data-testid="stHeader"] {
  background: transparent;
}

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 2rem;
  max-width: 1180px;
}

.fg-hero {
  display: grid;
  gap: 0.35rem;
  margin: 0 0 1.25rem 0;
  padding: 1.15rem 1.25rem;
  border: 1px solid var(--fg-line);
  border-radius: 18px;
  background:
    linear-gradient(135deg, rgba(31, 122, 77, 0.12), rgba(255, 255, 255, 0.7) 55%),
    var(--fg-paper);
  box-shadow: 0 10px 30px rgba(22, 53, 40, 0.06);
}

.fg-kicker {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--fg-leaf-deep);
}

.fg-hero h1 {
  margin: 0;
  font-family: "Fraunces", Georgia, serif;
  font-size: clamp(1.9rem, 3vw, 2.5rem);
  font-weight: 700;
  line-height: 1.1;
  color: var(--fg-ink);
}

.fg-hero p {
  margin: 0;
  max-width: 42rem;
  color: #355446;
  font-size: 0.98rem;
  line-height: 1.45;
}

.fg-panel-label {
  margin: 0 0 0.55rem 0;
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--fg-ink);
}

div[data-testid="stElementContainer"]:has(.fg-hint-slot),
div[data-testid="stMarkdownContainer"]:has(.fg-hint-slot),
div[data-testid="stMarkdownContainer"]:has(.fg-hint-slot) > * {
  height: auto !important;
  overflow: visible !important;
}

.fg-hint-slot {
  display: block;
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  margin: 0 0 0.9rem 0;
  padding: 0.95rem 1.05rem;
  border-left: 3px solid var(--fg-leaf);
  border-radius: 0 12px 12px 0;
  background: rgba(31, 122, 77, 0.08);
  color: #2d4f3d;
  font-family: "Manrope", sans-serif;
  font-size: 0.9rem;
  font-weight: 500;
  line-height: 1.55;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.fg-hint-slot code {
  font-size: 0.88em;
  padding: 0.12em 0.4em;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--fg-line);
  color: var(--fg-leaf-deep);
}

.fg-meta {
  margin: 0 0 1rem 0;
  padding: 0.65rem 0.8rem;
  border-radius: 12px;
  background: var(--fg-mist);
  border: 1px solid var(--fg-line);
  color: #2f5240;
  font-size: 0.88rem;
  line-height: 1.4;
}

.fg-voucher {
  margin-top: 0.75rem;
  padding: 0.85rem 1rem;
  border-radius: 14px;
  border: 1px solid var(--fg-line);
  background: white;
  font-weight: 600;
}

.fg-voucher.yes {
  border-color: #86efac;
  background: #ecfdf5;
  color: var(--fg-ok);
}

.fg-voucher.no {
  border-color: #fdba74;
  background: #fff7ed;
  color: var(--fg-warn);
}

div[data-testid="stForm"] {
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid var(--fg-line);
  border-radius: 18px;
  padding: 0.85rem 1rem 1rem;
  box-shadow: 0 8px 24px rgba(22, 53, 40, 0.05);
}

div[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
div[data-testid="stForm"] label[data-testid="stWidgetLabel"] p,
div[data-testid="stForm"] .stSelectbox label p,
div[data-testid="stForm"] .stNumberInput label p,
div[data-testid="stForm"] [data-baseweb="select"] ~ label p {
  font-family: "Manrope", sans-serif !important;
  font-size: 0.84rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  text-transform: none !important;
  color: var(--fg-leaf-deep) !important;
  opacity: 1 !important;
}

div[data-testid="stForm"] [data-testid="stWidgetLabel"],
div[data-testid="stForm"] label[data-testid="stWidgetLabel"] {
  color: var(--fg-leaf-deep) !important;
  opacity: 1 !important;
  margin-bottom: 0.2rem !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255, 255, 255, 0.78);
  border-color: var(--fg-line) !important;
  border-radius: 18px;
  box-shadow: 0 8px 24px rgba(22, 53, 40, 0.05);
}

div[data-testid="stMetric"] {
  background: white;
  border: 1px solid var(--fg-line);
  border-radius: 14px;
  padding: 0.85rem 1rem;
}

div[data-testid="stMetric"] label {
  color: #4a6a58 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: "Fraunces", Georgia, serif;
  color: var(--fg-ink) !important;
}

.stFormSubmitButton button {
  background: linear-gradient(180deg, var(--fg-leaf), var(--fg-leaf-deep)) !important;
  color: white !important;
  border: none !important;
  border-radius: 999px !important;
  font-weight: 700 !important;
  padding-top: 0.55rem !important;
  padding-bottom: 0.55rem !important;
}

.stFormSubmitButton button:hover {
  filter: brightness(1.05);
}

div[data-testid="stElementContainer"]:has(.fg-stretch),
div[data-testid="stMarkdownContainer"]:has(.fg-stretch),
div[data-testid="stMarkdownContainer"]:has(.fg-stretch) > * {
  height: auto !important;
  overflow: visible !important;
}

.fg-stretch {
  display: block;
  box-sizing: border-box;
  width: 100%;
  max-width: 100%;
  margin: 0.85rem 0 0 0;
  padding: 0.95rem 1.05rem;
  border: 1px solid var(--fg-line);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.72);
  box-shadow: 0 8px 24px rgba(22, 53, 40, 0.05);
  color: #2f5240;
  font-family: "Manrope", sans-serif;
  font-size: 0.92rem;
  font-weight: 500;
  line-height: 1.55;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.fg-stretch strong {
  display: block;
  margin: 0 0 0.4rem 0;
  font-family: "Fraunces", Georgia, serif;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--fg-ink);
}

.fg-stretch code {
  font-size: 0.88em;
  padding: 0.12em 0.4em;
  border-radius: 6px;
  background: var(--fg-mist);
  border: 1px solid var(--fg-line);
  color: var(--fg-leaf-deep);
}
</style>
"""


def churn_decision(probability: float, threshold: float) -> str:
    """Return yes or no for a retention voucher.

    Exercise. Return yes when probability is at least threshold.
    Otherwise return no.
    """
    raise NotImplementedError(
        "Finish churn_decision in app.py. Return yes when probability is at least the threshold."
    )


def _path(env_name: str, default: Path) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override)
    return default


def _load(path: Path, task: str, missing_notebook: str):
    if not path.is_file():
        return None, (
            f"No {task} file at {path}. Finish {missing_notebook}, then refresh this page. "
            "Load only a file you trust. A joblib file can run code when it is loaded."
        )
    try:
        return load_bundle(path, expected_task=task), None
    except Exception as exc:
        return None, str(exc)


def _render_scores(churn_bundle, spend_bundle, record: dict | None) -> None:
    if record is None:
        st.info("Fill the form on the left, then score this customer.")
        return

    errors = input_errors(record)
    if errors:
        for message in errors:
            st.error(message)
        return

    row = pd.DataFrame([record]).loc[:, list(FEATURE_COLUMNS)]
    if churn_bundle is not None:
        probability = float(churn_bundle.pipeline.predict_proba(row)[0, 1])
        st.metric("Churn probability", f"{probability:.1%}")
        try:
            decision = churn_decision(probability, churn_bundle.threshold)
        except NotImplementedError as exc:
            st.error(str(exc))
        else:
            if decision not in {"yes", "no"}:
                st.error("churn_decision must return yes or no.")
            else:
                # Keep this exact prefix for classroom checks.
                st.write(f"Offer a voucher: {decision}")
                tone = "yes" if decision == "yes" else "no"
                label = "Send a small voucher" if decision == "yes" else "Leave the customer alone"
                st.markdown(
                    f'<div class="fg-voucher {tone}">{label}</div>',
                    unsafe_allow_html=True,
                )
    if spend_bundle is not None:
        raw = float(spend_bundle.pipeline.predict(row)[0])
        estimate = float(clip_spend([raw])[0])
        st.metric("Estimated next-month spend", f"{estimate:,.0f} THB")


def main() -> None:
    st.set_page_config(
        page_title="Fresh and Good",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="fg-hero">
          <div class="fg-kicker">Bangkok online grocery · classroom demo</div>
          <h1>Fresh and Good</h1>
          <p>
            Score one customer for churn risk and next-month spend.
            Synthetic data only — not a production system.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    churn_path = _path("FRESH_AND_GOOD_CHURN_BUNDLE", churn_bundle_path())
    spend_path = _path("FRESH_AND_GOOD_SPEND_BUNDLE", spend_bundle_path())
    churn_bundle, churn_error = _load(churn_path, TASK_CHURN, "notebook 05")
    spend_bundle, spend_error = _load(spend_path, TASK_SPEND, "notebook 06")
    if churn_error:
        st.error(churn_error)
    if spend_error:
        st.error(spend_error)
    if churn_bundle is None and spend_bundle is None:
        st.stop()

    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="fg-panel-label">Customer</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="fg-hint-slot">
              Try a high-churn example: plan <code>none</code>, 0 orders and 0 spend,
              90 days since last order, 0 app sessions, 4 support tickets,
              district Bang Na, aisle fresh. Then score again.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("customer"):
            nums_left, nums_right = st.columns(2)
            with nums_left:
                tenure_days = st.number_input(
                    "Tenure days", min_value=0, max_value=5000, value=180, step=1
                )
                orders_90d = st.number_input(
                    "Orders in last 90 days", min_value=0, max_value=80, value=4, step=1
                )
                spend_thb_90d = st.number_input(
                    "Spend last 90 days, THB", min_value=0.0, value=1500.0, step=10.0
                )
                avg_basket_thb_90d = st.number_input(
                    "Average basket, THB", min_value=0.0, value=380.0, step=10.0
                )
            with nums_right:
                days_since_last_order = st.number_input(
                    "Days since last order", min_value=0, max_value=400, value=12, step=1
                )
                promo_order_share_90d = st.number_input(
                    "Promo order share", min_value=0.0, max_value=1.0, value=0.2, step=0.05
                )
                app_sessions_90d = st.number_input(
                    "App sessions last 90 days", min_value=0, max_value=200, value=8, step=1
                )
                support_tickets_90d = st.number_input(
                    "Support tickets last 90 days", min_value=0, max_value=30, value=0, step=1
                )

            cats = st.columns(3)
            with cats[0]:
                district = st.selectbox("District", DISTRICTS)
            with cats[1]:
                plan = st.selectbox("Plan", PLANS)
            with cats[2]:
                primary_aisle = st.selectbox("Primary aisle", AISLES)

            submitted = st.form_submit_button("Score this customer", use_container_width=True)

        st.markdown(
            """
            <div class="fg-stretch">
              <strong>Optional engineer stretch</strong>
              <code>input_errors</code> already rejects spend when the order count is zero.
              Add one more rule in <code>validate.py</code> if the team wants it,
              and surface that message here.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        with st.container(border=True):
            st.markdown('<div class="fg-panel-label">Score</div>', unsafe_allow_html=True)
            if churn_bundle is not None:
                st.markdown(
                    f"""
                    <p class="fg-meta">
                      Threshold <strong>{churn_bundle.threshold:.2f}</strong>
                      · Missed churner <strong>{churn_bundle.cost_fn_thb:.0f} THB</strong>
                      · Wasted voucher <strong>{churn_bundle.cost_fp_thb:.0f} THB</strong>
                    </p>
                    """,
                    unsafe_allow_html=True,
                )

            record = None
            if submitted:
                record = {
                    "tenure_days": int(tenure_days),
                    "orders_90d": int(orders_90d),
                    "spend_thb_90d": float(spend_thb_90d),
                    "avg_basket_thb_90d": float(avg_basket_thb_90d),
                    "days_since_last_order": int(days_since_last_order),
                    "promo_order_share_90d": float(promo_order_share_90d),
                    "app_sessions_90d": int(app_sessions_90d),
                    "support_tickets_90d": int(support_tickets_90d),
                    "district": district,
                    "plan": plan,
                    "primary_aisle": primary_aisle,
                }
            _render_scores(churn_bundle, spend_bundle, record)


main()
