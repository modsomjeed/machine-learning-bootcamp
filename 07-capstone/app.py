import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from pipeline import PSI_LIMIT, feast_lookup, monitor_table, score_customer


def main() -> None:
    st.title("Fresh and Good capstone")
    st.caption(
        "Demo form for a fictional grocery. "
        "Run make prepare in this folder before scoring. "
        "The model, the drift tables, and the Feast lookup all live here."
    )
    page = st.radio("Page", ["Score", "Monitor", "Feast"], horizontal=True)
    customer_id = st.text_input("Customer id", value="FG0010")
    if page == "Score":
        if not st.button("Score this customer"):
            st.info("Score a customer, or open Monitor.")
            return
        try:
            probability = score_customer(customer_id)
        except Exception as exc:
            st.error(str(exc))
            return
        st.metric("Churn probability", f"{probability:.1%}")
        return
    if page == "Monitor":
        try:
            table = monitor_table()
        except Exception as exc:
            st.error(str(exc))
            return
        st.dataframe(table)
        flagged = table.loc[table["check"], "column"].tolist()
        if flagged:
            st.error("Drift is above the line for " + ", ".join(flagged) + ".")
        else:
            st.success("No column is over the PSI line.")
        return
    if not st.button("Look up in Feast"):
        st.info(f"A column with PSI at or above {PSI_LIMIT} is marked on the monitor page.")
        return
    try:
        orders, plan = feast_lookup(customer_id)
    except Exception as exc:
        st.error(str(exc))
        return
    st.write(f"{customer_id} has plan {plan} and {orders} orders in the last 90 days.")


main()
