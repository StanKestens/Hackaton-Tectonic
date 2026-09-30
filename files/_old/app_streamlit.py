"""Step 5: advisor dashboard. Run: streamlit run app.py"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

from config import *
from explain import top_reasons
from features import build_dataset, load_raw
from flatten import kbc_json_to_events

st.set_page_config(page_title="KBC customer understanding PoC", layout="wide")
st.title("What does this customer need, right now?")
tab_intent, tab_product = st.tabs(["Intent engine (TIMeSynC)", "Next-best-product (LightGBM)"])


# ---------------------------------------------------------------- TIMeSynC
@st.cache_resource
def load_predictor():
    from timesync_predict import IntentPredictor
    return IntentPredictor()


@st.cache_data
def load_sim_timelines():
    ev = pd.read_csv(os.path.join(DATA_DIR, "events.csv"), parse_dates=["timestamp"])
    it = pd.read_csv(os.path.join(DATA_DIR, "intents.csv"), parse_dates=["timestamp"])
    pr = pd.read_csv(os.path.join(DATA_DIR, "products.csv"), parse_dates=["timestamp"])
    return ev, it, pr


def show_prediction(pred):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Next intent")
        st.bar_chart(pd.Series(dict(pred["top_intents"])).sort_values(ascending=False), horizontal=True)
    with c2:
        st.subheader("Life event")
        st.bar_chart(pd.Series(pred["life_events"]), horizontal=True)
    st.subheader("Why? (what the model looked at)")
    st.dataframe(pd.DataFrame(pred["reasons"]), hide_index=True, use_container_width=True)


with tab_intent:
    if not os.path.exists(os.path.join(MODEL_DIR, "timesync.pt")):
        st.warning("No TIMeSynC model yet. Run `python train_timesync.py` first.")
    else:
        from timesync_predict import products_from_timeline
        predictor = load_predictor()
        source = st.radio("Customers", ["KBC mock data", "Simulated customers"], horizontal=True)
        if source == "KBC mock data":
            clients = json.load(open(MOCK_FILE))
            c = st.selectbox("Client", clients, format_func=lambda c: f"{c['client_id']} ({c.get('ground_truth_link')})")
            tl = pd.DataFrame(c["timeline"])
            st.dataframe(tl, hide_index=True, use_container_width=True)
            step = st.select_slider("Moment: one hour after transaction", options=list(tl.tx_id), value=tl.tx_id.iloc[0])
            at = pd.Timestamp(tl.set_index("tx_id").loc[step, "timestamp"]).tz_convert(None) + pd.Timedelta(hours=1)
            show_prediction(predictor.predict(kbc_json_to_events([c]), at, products=products_from_timeline(c["timeline"])))
        else:
            ev, it, pr = load_sim_timelines()
            cid = st.selectbox("Customer", sorted(ev.customer.unique())[:300])
            ev_c, it_c, pr_c = ev[ev.customer == cid], it[it.customer == cid], pr[pr.customer == cid]
            lo, hi = ev_c.timestamp.min().date(), ev_c.timestamp.max().date()
            day = st.slider("Moment", min_value=lo, max_value=hi,
                            value=(pd.Timestamp(START_DATE) + pd.DateOffset(months=TEST_MONTH)).date())
            at = pd.Timestamp(day) + pd.Timedelta(hours=12)
            show_prediction(predictor.predict(ev_c, at, intents=it_c,
                                              products=list(zip(pr_c["product"], pr_c.timestamp))))
            with st.expander("Actual intents around this moment (ground truth)"):
                st.dataframe(it_c[(it_c.timestamp >= at - pd.Timedelta(days=30)) &
                                  (it_c.timestamp <= at + pd.Timedelta(days=30))], hide_index=True)
            with st.expander("Raw event stream before this moment"):
                st.dataframe(ev_c[ev_c.timestamp < at].tail(60), hide_index=True)


# ---------------------------------------------------------------- LightGBM
@st.cache_data
def load_tabular():
    raw = load_raw()
    X, _ = build_dataset(TEST_CUTOFF, raw)
    return raw, X


@st.cache_resource
def load_models():
    return {p: joblib.load(os.path.join(MODEL_DIR, f"lgbm_{p}.joblib"))
            for p in PRODUCTS if os.path.exists(os.path.join(MODEL_DIR, f"lgbm_{p}.joblib"))}


with tab_product:
    st.caption(f"Which product will the customer take in the next {HORIZON} months after month {TEST_CUTOFF}?")
    models = load_models()
    if not models:
        st.warning("No LightGBM models yet. Run `python train.py` first.")
    else:
        (cust, tx, prod), X = load_tabular()
        cid = st.selectbox("Customer ", list(X.index[:500]))
        row = X.loc[[cid]]
        st.write(cust.loc[cid].to_frame().T)
        scores = {p: float(m.predict_proba(row)[0, 1]) for p, m in models.items()
                  if row[f"held_{p}"].iloc[0] == 0}
        if not scores:
            st.info("This customer already holds every product.")
        else:
            left, right = st.columns(2)
            with left:
                st.subheader("Predicted need (probability)")
                st.bar_chart(pd.Series(scores).sort_values(ascending=False))
                best = max(scores, key=scores.get)
                st.success(f"Top suggestion: **{best.replace('_', ' ')}** ({scores[best]:.0%})")
                st.subheader("Why?")
                for reason in top_reasons(models[best], row):
                    st.write("- " + reason)
            with right:
                st.subheader("Recent spending by category")
                t = tx[(tx.customer == cid) & (tx.month < TEST_CUTOFF)]
                st.line_chart(t.pivot(index="month", columns="category", values="amount"))
