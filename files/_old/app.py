"""Step 5: advisor dashboard. Run: streamlit run app.py"""
import os
import joblib
import pandas as pd
import streamlit as st
from config import *
from features import load_raw, build_dataset
from explain import top_reasons

st.set_page_config(page_title="Next-best-product PoC", layout="wide")
st.title("What will this customer need next?")
st.caption("Proof of concept on synthetic data. Forecast window: next "
           f"{HORIZON} months after month {TEST_CUTOFF}.")


@st.cache_data
def load():
    raw = load_raw()
    X, _ = build_dataset(TEST_CUTOFF, raw)
    return raw, X


@st.cache_resource
def load_models():
    return {p: joblib.load(os.path.join(MODEL_DIR, f"lgbm_{p}.joblib"))
            for p in PRODUCTS if os.path.exists(os.path.join(MODEL_DIR, f"lgbm_{p}.joblib"))}


(cust, tx, prod), X = load()
models = load_models()

cid = st.sidebar.selectbox("Customer", list(X.index[:500]))
row = X.loc[[cid]]
st.sidebar.write(cust.loc[cid])

scores = {p: float(m.predict_proba(row)[0, 1]) for p, m in models.items()
          if row[f"held_{p}"].iloc[0] == 0}
if not scores:
    st.info("This customer already holds every product.")
    st.stop()

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
