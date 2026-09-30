"""Step 2: turn raw tables into a feature matrix X and labels Y at a given cutoff month.
Only information from BEFORE the cutoff goes into X (no leakage).
Labels say: did the customer take product P in the HORIZON months after the cutoff?
"""
import os
import pandas as pd
from config import *


def load_raw():
    cust = pd.read_csv(os.path.join(DATA_DIR, "customers.csv")).set_index("customer")
    tx = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
    prod = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    return cust, tx, prod


def _avg_spend(tx, lo, hi, index):
    w = tx[(tx.month >= lo) & (tx.month < hi)]
    piv = w.groupby(["customer", "category"]).amount.mean().unstack(fill_value=0)
    return piv.reindex(index=index, columns=CATEGORIES, fill_value=0)


def build_dataset(cutoff, raw=None):
    cust, tx, prod = raw if raw is not None else load_raw()
    idx = cust.index

    recent = _avg_spend(tx, cutoff - 3, cutoff, idx)        # last 3 months
    prior = _avg_spend(tx, cutoff - 6, cutoff - 3, idx)     # the 3 months before that
    change = (recent - prior) / (prior + 1)

    held = pd.DataFrame(0, index=idx, columns=PRODUCTS)
    for _, r in prod[prod.start_month < cutoff].iterrows():
        held.loc[r.customer, r["product"]] = 1

    labels = pd.DataFrame(0, index=idx, columns=PRODUCTS)
    future = prod[(prod.start_month >= cutoff) & (prod.start_month < cutoff + HORIZON)]
    for _, r in future.iterrows():
        labels.loc[r.customer, r["product"]] = 1

    X = pd.concat([
        cust[["age", "income"]],
        pd.get_dummies(cust.persona, prefix="persona").astype(int),
        held.add_prefix("held_"),
        recent.add_prefix("recent_"),
        change.add_prefix("chg_"),
    ], axis=1)
    Y = labels.add_prefix("y_")
    return X, Y


def build_train(raw=None):
    """Stack several training cutoffs -> more rows, still strictly before the test period."""
    raw = raw if raw is not None else load_raw()
    parts = [build_dataset(c, raw) for c in TRAIN_CUTOFFS]
    X = pd.concat([p[0] for p in parts], ignore_index=True)
    Y = pd.concat([p[1] for p in parts], ignore_index=True)
    return X, Y
