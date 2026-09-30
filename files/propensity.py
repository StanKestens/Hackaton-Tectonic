"""LightGBM product propensity (+ SHAP reasons) for all simulated customers -> models/propensity.csv

Runs in its own process on purpose: LightGBM and PyTorch each bring their own OpenMP runtime,
and loading both in one Python process crashes/hangs on macOS.
Run: python propensity.py [--limit N]
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
import shap

from config import *
from explain import describe
from features import build_dataset, load_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    X, _ = build_dataset(TEST_CUTOFF, load_raw())
    if a.limit:
        X = X.iloc[:a.limit]
    rows = []
    for p in PRODUCTS:
        path = os.path.join(MODEL_DIR, f"lgbm_{p}.joblib")
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        cand = X[X[f"held_{p}"] == 0]
        if cand.empty:
            continue
        prob = model.predict_proba(cand)[:, 1]
        sv = np.asarray(shap.TreeExplainer(model).shap_values(cand))
        sv = sv[..., 1] if sv.ndim == 3 else sv
        for i, c in enumerate(cand.index):
            top = [describe(cand.columns[j], cand.iloc[i, j], sv[i, j]) for j in np.argsort(-np.abs(sv[i]))[:4]]
            rows.append((c, p, float(prob[i]), "; ".join(t for t in top if t)[:300]))
    out = pd.DataFrame(rows, columns=["customer", "product", "probability", "evidence"])
    out.to_csv(os.path.join(MODEL_DIR, "propensity.csv"), index=False)
    print(f"wrote {MODEL_DIR}/propensity.csv: {len(out):,} rows")


if __name__ == "__main__":
    main()
