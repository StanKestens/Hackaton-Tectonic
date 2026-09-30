"""Step 3: baseline vs LightGBM per product, evaluated on a later time period.
Run: python train.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from config import *
from features import load_raw, build_train, build_dataset


def precision_at_k(y, score, frac=0.05):
    k = max(1, int(len(y) * frac))
    return y[np.argsort(-score)[:k]].mean()


def main():
    raw = load_raw()
    Xtr, Ytr = build_train(raw)
    Xte, Yte = build_dataset(TEST_CUTOFF, raw)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(list(Xtr.columns), os.path.join(MODEL_DIR, "columns.joblib"))

    rows = []
    for p in PRODUCTS:
        # only customers who don't hold the product yet are candidates
        tr = Xtr[f"held_{p}"] == 0
        te = Xte[f"held_{p}"] == 0
        ytr, yte = Ytr.loc[tr, f"y_{p}"].values, Yte.loc[te, f"y_{p}"].values
        if ytr.sum() < 20 or yte.sum() < 5:
            print(f"skip {p}: too few positives")
            continue

        logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        logit.fit(Xtr[tr], ytr)
        gbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                 subsample=0.8, colsample_bytree=0.8, verbose=-1)
        gbm.fit(Xtr[tr], ytr)
        joblib.dump(gbm, os.path.join(MODEL_DIR, f"lgbm_{p}.joblib"))

        s_log = logit.predict_proba(Xte[te])[:, 1]
        s_gbm = gbm.predict_proba(Xte[te])[:, 1]
        rows.append({
            "product": p,
            "base_rate": yte.mean(),                          # 'most popular' baseline: PR-AUC = base rate
            "pr_auc_logit": average_precision_score(yte, s_log),
            "pr_auc_lgbm": average_precision_score(yte, s_gbm),
            "prec@5%_logit": precision_at_k(yte, s_log),
            "prec@5%_lgbm": precision_at_k(yte, s_gbm),
        })

    res = pd.DataFrame(rows).round(3)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(MODEL_DIR, "results.csv"), index=False)


if __name__ == "__main__":
    main()
