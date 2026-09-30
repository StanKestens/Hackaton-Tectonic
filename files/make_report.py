"""Generate the REAL financial report: actual data + model predictions -> report/financial_report.csv

Simulated customers: report as of the start of the test period (TEST_MONTH), so predictions only use
data from before that date (honest, out-of-sample). KBC-schema JSON clients: as of 1h after their last
transaction; only transaction-based sections are available for them.

Run: python make_report.py                          # all simulated customers + data/kbc_mock.json
     python make_report.py --limit 200 --kbc other.json -o report/x.csv
"""
import argparse
import json
import os

import subprocess
import sys

import numpy as np
import pandas as pd

from config import *
from financial_report import PRODUCT_FAMILY, write_csv
from flatten import MCC_CATEGORY, kbc_json_to_events

T0 = pd.Timestamp(START_DATE)
NEED_LABELS = {
    "BUY_CAR": "Buying a car", "INSURE_CAR": "Car insurance", "ROADSIDE_ASSISTANCE": "Roadside assistance",
    "BUY_HOME": "Buying a home", "INSURE_HOME": "Home insurance", "RENOVATION_LOAN": "Renovation loan",
    "BUY_BIKE": "Buying a bike", "INSURE_BIKE": "Bike insurance",
    "PREPARE_FOR_BABY": "Preparing for a baby", "PROTECT_FAMILY": "Family / life insurance",
    "SAVE_FOR_CHILD": "Savings for a child", "INVEST": "Investing", "PENSION_SAVING": "Pension saving",
}
EVENT_LABELS = {"automotive_lifecycle": "Buying a car", "real_estate_acquisition": "Buying a home",
                "micro_mobility_acquisition": "Buying a bike / e-bike", "family_expansion": "Baby on the way",
                "wealth_building": "Building wealth / investing"}


def period(m):
    return str((T0 + pd.DateOffset(months=int(m))).to_period("M"))


def predictions(pred, top=5):
    from timesync_predict import INTENT_TO_EVENT
    """TIMeSynC output -> needs + life events (P(event) = P(at least one of its needs in 30 days))."""
    evidence = "; ".join(f"{r['token']} ({r['time'][:10]})" for r in pred["reasons"][:3])
    needs = [(n, NEED_LABELS.get(n, n), p, evidence, INTENT_TO_EVENT.get(n, "")) for n, p in pred["needs_30d"][:top]]
    ev = {}
    for n, p in pred["needs_30d"]:
        e = INTENT_TO_EVENT.get(n)
        if e:
            ev[e] = 1 - (1 - ev.get(e, 0.0)) * (1 - p)
    events = [(e, EVENT_LABELS.get(e, e), p, evidence) for e, p in sorted(ev.items(), key=lambda kv: -kv[1])]
    return needs, events


def propensity_table(ids):
    """LightGBM output, computed in a separate process (see propensity.py: OpenMP clash with torch)."""
    subprocess.run([sys.executable, "propensity.py", "--limit", str(len(ids))], check=True)
    df = pd.read_csv(os.path.join(MODEL_DIR, "propensity.csv"))
    out = {}
    for c, g in df.groupby("customer"):
        g = g.sort_values("probability", ascending=False)
        out[c] = [(p, p.replace("_", " ").capitalize(), pr, ev) for p, pr, ev in zip(g["product"], g.probability, g.evidence.fillna(""))]
    return out


def sim_specs(predictor, limit=None):
    as_of = T0 + pd.DateOffset(months=TEST_MONTH)
    last = TEST_MONTH - 1                                     # last complete month
    cust = pd.read_csv(f"{DATA_DIR}/customers.csv").set_index("customer")
    ids = list(cust.index[:limit] if limit else cust.index)
    ev = pd.read_csv(f"{DATA_DIR}/events.csv", parse_dates=["timestamp"])
    it = pd.read_csv(f"{DATA_DIR}/intents.csv", parse_dates=["timestamp"])
    pr = pd.read_csv(f"{DATA_DIR}/products.csv", parse_dates=["timestamp"])
    tx = pd.read_csv(f"{DATA_DIR}/transactions.csv")
    bal = pd.read_csv(f"{DATA_DIR}/balances.csv")
    hold = pd.read_csv(f"{DATA_DIR}/holdings.csv")
    prices = pd.read_csv(f"{DATA_DIR}/prices.csv").set_index(["item", "month"]).price
    raw = pd.read_csv(f"{DATA_DIR}/raw_transactions.csv")
    raw["ts"] = pd.to_datetime(raw.timestamp).dt.tz_localize(None)

    ev_g, it_g, pr_g = (dict(tuple(d.groupby("customer"))) for d in (ev, it, pr))
    tx_g, bal_g, hold_g, raw_g = (dict(tuple(d.groupby("customer"))) for d in (tx, bal, hold, raw))
    prop = propensity_table(ids)
    empty = pd.DataFrame(columns=["timestamp", "intent", "product", "month", "start_month"])

    specs = []
    for n, c in enumerate(ids):
        info = cust.loc[c]
        t = tx_g.get(c, empty)
        t = t[(t.month >= TEST_MONTH - 12) & (t.month <= last)]
        b = bal_g[c].set_index("month")
        h = hold_g.get(c, empty)
        h = h[h.start_month <= last]
        p = pr_g.get(c, empty)
        p = p[p.timestamp < as_of]
        r = raw_g.get(c, pd.DataFrame(columns=["ts"]))
        r = r[(r.ts < as_of) & (r.ts >= as_of - pd.Timedelta(days=90))]
        pred = predictor.predict(ev_g[c], as_of, intents=it_g.get(c, empty)[["timestamp", "intent"]],
                                 products=list(zip(p["product"], p.timestamp)))
        needs, events = predictions(pred)
        specs.append({
            "client_id": c, "as_of": str(as_of.date()),
            "profile": {"segment": info.persona, "age": int(info.age),
                        "income_band": f"{int(info.income // 1000) * 1000}-{int(info.income // 1000 + 1) * 1000} EUR"},
            "income": [(period(m), info.income) for m in range(TEST_MONTH - 12, TEST_MONTH)],
            "balances": {"current_account": b.loc[last, "current_account"], "savings_account": b.loc[last, "savings_account"],
                         "total_debt": b.loc[last, "debt"]},
            "balance_history": [(period(m), *b.loc[m, ["current_account", "savings_account", "investments", "debt"]])
                                for m in range(TEST_MONTH - 12, TEST_MONTH)],
            "expenses": {period(m): dict(zip(g.category, g.amount)) for m, g in t.groupby("month")},
            "holdings": [(row["item"], row.label, row.asset_class, row.units, prices[(row["item"], last)])
                         for _, row in h.iterrows()],
            "products": [(row["product"], period(max(row.start_month, 0)), PRODUCT_FAMILY.get(row["product"], ""))
                         for _, row in p.iterrows()],
            "transactions": r.drop(columns=["customer", "ts"], errors="ignore").to_dict("records"),
            "needs": needs, "life_events": events, "propensity": prop.get(c, []),
        })
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(ids)} simulated customers")
    return specs


def kbc_specs(predictor, path):
    from timesync_predict import products_from_timeline
    specs = []
    for c in json.load(open(path)):
        ev = kbc_json_to_events([c])
        as_of = ev.timestamp.max() + pd.Timedelta(hours=1)
        pred = predictor.predict(ev, as_of, products=products_from_timeline(c["timeline"]))
        needs, events = predictions(pred)
        tl = pd.DataFrame(c["timeline"])
        tl["month"] = pd.to_datetime(tl.timestamp).dt.tz_localize(None).dt.to_period("M").astype(str)
        tl["category"] = tl.mcc.map(MCC_CATEGORY).fillna("shopping")
        prods = {}
        for p, ts in products_from_timeline(c["timeline"]):
            prods.setdefault(p, str(ts.to_period("M")))
        specs.append({
            "client_id": c["client_id"], "as_of": str(as_of.date()),
            "expenses": {m: g.groupby("category").amount.sum().to_dict() for m, g in tl.groupby("month")},
            "products": [(p, since, PRODUCT_FAMILY.get(p, "")) for p, since in prods.items()],
            "transactions": c["timeline"], "needs": needs, "life_events": events,
        })
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kbc", default=MOCK_FILE, help="KBC-schema JSON to include ('' to skip)")
    ap.add_argument("--limit", type=int, help="only the first N simulated customers (0 = none)")
    ap.add_argument("-o", "--output", default="report/financial_report.csv")
    a = ap.parse_args()
    from timesync_predict import IntentPredictor
    predictor = IntentPredictor()
    specs = [] if a.limit == 0 else sim_specs(predictor, a.limit)
    if a.kbc:
        specs += kbc_specs(predictor, a.kbc)
    rows = write_csv(specs, a.output)
    df = pd.DataFrame(rows)
    print(f"wrote {a.output}: {len(df):,} rows, {df.client_id.nunique():,} clients")
    print(df.groupby("section").size().to_string())


if __name__ == "__main__":
    main()
