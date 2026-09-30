"""Builds the long-format financial report CSV (schema: report_schema.py) from a per-client spec.

A spec is a plain dict of actual data + model predictions. Every key is optional: sections without
data are simply left out (e.g. KBC mock clients only have transactions). Derived numbers
(summaries, cashflow, allocation, net worth) are computed here, so the mock and the real
pipeline produce exactly the same shape.

spec keys:
  client_id, as_of                      str
  profile        {key: value}
  income         [(period, amount)]
  balances       {current_account, savings_account, total_debt}      (investments come from holdings)
  balance_history[(period, current, savings, investments, debt)]
  expenses       {period: {category: amount}}
  holdings       [(item, label, asset_class, units, price)]
  products       [(product, since, family)]
  transactions   [{timestamp, merchant, mcc, amount, type, memo}]
  life_events    [(event, label, probability, evidence)]
  needs          [(need, label, probability, evidence, life_event)]
  propensity     [(product, label, probability, evidence)]
"""
import csv
import os

import numpy as np

from report_schema import COLUMNS

LABELS = {"housing": "Housing", "food": "Groceries & food", "transport": "Transport",
          "utilities": "Utilities & insurance", "health": "Health", "leisure": "Leisure & travel",
          "shopping": "Shopping", "savings": "Savings transfers", "total": "Total spending"}
PRODUCT_FAMILY = {"savings_account": "savings", "credit_card": "payments", "car_loan": "loans", "home_loan": "loans",
                  "car_insurance": "insurance", "home_insurance": "insurance", "life_insurance": "insurance",
                  "bike_insurance": "insurance", "investment_account": "investments"}


def nice(key):
    return key.replace("_", " ").capitalize()


def build_rows(spec):
    R = []

    def add(section, item, source, **kw):
        r = {c: "" for c in COLUMNS}
        r.update(client_id=spec["client_id"], as_of=spec["as_of"], section=section, item=item, source=source)
        for k, v in kw.items():
            if isinstance(v, (float, np.floating)):
                v = round(float(v), 4 if k == "probability" else 2)
            r[k] = "" if v is None else v
        R.append(r)

    for k, v in spec.get("profile", {}).items():
        add("profile", k, "actual", label=nice(k), value=v, unit="text")

    avg_inc = None
    if spec.get("income"):
        for period, amt in spec["income"]:
            add("income", "net_income", "actual", label="Net income", value=float(amt), unit="EUR", period=period)
        avg_inc = float(np.mean([a for _, a in spec["income"]]))
        add("income", "avg_monthly_income", "derived", label="Average monthly income", value=avg_inc, unit="EUR")

    inv_total = sum(u * p for _, _, _, u, p in spec.get("holdings", []))
    if "balances" in spec:
        b = spec["balances"]
        for k in ("current_account", "savings_account", "total_debt"):
            add("balances", k, "actual", label=nice(k), value=float(b[k]), unit="EUR")
        add("balances", "investments", "actual", label="Investments", value=inv_total, unit="EUR")
        assets = b["current_account"] + b["savings_account"] + inv_total
        add("balances", "total_assets", "derived", label="Total assets", value=assets, unit="EUR")
        add("balances", "net_worth", "derived", label="Net worth", value=assets - b["total_debt"], unit="EUR")

    for period, cur, sav, inv, debt in spec.get("balance_history", []):
        for k, v in (("current_account", cur), ("savings_account", sav), ("investments", inv), ("total_debt", debt)):
            add("balance_history", k, "actual", label=nice(k), value=float(v), unit="EUR", period=period)

    exp = spec.get("expenses")
    if exp:
        periods = sorted(exp)
        cats = sorted({c for p in exp.values() for c in p})
        for p in periods:
            for c in cats:
                add("monthly_expenses", c, "actual", label=LABELS.get(c, nice(c)), value=float(exp[p].get(c, 0.0)),
                    unit="EUR", period=p, category=c)
            add("monthly_expenses", "total", "actual", label="Total spending", value=float(sum(exp[p].values())),
                unit="EUR", period=p, category="total")
        last3, prev3 = periods[-3:], periods[-6:-3]
        for c in cats + ["total"]:
            get = (lambda p: sum(exp[p].values())) if c == "total" else (lambda p, c=c: exp[p].get(c, 0.0))
            a3 = float(np.mean([get(p) for p in last3]))
            add("expense_summary", f"{c}_avg_3m", "derived", label=f"{LABELS.get(c, nice(c))}: average last 3 months",
                value=a3, unit="EUR", category=c)
            if prev3:
                p3 = float(np.mean([get(p) for p in prev3]))
                add("expense_summary", f"{c}_change_pct", "derived", label=f"{LABELS.get(c, nice(c))}: change vs previous 3 months",
                    value=(a3 - p3) / p3 * 100 if p3 > 0 else 0.0, unit="pct", category=c)
            if avg_inc:
                add("expense_summary", f"{c}_share_of_income_pct", "derived", label=f"{LABELS.get(c, nice(c))}: share of income",
                    value=a3 / avg_inc * 100, unit="pct", category=c)
        if avg_inc:
            spend = float(np.mean([sum(v for k, v in exp[p].items() if k != "savings") for p in last3]))
            add("cashflow", "avg_monthly_surplus", "derived", label="Average monthly surplus (income - spending)",
                value=avg_inc - spend, unit="EUR")
            add("cashflow", "savings_rate_pct", "derived", label="Savings rate", value=(avg_inc - spend) / avg_inc * 100, unit="pct")
            if "balances" in spec:
                liquid = spec["balances"]["current_account"] + spec["balances"]["savings_account"]
                add("cashflow", "emergency_buffer_months", "derived", label="Liquid savings / monthly spending",
                    value=liquid / max(spend, 1.0), unit="months")

    for item, label, ac, units, price in spec.get("holdings", []):
        add("investments", item, "actual", label=label, value=units * price, unit="EUR", category=ac,
            evidence=f"{units:.4f} units @ {price:.2f}")
    for ac in sorted({h[2] for h in spec.get("holdings", [])}):
        v = sum(u * p for _, _, a, u, p in spec["holdings"] if a == ac)
        add("investments", f"allocation_{ac}_pct", "derived", label=f"Share of portfolio: {ac}",
            value=v / inv_total * 100 if inv_total else 0.0, unit="pct", category=ac)

    for product, since, family in spec.get("products", []):
        add("products_held", product, "actual", label=nice(product), value="active", unit="text", period=since,
            category=family)

    for tx in spec.get("transactions", []):
        add("recent_transactions", tx.get("tx_id", ""), "actual", label=f"{tx['merchant']}: {tx['memo']}",
            value=float(tx["amount"]), unit="EUR", period=str(tx["timestamp"])[:10], category=f"mcc_{tx['mcc']}",
            evidence=tx["type"])

    for rank, (event, label, prob, ev) in enumerate(spec.get("life_events", []), 1):
        add("life_events", event, "model:timesync", label=label, value=prob * 100, unit="pct", category=event,
            rank=rank, probability=prob, evidence=ev)
    for rank, (need, label, prob, ev, event) in enumerate(spec.get("needs", []), 1):
        add("next_needs_30d", need, "model:timesync", label=label, value=prob * 100, unit="pct", category=event,
            rank=rank, probability=prob, evidence=ev)
    for rank, (product, label, prob, ev) in enumerate(spec.get("propensity", []), 1):
        add("product_propensity", product, "model:lightgbm", label=label, value=prob * 100, unit="pct",
            category=PRODUCT_FAMILY.get(product, ""), rank=rank, probability=prob, evidence=ev)
    return R


def write_csv(specs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [r for s in specs for r in build_rows(s)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows
