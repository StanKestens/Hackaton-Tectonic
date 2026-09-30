"""Builds the long-format financial report CSV (schema: report_schema.py) from a per-client spec.

A spec is a plain dict with raw facts (income, balances, monthly expenses, holdings, products, model
predictions...). Everything derivable (summaries, cashflow, allocation, net worth) is computed here,
so the mock and the real pipeline produce exactly the same shape.
"""
import csv
import os

import numpy as np

from report_schema import COLUMNS

LABELS = {"housing": "Housing", "food": "Groceries & food", "transport": "Transport", "utilities": "Utilities & insurance",
          "health": "Health", "leisure": "Leisure & travel", "shopping": "Shopping", "childcare": "Children",
          "savings": "Savings transfers"}


def _row(spec, section, item, **kw):
    r = {c: "" for c in COLUMNS}
    r.update(client_id=spec["client_id"], as_of=spec["as_of"], section=section, item=item)
    r.update({k: ("" if v is None else v) for k, v in kw.items()})
    if isinstance(r["value"], float):
        r["value"] = round(r["value"], 2)
    if isinstance(r["probability"], float):
        r["probability"] = round(r["probability"], 3)
    return r


def build_rows(spec):
    R = []
    add = lambda section, item, **kw: R.append(_row(spec, section, item, **kw))

    # profile
    for k, v in spec["profile"].items():
        add("profile", k, label=k.replace("_", " ").capitalize(), value=v, unit="text", ui_hint="hidden")

    # income
    inc = spec["income"]                                  # [(period, amount)]
    for period, amt in inc:
        add("income", "net_income", label="Net income", value=float(amt), unit="EUR", period=period, ui_hint="chart")
    avg_inc = float(np.mean([a for _, a in inc]))
    add("income", "avg_monthly_income", label="Average monthly income", value=avg_inc, unit="EUR", ui_hint="card")

    # balances
    b = dict(spec["balances"])
    b["investments"] = float(sum(h[3] for h in spec["holdings"]))
    b["total_assets"] = b["current_account"] + b["savings_account"] + b["investments"]
    b["net_worth"] = b["total_assets"] - b["total_debt"]
    for k, v in b.items():
        add("balances", k, label=k.replace("_", " ").capitalize(), value=float(v), unit="EUR",
            ui_hint="hero" if k == "net_worth" else "card")

    # monthly expenses (time series)
    exp = spec["expenses"]                                # {period: {category: amount}}
    periods = sorted(exp)
    cats = sorted({c for p in exp.values() for c in p})
    for p in periods:
        for c in cats:
            add("monthly_expenses", c, label=LABELS.get(c, c), value=float(exp[p].get(c, 0.0)), unit="EUR",
                period=p, category=c, ui_hint="chart")
        add("monthly_expenses", "total", label="Total spending", value=float(sum(exp[p].values())), unit="EUR",
            period=p, category="total", ui_hint="chart")

    # expense summary: last 3 months vs 3 before
    last3, prev3 = periods[-3:], periods[-6:-3]
    for c in cats + ["total"]:
        get = (lambda p: sum(exp[p].values())) if c == "total" else (lambda p, c=c: exp[p].get(c, 0.0))
        a3, p3 = np.mean([get(p) for p in last3]), np.mean([get(p) for p in prev3]) if prev3 else np.nan
        chg = (a3 - p3) / p3 * 100 if prev3 and p3 > 0 else 0.0
        lab = LABELS.get(c, "Total spending")
        add("expense_summary", f"{c}_avg_3m", label=f"{lab}: avg last 3 months", value=float(a3), unit="EUR", category=c, ui_hint="list")
        add("expense_summary", f"{c}_change_pct", label=f"{lab}: change vs previous 3 months", value=float(chg), unit="pct", category=c,
            ui_hint="badge" if abs(chg) > 25 else "list")
        add("expense_summary", f"{c}_share_of_income_pct", label=f"{lab}: share of income", value=float(a3 / avg_inc * 100),
            unit="pct", category=c, ui_hint="list")

    # cashflow
    spend = np.mean([sum(exp[p].values()) - exp[p].get("savings", 0.0) for p in last3])
    surplus = avg_inc - spend
    add("cashflow", "avg_monthly_surplus", label="Average monthly surplus", value=float(surplus), unit="EUR", ui_hint="card")
    add("cashflow", "savings_rate_pct", label="Savings rate", value=float(surplus / avg_inc * 100), unit="pct", ui_hint="card")
    add("cashflow", "emergency_buffer_months", label="Emergency buffer",
        value=float((b["current_account"] + b["savings_account"]) / max(spend, 1)), unit="months", ui_hint="card")

    # investments
    total_inv = b["investments"]
    for item, label, asset_class, value in spec["holdings"]:
        add("investments", item, label=label, value=float(value), unit="EUR", category=asset_class, ui_hint="list")
    for ac in sorted({h[2] for h in spec["holdings"]}):
        v = sum(h[3] for h in spec["holdings"] if h[2] == ac)
        add("investments", f"allocation_{ac}_pct", label=f"Allocation: {ac}", value=float(v / total_inv * 100),
            unit="pct", category=ac, ui_hint="chart")

    for product, since, family in spec["products"]:
        add("products_held", product, label=product.replace("_", " ").capitalize(), value="active", unit="text",
            period=since, category=family, ui_hint="list")

    for event, label, prob, reason, stage in spec["life_events"]:
        add("life_events", event, label=label, value=stage, unit="text", category=event, probability=float(prob),
            reason=reason, ui_hint="hero" if prob >= 0.6 else "card")

    for i, (need, label, prob, reason, event) in enumerate(spec["needs"], 1):
        add("next_needs_30d", need, label=label, value=float(prob), unit="pct", category=event, rank=i,
            probability=float(prob), reason=reason, ui_hint="card" if i <= 2 else "list")

    for i, (item, label, cat, amount, prob, reason) in enumerate(spec["buys"][:5], 1):
        add("next_buys", item, label=label, value=float(amount), unit="EUR", category=cat, rank=i,
            probability=float(prob), reason=reason, ui_hint="card")

    for i, (product, label, prob, reason, family) in enumerate(spec["recommendations"], 1):
        add("recommendations", product, label=label, value=float(prob), unit="pct", category=family, rank=i,
            probability=float(prob), reason=reason, ui_hint="hero" if i == 1 else "card")

    for i, (module, label, reason) in enumerate(spec["modules"], 1):
        add("personalization", module, label=label, value="true", unit="bool", rank=i, reason=reason,
            ui_hint="hero" if i == 1 else "card")

    for item, label, value, unit, reason in spec["alerts"]:
        add("alerts", item, label=label, value=value, unit=unit, reason=reason, ui_hint="alert")
    return R


def write_csv(specs, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = [r for s in specs for r in build_rows(s)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows
