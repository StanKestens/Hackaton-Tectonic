"""Step 1: synthetic multi-channel customer data, shaped like the KBC mock data.

Latent life events (buying a car, a house, an e-bike, a baby, starting to invest) drive
everything: a few months of warm-up signals (spend shift, app searches), intents, raw
transactions in the KBC schema (merchant/mcc/amount/type/memo) and product uptake.

Outputs (data/):
  customers.csv       customer, persona, age, income, sparse
  products.csv        customer, product, start_month, timestamp
  transactions.csv    monthly spend per category        (tabular LightGBM baseline)
  events.csv          flattened multi-domain event stream (TIMeSynC encoder input)
  intents.csv         timestamped intents                (TIMeSynC decoder input/target)
  balances.csv        month-end current/savings/investments/debt per customer
  holdings.csv        investment positions (units) + prices.csv monthly prices per instrument
  raw_transactions.csv  all notable raw transactions in the KBC schema
  sample_timelines.json  a few simulated customers in the KBC mock JSON schema
Run: python simulate.py
"""
import json
import os

import numpy as np
import pandas as pd

from config import *
from flatten import EVENT_COLUMNS, MCC_CATEGORY, flatten_transaction

T0 = pd.Timestamp(START_DATE)
DAY = pd.Timedelta(days=1)


def month_start(m):
    return T0 + pd.DateOffset(months=int(m))


def rand_time_in_month(m, rng):
    return month_start(m) + pd.Timedelta(days=float(rng.uniform(0, 27)), hours=float(rng.uniform(7, 22)))


def band(x, edges):
    return str(int(np.searchsorted(edges, x)))


def make_prices(rng):
    """Monthly price path per instrument (geometric random walk)."""
    rows = []
    for item, _, _, mu, vol in INSTRUMENTS:
        p = float(rng.uniform(20, 400))
        for m in range(N_MONTHS):
            rows.append((item, m, round(p, 2)))
            p *= float(np.exp((mu - vol ** 2 / 2) / 12 + vol / np.sqrt(12) * rng.normal()))
    return pd.DataFrame(rows, columns=["item", "month", "price"])


def simulate_customer(cid, rng, prices):
    persona = str(rng.choice(list(PERSONAS)))
    cfg = PERSONAS[persona]
    age = int(rng.integers(*cfg["age"]))
    income = float(rng.uniform(*cfg["income"]))
    sparse = bool(rng.random() < SPARSE_CONTEXT_RATE)
    end = month_start(N_MONTHS)

    events, intents, raw_txs, products = [], [], [], {}
    mult = np.ones((N_MONTHS, len(CATEGORIES)))
    extra = np.zeros((N_MONTHS, len(CATEGORIES)))   # big one-off purchases added to monthly totals
    paid_from_savings = np.zeros(N_MONTHS)           # big purchases drain the savings account
    invest_in = []                                    # (month, amount) money moved into funds

    def add_product(p, ts):
        if p not in products or ts < products[p]:
            products[p] = ts

    # static profile (always visible to the model)
    events.append((cid, T0 - DAY, "profile", "age_band", band(age, [25, 35, 45, 55, 65]), "NONE"))
    events.append((cid, T0 - DAY, "profile", "income_band", band(income, [1500, 2500, 4000, 6000, 9000]), "NONE"))
    for p, prob in cfg["base_products"].items():
        if rng.random() < prob:
            add_product(p, T0 - DAY)

    # ---- life events ----
    for ev_name, prob in cfg["events"].items():
        ev = LIFE_EVENTS[ev_name]
        real = rng.random() < prob
        decoy = (not real) and rng.random() < DECOY_RATE * prob * 2
        if not (real or decoy):
            continue
        lead = ev.get("lead_months", LEAD)
        m = int(rng.integers(lead, N_MONTHS))
        T = rand_time_in_month(m, rng)

        # warm-up: spend shift in the lead months before, and app searches
        for cat, f in ev["spend_shift"].items():
            mult[m - lead:m, CATEGORIES.index(cat)] *= f
        first_signal = T - pd.Timedelta(days=float(rng.uniform(20, lead * 30)))
        for _ in range(int(rng.integers(1, 4))):
            ts = first_signal + (T - first_signal) * float(rng.uniform(0, 0.9))
            events.append((cid, ts, "digital", "search", str(rng.choice(ev["searches"])), "NONE"))
        # research-phase intent (e.g. BUY_CAR: simulating a car loan in the app)
        if real or rng.random() < 0.5:
            ts = first_signal + (T - first_signal) * float(rng.uniform(0.3, 0.95))
            intents.append((cid, ts, ev["intents"][0], ev_name))
        if not real:
            continue

        # the event itself: raw transactions in KBC schema
        offsets = [t[0] for t in ev["txs"]]
        for off, merchant, mcc, (lo, hi), ttype, memo in ev["txs"]:
            ts = T + pd.Timedelta(days=off, minutes=float(rng.uniform(0, 600)))
            tx = {"tx_id": f"TX-{cid}-{len(raw_txs)}", "timestamp": ts.isoformat() + "Z",
                  "merchant": merchant, "mcc": mcc, "amount": round(float(rng.uniform(lo, hi)), 2),
                  "currency": "EUR", "type": ttype, "memo": memo}
            raw_txs.append(tx)
            mm = (ts.year - T0.year) * 12 + ts.month - T0.month
            if 0 <= mm < N_MONTHS:
                extra[mm, CATEGORIES.index(MCC_CATEGORY.get(mcc, "shopping"))] += tx["amount"]
                if ttype != "Direct Debit" and tx["amount"] > 1000:
                    paid_from_savings[mm] += tx["amount"]
                if mcc == 6211:
                    invest_in.append((mm, tx["amount"]))

        # follow-up intent right after the purchase (e.g. INSURE_CAR), before the policy starts
        if len(ev["intents"]) > 1:
            gap = min([o for o in offsets if o > 0], default=20)
            intents.append((cid, T + DAY * float(rng.uniform(0.05, 0.9) * gap), ev["intents"][1], ev_name))
        if len(ev["intents"]) > 2 and rng.random() < 0.6:
            intents.append((cid, T + DAY * float(offsets[-1] + rng.uniform(3, 30)), ev["intents"][2], ev_name))

        for p, off in ev["products"].items():
            if rng.random() < ev["optional"].get(p, 1.0):
                add_product(p, T + pd.Timedelta(days=off, hours=12))

    # ---- monthly spending: low-resolution stream, available at the start of next month ----
    amounts = (np.array([BASE_SHARE[c] for c in CATEGORIES]) * income * mult
               * rng.lognormal(0.0, 0.15, size=mult.shape)) + extra
    monthly = pd.DataFrame(amounts, columns=CATEGORIES)
    monthly["month"] = np.arange(N_MONTHS)
    monthly["customer"] = cid
    for m in range(N_MONTHS):
        for j, cat in enumerate(CATEGORIES):
            events.append((cid, month_start(m + 1), "monthly_spend", cat, round(float(amounts[m, j]), 2), "NONE"))

    # ---- balances (month-end) and investment holdings ----
    sav_rng, inv_share, pf_rng = WEALTH[persona]
    item_meta = {i[0]: i for i in INSTRUMENTS}
    px = prices.pivot(index="month", columns="item", values="price")
    holdings = []                                            # (item, units, start_month)
    if rng.random() < inv_share:
        value = income * float(rng.uniform(*pf_rng))
        picks = rng.choice(len(INSTRUMENTS), size=int(rng.integers(2, 7)), replace=False)
        for k, w in zip(picks, rng.dirichlet(np.ones(len(picks)))):
            item = INSTRUMENTS[k][0]
            holdings.append((item, value * w / px.loc[0, item], 0))
    for mm, amt in invest_in:                                # wealth_building: buys a KBC fund
        item = str(rng.choice(["fund:KBC_ECO", "fund:KBC_TECH", "fund:KBC_PENSION"]))
        holdings.append((item, amt / px.loc[mm, item], mm))
    loans = []                                               # (start_month, principal, monthly repayment share)
    for p, ts in products.items():
        mm = (ts.year - T0.year) * 12 + ts.month - T0.month
        if p == "home_loan":
            loans.append((mm, float(rng.uniform(150_000, 350_000)), 1 / 240))
        elif p == "car_loan":
            loans.append((mm, float(rng.uniform(8_000, 25_000)), 1 / 60))
    savings = income * float(rng.uniform(*sav_rng))
    bal_rows = []
    for m in range(N_MONTHS):
        savings = max(0.0, savings + amounts[m, CATEGORIES.index("savings")] - paid_from_savings[m]
                      - sum(a for mm, a in invest_in if mm == m))
        current = income * float(rng.uniform(0.2, 1.0)) - max(0.0, amounts[m].sum() - income) * 0.5
        inv = sum(u * px.loc[m, it] for it, u, start in holdings if m >= start)
        debt = sum(pr * max(0.0, 1 - rate * (m - st)) for st, pr, rate in loans if m >= st)
        bal_rows.append((cid, m, round(current, 2), round(savings, 2), round(inv, 2), round(debt, 2)))
        # low-resolution balance stream for the model (known at the start of next month)
        events.append((cid, month_start(m + 1), "balance", "savings_account", round(savings, 2), "NONE"))
        events.append((cid, month_start(m + 1), "balance", "investments", round(inv, 2), "NONE"))
    hold_rows = [(cid, it, item_meta[it][1], item_meta[it][2], round(u, 4), st) for it, u, st in holdings]

    # ---- digital logins, routine intents, service contacts ----
    for m in range(N_MONTHS):
        spend_ratio = amounts[m].sum() / income
        for _ in range(int(rng.integers(1, 5))):
            ts = rand_time_in_month(m, rng)
            events.append((cid, ts, "digital", "login", str(rng.choice(["mobile", "mobile", "web"])), "NONE"))
            if rng.random() < 0.5:
                w = np.array([4, 4, 1, 1, 3 if spend_ratio > 1.05 else 0.3, 0.2])
                intent = str(rng.choice(ROUTINE_INTENTS, p=w / w.sum()))
                intents.append((cid, ts + pd.Timedelta(minutes=float(rng.uniform(1, 20))), intent, ""))
        if rng.random() < 0.07:
            events.append((cid, rand_time_in_month(m, rng), "service", "support_contact",
                           str(rng.choice(["payment_problem", "card_question", "loan_question"])), "NONE"))

    # raw transactions + product enrollments into the flat stream
    for tx in raw_txs:
        events += flatten_transaction(cid, tx)
    for p, ts in products.items():
        events.append((cid, ts, "product", "enrolled", p, p))

    if sparse:   # like the KBC mock: only transactions/products/profile are known
        events = [e for e in events if e[2] not in ("monthly_spend", "digital", "service", "balance")]

    events = [e for e in events if e[1] < end]
    intents = [i for i in intents if i[1] < end]
    prod_rows = [(cid, p, int((ts.year - T0.year) * 12 + ts.month - T0.month), ts)
                 for p, ts in products.items() if ts < end]
    cust = {"customer": cid, "persona": persona, "age": age, "income": income, "sparse": sparse}
    return cust, prod_rows, monthly, events, intents, raw_txs, bal_rows, hold_rows


def simulate():
    rng = np.random.default_rng(SEED)
    prices = make_prices(rng)
    custs, prods, monthly, events, intents, samples, bals, holds, raws = [], [], [], [], [], [], [], [], []
    for n in range(N_CUSTOMERS):
        cid = f"SIM{n:05d}"
        c, p, mo, ev, it, raw, bal, hold = simulate_customer(cid, rng, prices)
        custs.append(c); prods += p; monthly.append(mo); events += ev; intents += it; bals += bal; holds += hold
        raws += [{"customer": cid, **t} for t in raw]
        if n < 30 and raw:
            samples.append({"client_id": cid, "timeline": sorted(raw, key=lambda t: t["timestamp"])})

    os.makedirs(DATA_DIR, exist_ok=True)
    customers = pd.DataFrame(custs)
    products = pd.DataFrame(prods, columns=["customer", "product", "start_month", "timestamp"])
    tx = pd.concat(monthly).melt(id_vars=["customer", "month"], var_name="category", value_name="amount")
    ev = pd.DataFrame(events, columns=EVENT_COLUMNS).sort_values(["customer", "timestamp"], kind="stable")
    it = pd.DataFrame(intents, columns=["customer", "timestamp", "intent", "life_event"]).sort_values(
        ["customer", "timestamp"], kind="stable")

    customers.to_csv(os.path.join(DATA_DIR, "customers.csv"), index=False)
    products.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    tx.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
    ev.to_csv(os.path.join(DATA_DIR, "events.csv"), index=False)
    it.to_csv(os.path.join(DATA_DIR, "intents.csv"), index=False)
    pd.DataFrame(bals, columns=["customer", "month", "current_account", "savings_account", "investments", "debt"]).to_csv(
        os.path.join(DATA_DIR, "balances.csv"), index=False)
    pd.DataFrame(holds, columns=["customer", "item", "label", "asset_class", "units", "start_month"]).to_csv(
        os.path.join(DATA_DIR, "holdings.csv"), index=False)
    prices.to_csv(os.path.join(DATA_DIR, "prices.csv"), index=False)
    pd.DataFrame(raws).to_csv(os.path.join(DATA_DIR, "raw_transactions.csv"), index=False)
    with open(os.path.join(DATA_DIR, "sample_timelines.json"), "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"customers {len(customers):,} | products {len(products):,} | monthly rows {len(tx):,} | "
          f"events {len(ev):,} | intents {len(it):,}")
    print(it.intent.value_counts().to_string())


if __name__ == "__main__":
    simulate()
