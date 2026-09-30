"""Flattening of heterogeneous records into one event stream (TIMeSynC paper, Fig. 3).

Every raw record (a transaction, a login, a monthly spend summary, a product enrollment...)
becomes one or more rows:  (customer, timestamp, domain, field, value, product)
The model later turns (domain, field) into a field-name token, value into a (binned) value
token and product into a product token. Used by both simulate.py and the KBC mock loader,
so simulated and real-looking data go through exactly the same path.
"""
import pandas as pd

EVENT_COLUMNS = ["customer", "timestamp", "domain", "field", "value", "product"]

# numeric fields get quantile-binned; everything else is a categorical string token
NUMERIC_DOMAINS = {"monthly_spend", "balance"}
NUMERIC_FIELDS = {"amount"}

# domains that are "static" customer attributes: always visible to every query
STATIC_DOMAINS = {"profile"}

# MCC -> spending category (used for monthly aggregates / tabular baseline)
MCC_CATEGORY = {5511: "transport", 5541: "transport", 9311: "transport", 8111: "housing",
                6012: "housing", 6300: "utilities", 5940: "leisure", 8062: "health", 8011: "health", 5621: "shopping",
                5641: "shopping", 6211: "savings", 5411: "food"}

# memo / merchant keywords -> which bank product the transaction is about (order matters)
PRODUCT_KEYWORDS = [
    ("fiets", "bike_insurance"),
    ("hypothec", "home_loan"),
    ("woningpolis", "home_insurance"),
    ("brand en water", "home_insurance"),
    ("schuldsaldo", "life_insurance"),
    ("levensverzekering", "life_insurance"),
    ("autolening", "car_loan"),
    ("auto", "car_insurance"),
    ("omnium", "car_insurance"),
    ("beleggingsfonds", "investment_account"),
]


def is_numeric(domain, field):
    return domain in NUMERIC_DOMAINS or field in NUMERIC_FIELDS


def product_from_text(merchant, memo):
    text = f"{merchant} {memo}".lower()
    for kw, product in PRODUCT_KEYWORDS:
        if kw in text:
            return product
    return "NONE"


def flatten_transaction(customer, tx):
    """tx: dict in the KBC mock schema (timestamp, merchant, mcc, amount, type, memo)."""
    ts = pd.Timestamp(tx["timestamp"])
    if ts.tzinfo is not None:                    # everything internally is naive UTC
        ts = ts.tz_convert(None)
    product = product_from_text(tx.get("merchant", ""), tx.get("memo", ""))
    return [
        (customer, ts, "transaction", "mcc", str(int(tx["mcc"])), product),
        (customer, ts, "transaction", "amount", float(tx["amount"]), product),
        (customer, ts, "transaction", "type", str(tx["type"]), product),
    ]


def kbc_json_to_events(clients):
    """clients: parsed KBC mock JSON (list of {client_id, timeline:[...]}) -> events DataFrame."""
    rows = []
    for c in clients:
        for tx in c["timeline"]:
            rows += flatten_transaction(c["client_id"], tx)
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df
