# simulate.py

import os
import numpy as np
import pandas as pd

from config import (
    SEED,
    DATA_DIR,
    N_CUSTOMERS,
    N_MONTHS,
    LEAD,
    DECOY_RATE,
    CATEGORIES,
    BASE_SHARE,
    PERSONAS,
    SIGNALS,
    PRODUCTS,
)


CHANNELS = ["mobile", "web", "call_center", "branch", "email"]

DOMAINS = [
    "transaction",
    "product",
    "digital",
    "service",
    "context",
]

EVENT_TYPES = [
    "transaction_summary",
    "product_view",
    "product_uptake",
    "login",
    "search",
    "support_contact",
    "message",
]

INTENTS = [
    "NO_INTENT",
    "SAVE_MONEY",
    "MANAGE_CASHFLOW",
    "BORROW_MONEY",
    "BUY_HOME",
    "BUY_CAR",
    "INVEST",
    "PROTECT_FAMILY",
    "REDUCE_SPENDING",
    "GET_SUPPORT",
]


def month_timestamp(month: int, rng: np.random.Generator) -> pd.Timestamp:
    """Generate a timestamp inside a synthetic month."""
    base = pd.Timestamp("2023-01-01") + pd.DateOffset(months=int(month))
    day = int(rng.integers(1, 27))
    hour = int(rng.integers(7, 23))
    minute = int(rng.integers(0, 60))
    return base + pd.Timedelta(days=day - 1, hours=hour, minutes=minute)


def add_event(
    events,
    customer,
    timestamp,
    domain,
    field,
    value,
    channel,
    product="NONE",
):
    events.append(
        {
            "customer": customer,
            "timestamp": timestamp,
            "domain": domain,
            "field": field,
            "value": str(value),
            "channel": channel,
            "product": product,
        }
    )


def infer_intent(
    month_df: pd.DataFrame,
    income: float,
    active_products: set,
    uptake_this_month: list,
    rng: np.random.Generator,
) -> str:
    """
    Synthetic intent generator.

    In a real project, this label would come from:
    - explicit customer requests,
    - CRM outcomes,
    - advisor annotations,
    - product applications,
    - support topics,
    - or weak supervision.
    """

    category_totals = month_df.groupby("category")["amount"].sum().to_dict()
    savings = category_totals.get("savings", 0.0)
    housing = category_totals.get("housing", 0.0)
    transport = category_totals.get("transport", 0.0)
    shopping = category_totals.get("shopping", 0.0)

    if "home_loan" in uptake_this_month:
        return "BUY_HOME"

    if "car_loan" in uptake_this_month:
        return "BUY_CAR"

    if "investment_account" in uptake_this_month:
        return "INVEST"

    if "insurance" in uptake_this_month:
        return "PROTECT_FAMILY"

    if "savings_account" in uptake_this_month:
        return "SAVE_MONEY"

    if len(month_df) > 0:
        total_spend = month_df["amount"].sum()
        spending_ratio = total_spend / max(income, 1.0)

        if spending_ratio > 0.95:
            return "MANAGE_CASHFLOW"

        if shopping > 0.20 * income:
            return "REDUCE_SPENDING"

        if savings < 0.05 * income and rng.random() < 0.50:
            return "SAVE_MONEY"

        if transport > 0.14 * income and rng.random() < 0.30:
            return "BUY_CAR"

        if housing > 0.36 * income and rng.random() < 0.20:
            return "BUY_HOME"

    if rng.random() < 0.08:
        return "GET_SUPPORT"

    return "NO_INTENT"


def simulate():
    rng = np.random.default_rng(SEED)

    customer_rows = []
    product_rows = []
    transaction_frames = []
    event_rows = []
    intent_rows = []

    base_share = np.array([BASE_SHARE[c] for c in CATEGORIES])

    for customer_id in range(N_CUSTOMERS):
        persona = rng.choice(list(PERSONAS))
        persona_config = PERSONAS[persona]

        age = int(rng.integers(*persona_config["age"]))
        income = float(rng.uniform(*persona_config["income"]))

        customer_rows.append(
            {
                "customer": customer_id,
                "persona": persona,
                "age": age,
                "income": income,
            }
        )

        multipliers = np.ones((N_MONTHS, len(CATEGORIES)))
        customer_products = []
        uptake_by_month = {}

        # Generate static product enrollment and pre-uptake spending signals.
        for product, probability in persona_config["p"].items():
            real_uptake = rng.random() < probability
            decoy = (not real_uptake) and rng.random() < DECOY_RATE

            if not (real_uptake or decoy):
                continue

            uptake_month = int(rng.integers(LEAD, N_MONTHS))

            for category, multiplier in SIGNALS[product].items():
                category_index = CATEGORIES.index(category)
                multipliers[
                    uptake_month - LEAD : uptake_month,
                    category_index
                ] *= multiplier

            if real_uptake:
                customer_products.append(product)
                product_rows.append(
                    {
                        "customer": customer_id,
                        "product": product,
                        "start_month": uptake_month,
                    }
                )
                uptake_by_month.setdefault(uptake_month, []).append(product)

        # Generate monthly transaction summaries.
        amounts = (
            base_share
            * income
            * multipliers
            * rng.lognormal(0.0, 0.15, size=multipliers.shape)
        )

        transaction_df = pd.DataFrame(amounts, columns=CATEGORIES)
        transaction_df["month"] = np.arange(N_MONTHS)
        transaction_df["customer"] = customer_id

        transaction_frames.append(
            transaction_df.melt(
                id_vars=["customer", "month"],
                var_name="category",
                value_name="amount",
            )
        )

        # Generate synchronized, heterogeneous event stream.
        for month in range(N_MONTHS):
            month_df = transaction_df[transaction_df["month"] == month]
            timestamp = month_timestamp(month, rng)

            # Aggregate transaction event for each category.
            for _, row in month_df.iterrows():
                add_event(
                    events=event_rows,
                    customer=customer_id,
                    timestamp=timestamp,
                    domain="transaction",
                    field=row["category"],
                    value=round(float(row["amount"]), 2),
                    channel="mobile",
                )

            # Digital behavior.
            add_event(
                events=event_rows,
                customer=customer_id,
                timestamp=timestamp + pd.Timedelta(hours=1),
                domain="digital",
                field="login",
                value=rng.choice(["login", "login", "login", "failed_login"]),
                channel=rng.choice(["mobile", "web"]),
            )

            if rng.random() < 0.25:
                add_event(
                    events=event_rows,
                    customer=customer_id,
                    timestamp=timestamp + pd.Timedelta(hours=2),
                    domain="digital",
                    field="search",
                    value=rng.choice(
                        [
                            "savings",
                            "mortgage",
                            "loan",
                            "insurance",
                            "investment",
                            "budget",
                        ]
                    ),
                    channel=rng.choice(["mobile", "web"]),
                )

            # Product uptake event.
            for product in uptake_by_month.get(month, []):
                add_event(
                    events=event_rows,
                    customer=customer_id,
                    timestamp=timestamp + pd.Timedelta(hours=3),
                    domain="product",
                    field="uptake",
                    value="enrolled",
                    channel=rng.choice(["mobile", "web", "branch"]),
                    product=product,
                )

            # Product ownership context.
            for product in customer_products:
                if month >= next(
                    row["start_month"]
                    for row in product_rows
                    if row["customer"] == customer_id
                    and row["product"] == product
                ):
                    add_event(
                        events=event_rows,
                        customer=customer_id,
                        timestamp=timestamp + pd.Timedelta(hours=4),
                        domain="product",
                        field="owned_product",
                        value="active",
                        channel="mobile",
                        product=product,
                    )

            # Occasional customer-service interaction.
            if rng.random() < 0.07:
                add_event(
                    events=event_rows,
                    customer=customer_id,
                    timestamp=timestamp + pd.Timedelta(hours=5),
                    domain="service",
                    field="support_contact",
                    value=rng.choice(
                        [
                            "payment_problem",
                            "card_question",
                            "loan_question",
                            "account_question",
                        ]
                    ),
                    channel=rng.choice(["call_center", "branch", "chat"]),
                )

            # Generate an intent label for this month.
            intent = infer_intent(
                month_df=month_df,
                income=income,
                active_products=set(customer_products),
                uptake_this_month=uptake_by_month.get(month, []),
                rng=rng,
            )

            intent_rows.append(
                {
                    "customer": customer_id,
                    "month": month,
                    "timestamp": timestamp + pd.Timedelta(hours=6),
                    "intent": intent,
                }
            )

    customers = pd.DataFrame(customer_rows)
    products = pd.DataFrame(product_rows)
    transactions = pd.concat(transaction_frames, ignore_index=True)
    events = pd.DataFrame(event_rows)
    intents = pd.DataFrame(intent_rows)

    os.makedirs(DATA_DIR, exist_ok=True)

    customers.to_csv(os.path.join(DATA_DIR, "customers.csv"), index=False)
    products.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    transactions.to_csv(
        os.path.join(DATA_DIR, "transactions.csv"),
        index=False,
    )
    events.to_csv(os.path.join(DATA_DIR, "events.csv"), index=False)
    intents.to_csv(os.path.join(DATA_DIR, "intents.csv"), index=False)

    print(f"Saved customers: {len(customers):,}")
    print(f"Saved transactions: {len(transactions):,}")
    print(f"Saved products: {len(products):,}")
    print(f"Saved context events: {len(events):,}")
    print(f"Saved intent labels: {len(intents):,}")


if __name__ == "__main__":
    simulate()