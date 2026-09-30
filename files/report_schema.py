"""The financial report contract between the models (this repo) and the frontend.

ONE long-format CSV, one row per fact. Consumers filter on `section` (and `item`).
Adding a new kind of fact = adding rows, never new columns.
Only facts and predictions: what to show and how is up to the website.
"""

COLUMNS = [
    "client_id",    # customer id
    "as_of",        # ISO date the report describes (predictions look forward from here)
    "section",      # which block of the report (see SECTIONS)
    "item",         # machine key inside the section (stable, snake_case / instrument id)
    "label",        # human-readable name (English for now)
    "value",        # the number or text value
    "unit",         # EUR | pct (0-100) | months | count | units | text | date
    "period",       # YYYY-MM for monthly series, YYYY-MM-DD for transactions, empty otherwise
    "category",     # grouping: spending category, asset class, product family, life event
    "rank",         # 1 = most likely, for ranked predictions
    "probability",  # 0-1, for model predictions only
    "source",       # actual | derived | model:timesync | model:lightgbm
    "evidence",     # for predictions: the data points that drove it
]

SECTIONS = {
    "profile":             "actual: segment, age, income band",
    "income":              "actual: monthly net income + derived average",
    "balances":            "actual: current/savings/investments/debt snapshot + derived total assets and net worth",
    "balance_history":     "actual: month-end balances per month",
    "monthly_expenses":    "actual: spending per category per month (last 12 months)",
    "expense_summary":     "derived: per category 3-month average, change vs previous 3 months, share of income",
    "cashflow":            "derived: monthly surplus, savings rate, emergency buffer in months",
    "investments":         "actual: holdings (units, price, value) + derived allocation per asset class",
    "products_held":       "actual: KBC products owned, with start month",
    "recent_transactions": "actual: notable transactions of the last 90 days (KBC schema)",
    "life_events":         "model:timesync: probability that each life event is happening in the next 30 days",
    "next_needs_30d":      "model:timesync: ranked needs in the next 30 days",
    "product_propensity":  "model:lightgbm: probability of taking each not-yet-owned product in the next 3 months",
}
