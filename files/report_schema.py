"""The financial report contract between the models (this repo) and the frontend.

ONE long-format CSV, one row per fact. The frontend filters on `section` (and `item`).
Adding a new kind of fact = adding rows, never new columns, so the frontend never breaks.
"""

COLUMNS = [
    "client_id",    # customer id
    "as_of",        # ISO date the report was generated for
    "section",      # which block of the report (see SECTIONS)
    "item",         # machine key inside the section (stable, snake_case)
    "label",        # human-readable text for the UI (English for now)
    "value",        # the number or text value
    "unit",         # EUR | pct | months | count | bool | text | date
    "period",       # YYYY-MM for time series, empty otherwise
    "category",     # grouping: spending category, asset class, product family, life event...
    "rank",         # 1 = most important, for ordered lists (needs, buys, recommendations, modules)
    "probability",  # 0-1 model confidence, for predictions only
    "reason",       # short explanation why (for "why am I seeing this?")
    "ui_hint",      # suggested UI element: hero | card | chart | list | badge | alert | hidden
]

SECTIONS = {
    "profile":           "who the customer is: segment, age band, household, language",
    "income":            "monthly net income time series + average",
    "balances":          "current/savings/investments/debt/net worth snapshot",
    "monthly_expenses":  "spend per category per month (last 12 months) -> charts",
    "expense_summary":   "per category: 3-month average, change vs previous 3 months, share of income",
    "cashflow":          "savings rate, monthly surplus, emergency buffer in months",
    "investments":       "holdings (stocks, funds, bonds) and allocation per asset class",
    "products_held":     "KBC products the customer already has",
    "life_events":       "detected life events (baby on the way, buying a car...) with probability",
    "next_needs_30d":    "TIMeSynC model: needs likely in the next 30 days, ranked",
    "next_buys":         "top 5 things the customer is likely to buy next, with estimated amount",
    "recommendations":   "KBC products/services to offer, ranked, with reason",
    "personalization":   "which website modules to show and in what order (hero first)",
    "alerts":            "things to flag: spending spikes, low buffer, upcoming big payments",
}
