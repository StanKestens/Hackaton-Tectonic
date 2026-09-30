# Financial report: data contract

Files:
- `financial_report_mock.csv`: hand-written mock, 6 clients, to build against now
- `financial_report.csv`: the real output of `python make_report.py` (simulated customers + KBC sample), same format

The report contains **actual data and model predictions only**. What to show, and how, is up to the website.

**Format:** one long CSV, one row per fact. Filter on `client_id` + `section`, then use `item` as the key.
New facts are added as new rows (never new columns), so parsers won't break.

## Columns
| column | meaning |
|---|---|
| `client_id` | customer id |
| `as_of` | date the report describes. Predictions look forward from this date |
| `section` | block of the report (see below) |
| `item` | stable key inside the section, e.g. `stock:ASML`, `savings_account`, `PREPARE_FOR_BABY` |
| `label` | readable name (English for now) |
| `value` | number or text |
| `unit` | `EUR`, `pct` (0-100), `months`, `text` |
| `period` | `YYYY-MM` for monthly series, `YYYY-MM-DD` for transactions and product start dates |
| `category` | grouping: spending category, asset class (`stocks`/`etf`/`funds`/`bonds`), product family, life event, `mcc_<code>` |
| `rank` | 1 = most likely (predictions) |
| `probability` | 0-1, model predictions only |
| `source` | `actual`, `derived` (computed from actual data), `model:timesync`, `model:lightgbm` |
| `evidence` | for predictions: the data points that drove them; for holdings: units @ price; for transactions: payment type |

## Sections
| section | source | content |
|---|---|---|
| `profile` | actual | segment, age / age band, income band |
| `income` | actual + derived | `net_income` per month, `avg_monthly_income` |
| `balances` | actual + derived | current_account, savings_account, investments, total_debt, total_assets, net_worth |
| `balance_history` | actual | the same balances at each month end, last 12 months |
| `monthly_expenses` | actual | spending per category per month (last 12 months) + `total` |
| `expense_summary` | derived | `<category>_avg_3m`, `<category>_change_pct` (vs the 3 months before), `<category>_share_of_income_pct` |
| `cashflow` | derived | avg_monthly_surplus, savings_rate_pct, emergency_buffer_months |
| `investments` | actual + derived | one row per holding (value in EUR, `category` = asset class) + `allocation_<class>_pct` |
| `products_held` | actual | KBC products owned (`period` = since) |
| `recent_transactions` | actual | notable transactions of the last 90 days (merchant/memo in `label`, amount in `value`) |
| `life_events` | model:timesync | probability that each life event plays out in the next 30 days: `family_expansion` (baby on the way), `automotive_lifecycle`, `real_estate_acquisition`, `micro_mobility_acquisition`, `wealth_building` |
| `next_needs_30d` | model:timesync | top 5 needs in the next 30 days, e.g. `PREPARE_FOR_BABY`, `INSURE_CAR`, `RENOVATION_LOAN`, `INVEST` |
| `product_propensity` | model:lightgbm | probability of taking each not-yet-owned KBC product in the next 3 months |

Customers from KBC transaction JSON only have `monthly_expenses`, `products_held`, `recent_transactions` and the model sections.
The other sections need balance and income data.

## Mock clients
| id | story |
|---|---|
| BE9101 | just bought a car (KBC sample) |
| BE4402 | just bought a house (KBC sample) |
| BE5505 | student, bought an e-bike (KBC sample) |
| BE7701 | affluent, ~EUR 257k invested, half in individual stocks |
| BE3303 | baby on the way |
| BE6606 | student, negative balance, overspending |

Example (pandas): `df[(df.client_id=="BE7701") & (df.section=="investments")]`
