"""Generate a MOCK financial report (report/financial_report_mock.csv) for the frontend team.

Hand-written personas, so the frontend can build every module before the models are wired in:
  BE9101  bought a car            (KBC mock)      BE7701  wealthy stock investor
  BE4402  bought a house          (KBC mock)      BE3303  baby on the way
  BE5505  bought an e-bike        (KBC mock)      BE6606  student, cashflow stress
The real pipeline produces exactly the same columns/sections (see report_schema.py).
Run: python make_mock_report.py
"""
import numpy as np
import pandas as pd

from financial_report import write_csv

rng = np.random.default_rng(7)
SHARE = {"housing": 0.30, "food": 0.12, "transport": 0.08, "utilities": 0.08, "health": 0.04,
         "leisure": 0.08, "shopping": 0.10, "savings": 0.12}


def months_before(as_of, n=12):
    end = pd.Timestamp(as_of).to_period("M")
    return [str(end - i) for i in range(n, 0, -1)]


def expenses(as_of, income, tweak=None, spikes=None, extra_cats=None):
    """12 months of spending; tweak={cat: (from_month_idx, factor)}, spikes={(month_idx, cat): amount}."""
    out = {}
    share = {**SHARE, **(extra_cats or {})}
    for i, p in enumerate(months_before(as_of)):
        row = {}
        for c, s in share.items():
            f = 1.0
            if tweak and c in tweak and i >= tweak[c][0]:
                f = tweak[c][1]
            row[c] = income * s * f * rng.lognormal(0, 0.08)
        for (mi, c), amt in (spikes or {}).items():
            if mi == i:
                row[c] = row.get(c, 0) + amt
        out[p] = row
    return out


def income(as_of, amount, bonus_month=None):
    return [(p, amount * (2 if i == bonus_month else 1) * rng.normal(1, 0.01))
            for i, p in enumerate(months_before(as_of))]


SPECS = [
    {
        "client_id": "BE9101", "as_of": "2026-04-14",
        "profile": {"segment": "young_professional", "age_band": "25-34", "household": "couple", "language": "nl"},
        "income": income("2026-04-14", 3400),
        "balances": {"current_account": 2100.0, "savings_account": 4200.0, "total_debt": 0.0},
        "expenses": expenses("2026-04-14", 3400, tweak={"transport": (9, 1.5)}, spikes={(11, "transport"): 18500 + 348.5}),
        "holdings": [("fund:kbc_eco_fund", "KBC Eco Fund", "funds", 1500.0)],
        "products": [("car_insurance", "2026-04", "insurance"), ("savings_account", "2021-06", "savings"),
                     ("credit_card", "2020-02", "payments")],
        "life_events": [("automotive_lifecycle", "Just bought a car", 0.97, "Garage Cardoen EUR 18,500 + BIV tax + car policy", "completed")],
        "needs": [("ROADSIDE_ASSISTANCE", "Roadside assistance", 0.58, "car bought 3 days ago, no assistance cover yet", "automotive_lifecycle"),
                  ("REBUILD_SAVINGS", "Rebuild savings buffer", 0.41, "savings dropped from 22.7k to 4.2k", "automotive_lifecycle"),
                  ("CAR_MAINTENANCE_BUDGET", "Car maintenance budget", 0.22, "second-hand car", "automotive_lifecycle"),
                  ("INVEST", "Start investing", 0.05, "", "wealth_building"),
                  ("BUY_BIKE", "Buy a bike", 0.03, "", "micro_mobility_acquisition")],
        "buys": [("winter_tyres", "Winter tyres", "transport", 650, 0.46, "new car, winter in 6 months"),
                 ("car_accessories", "Car accessories (phone mount, mats)", "shopping", 120, 0.40, "typical after car purchase"),
                 ("fuel", "Fuel (monthly)", "transport", 160, 0.95, "new car -> recurring fuel"),
                 ("car_wash_subscription", "Car wash subscription", "transport", 25, 0.18, ""),
                 ("parking_permit", "Residential parking permit", "transport", 60, 0.30, "new car registered at address")],
        "recommendations": [("kbc_roadside_assistance", "KBC Roadside Assistance", 0.58, "no assistance cover on new car", "insurance"),
                            ("savings_plan", "Automatic savings plan", 0.41, "rebuild buffer after car purchase", "savings"),
                            ("fuel_card", "KBC fuel card cashback", 0.20, "new recurring fuel spend", "payments")],
        "modules": [("car_hub", "Your car: insurance, assistance, costs", "car purchase detected"),
                    ("savings_goal", "Rebuild your buffer", "savings dropped 81%"),
                    ("budget_overview", "Monthly budget", "default")],
        "alerts": [("savings_drop", "Savings dropped strongly this month", -81.5, "pct", "car purchase EUR 18,500")],
    },
    {
        "client_id": "BE4402", "as_of": "2026-05-17",
        "profile": {"segment": "family", "age_band": "35-44", "household": "family", "language": "nl"},
        "income": income("2026-05-17", 5200),
        "balances": {"current_account": 3800.0, "savings_account": 9500.0, "total_debt": 285000.0},
        "expenses": expenses("2026-05-17", 5200, tweak={"housing": (11, 1.6), "savings": (8, 1.6)},
                             spikes={(10, "housing"): 12500}),
        "holdings": [("fund:kbc_pension_fund", "Pension savings fund", "funds", 14200.0)],
        "products": [("home_loan", "2026-05", "loans"), ("home_insurance", "2026-05", "insurance"),
                     ("life_insurance", "2026-05", "insurance"), ("savings_account", "2015-03", "savings")],
        "life_events": [("real_estate_acquisition", "Bought a home", 0.98, "notary EUR 12,500 + first mortgage payment", "completed")],
        "needs": [("RENOVATION_LOAN", "Renovation loan", 0.47, "new home, typical renovation within 6 months", "real_estate_acquisition"),
                  ("ENERGY_CONTRACT", "Energy / utilities setup", 0.39, "just moved", "real_estate_acquisition"),
                  ("FAMILY_INSURANCE", "Family liability insurance", 0.25, "family household, no family policy", "family_expansion"),
                  ("BUY_CAR", "Buy a car", 0.06, "", "automotive_lifecycle"),
                  ("INVEST", "Start investing", 0.04, "", "wealth_building")],
        "buys": [("furniture", "Furniture", "shopping", 3500, 0.62, "just moved into new home"),
                 ("diy_materials", "DIY / renovation materials", "housing", 1800, 0.55, "new home"),
                 ("appliances", "Kitchen appliances", "shopping", 1200, 0.41, ""),
                 ("moving_company", "Moving company", "housing", 900, 0.35, "notary deed signed"),
                 ("garden_tools", "Garden tools", "shopping", 300, 0.22, "")],
        "recommendations": [("renovation_loan", "KBC Renovation Loan (green)", 0.47, "new home, energy label improvements", "loans"),
                            ("family_insurance", "Family liability insurance", 0.25, "no family policy yet", "insurance"),
                            ("energy_scan", "Energy scan with KBC partner", 0.30, "new home", "services")],
        "modules": [("home_hub", "Your new home: loan, insurance, renovation", "home purchase detected"),
                    ("renovation_planner", "Renovation planner & loan simulator", "renovation need 47%"),
                    ("budget_overview", "Monthly budget with mortgage", "new fixed cost EUR 1,150")],
        "alerts": [("new_fixed_cost", "New monthly fixed cost: mortgage", 1150.0, "EUR", "mortgage started 2026-05")],
    },
    {
        "client_id": "BE5505", "as_of": "2026-08-03",
        "profile": {"segment": "student", "age_band": "18-24", "household": "single", "language": "nl"},
        "income": income("2026-08-03", 1400),
        "balances": {"current_account": 650.0, "savings_account": 1200.0, "total_debt": 0.0},
        "expenses": expenses("2026-08-03", 1400, tweak={"leisure": (9, 1.3)}, spikes={(11, "leisure"): 3299}),
        "holdings": [],
        "products": [("bike_insurance", "2026-08", "insurance"), ("savings_account", "2024-09", "savings")],
        "life_events": [("micro_mobility_acquisition", "Bought an e-bike", 0.95, "Fietsen De Geus EUR 3,299 + bike policy", "completed")],
        "needs": [("REBUILD_SAVINGS", "Rebuild savings", 0.52, "large purchase vs small buffer", "micro_mobility_acquisition"),
                  ("BUDGET_HELP", "Budget coaching", 0.33, "spending above income this month", "cashflow"),
                  ("BIKE_ACCESSORIES", "Bike accessories", 0.28, "new e-bike", "micro_mobility_acquisition"),
                  ("CARD_QUESTION", "Card / payment question", 0.10, "", "routine"),
                  ("BUY_CAR", "Buy a car", 0.02, "", "automotive_lifecycle")],
        "buys": [("bike_lock", "Heavy-duty bike lock", "shopping", 90, 0.55, "new e-bike, theft risk"),
                 ("helmet", "Helmet", "shopping", 70, 0.48, ""),
                 ("rain_gear", "Rain gear", "shopping", 60, 0.30, ""),
                 ("bike_service", "First bike service", "leisure", 80, 0.25, "service after ~500 km"),
                 ("phone_mount", "Phone mount", "shopping", 25, 0.20, "")],
        "recommendations": [("savings_plan", "Round-up savings", 0.52, "rebuild buffer, small amounts", "savings"),
                            ("budget_coach", "KBC budget coach", 0.33, "spent more than income", "services")],
        "modules": [("budget_coach", "Budget coach", "overspent this month"),
                    ("bike_hub", "Your e-bike: insurance and extras", "e-bike purchase detected"),
                    ("savings_goal", "Small savings goal", "")],
        "alerts": [("overspending", "You spent more than your income this month", 238.0, "pct", "e-bike EUR 3,299")],
    },
    {
        "client_id": "BE7701", "as_of": "2026-09-30",
        "profile": {"segment": "affluent", "age_band": "45-54", "household": "couple", "language": "nl"},
        "income": income("2026-09-30", 11500, bonus_month=5),
        "balances": {"current_account": 18500.0, "savings_account": 62000.0, "total_debt": 45000.0},
        "expenses": expenses("2026-09-30", 11500, tweak={"leisure": (8, 1.4)}),
        "holdings": [("stock:ASML", "ASML Holding", "stocks", 48200.0), ("stock:KBC", "KBC Group", "stocks", 31500.0),
                     ("stock:UCB", "UCB", "stocks", 22800.0), ("stock:AAPL", "Apple", "stocks", 27400.0),
                     ("etf:iwda", "iShares MSCI World ETF", "etf", 64000.0),
                     ("fund:kbc_equity_tech", "KBC Equity Fund Technology", "funds", 38000.0),
                     ("bond:be_olo_2034", "Belgian OLO 2034", "bonds", 25000.0)],
        "products": [("investment_account", "2012-01", "investments"), ("home_loan", "2010-06", "loans"),
                     ("credit_card", "2008-03", "payments"), ("life_insurance", "2014-09", "insurance")],
        "life_events": [("wealth_building", "Active investor", 0.93, "portfolio EUR 257k, monthly buys", "ongoing")],
        "needs": [("PORTFOLIO_REVIEW", "Portfolio review", 0.44, "tech stocks 50% of stock holdings", "wealth_building"),
                  ("PENSION_SAVING", "Tax-optimal pension saving", 0.38, "pension savings not maxed this year", "wealth_building"),
                  ("ESTATE_PLANNING", "Estate planning", 0.21, "age + wealth profile", "wealth_building"),
                  ("TRAVEL_INSURANCE", "Travel insurance", 0.19, "leisure/travel spend up 40%", "leisure"),
                  ("INVEST", "Invest bonus", 0.35, "bonus of EUR 11.5k received in April", "wealth_building")],
        "buys": [("stock_top_up", "Top-up ETF / stocks", "investments", 5000, 0.61, "monthly investing pattern"),
                 ("travel", "Holiday booking", "leisure", 4200, 0.45, "travel spend rising"),
                 ("electronics", "Electronics", "shopping", 1500, 0.20, ""),
                 ("wine", "Wine subscription", "food", 150, 0.18, ""),
                 ("car_lease", "Car lease renewal", "transport", 750, 0.12, "")],
        "recommendations": [("private_banking", "KBC Private Banking advisor", 0.44, "portfolio > EUR 250k", "investments"),
                            ("diversification", "Diversification: reduce tech concentration", 0.40, "ASML + AAPL + tech fund = 44% of portfolio", "investments"),
                            ("pension_fund", "Pension savings top-up (tax benefit)", 0.38, "EUR 1,020 tax room left", "investments")],
        "modules": [("stock_portfolio", "Your portfolio", "EUR 257k invested, 50% in stocks"),
                    ("market_news", "News on your holdings: ASML, KBC, UCB, Apple", "holds individual stocks"),
                    ("tax_optimizer", "Pension & tax optimizer", "pension room left")],
        "alerts": [("concentration_risk", "44% of portfolio in technology", 44.0, "pct", "ASML, Apple, tech fund")],
    },
    {
        "client_id": "BE3303", "as_of": "2026-09-30",
        "profile": {"segment": "young_professional", "age_band": "25-34", "household": "couple", "language": "nl"},
        "income": income("2026-09-30", 4800),
        "balances": {"current_account": 3100.0, "savings_account": 16800.0, "total_debt": 190000.0},
        "expenses": expenses("2026-09-30", 4800, tweak={"health": (7, 2.2), "shopping": (9, 1.4)},
                             spikes={(10, "shopping"): 480, (11, "shopping"): 950}, extra_cats={"childcare": 0.0}),
        "holdings": [("fund:kbc_balanced", "KBC Balanced Fund", "funds", 6200.0)],
        "products": [("home_loan", "2023-02", "loans"), ("home_insurance", "2023-02", "insurance"),
                     ("savings_account", "2018-10", "savings"), ("credit_card", "2019-01", "payments")],
        "life_events": [("family_expansion", "Baby on the way", 0.86, "gynaecologist visits since March, maternity store purchases, searched 'kinderbijslag'", "expected")],
        "needs": [("SAVE_FOR_CHILD", "Savings account for the baby", 0.54, "baby expected, no child savings yet", "family_expansion"),
                  ("PROTECT_FAMILY", "Life / family insurance", 0.48, "mortgage + new dependant", "family_expansion"),
                  ("HOSPITALIZATION_INSURANCE", "Hospitalisation insurance top-up", 0.36, "upcoming delivery", "family_expansion"),
                  ("BUY_CAR", "Bigger car", 0.18, "family growing", "automotive_lifecycle"),
                  ("BUDGET_HELP", "Family budget planning", 0.15, "", "cashflow")],
        "buys": [("stroller", "Stroller / pram", "childcare", 650, 0.72, "baby expected, no stroller purchase seen"),
                 ("crib", "Crib & nursery furniture", "childcare", 900, 0.66, "maternity store visits"),
                 ("car_seat", "Baby car seat", "childcare", 250, 0.61, "required from day one"),
                 ("baby_clothes", "Baby clothes starter pack", "childcare", 200, 0.58, ""),
                 ("diapers", "Diapers (monthly)", "childcare", 60, 0.55, "recurring after birth")],
        "recommendations": [("child_savings_account", "KBC Child Savings Account", 0.54, "baby on the way", "savings"),
                            ("life_insurance", "Life insurance / debt balance top-up", 0.48, "mortgage + new dependant", "insurance"),
                            ("hospitalization_plus", "Hospitalisation insurance Plus", 0.36, "delivery expected", "insurance")],
        "modules": [("baby_checklist", "Baby on the way: things you might need", "family expansion detected (86%)"),
                    ("family_budget", "Your budget with a baby", "childcare costs start soon"),
                    ("protect_family", "Protect your family", "mortgage + new dependant")],
        "alerts": [("upcoming_costs", "Expected extra monthly cost after birth", 420.0, "EUR", "childcare, diapers, food")],
    },
    {
        "client_id": "BE6606", "as_of": "2026-09-30",
        "profile": {"segment": "student", "age_band": "18-24", "household": "single", "language": "fr"},
        "income": income("2026-09-30", 1100),
        "balances": {"current_account": -85.0, "savings_account": 150.0, "total_debt": 600.0},
        "expenses": expenses("2026-09-30", 1100, tweak={"shopping": (9, 1.8), "leisure": (9, 1.5)}),
        "holdings": [],
        "products": [("credit_card", "2025-10", "payments")],
        "life_events": [("cashflow_stress", "Tight budget", 0.81, "spending > income 3 months in a row, account negative", "ongoing")],
        "needs": [("MANAGE_CASHFLOW", "Manage cashflow", 0.71, "negative balance", "cashflow"),
                  ("BUDGET_HELP", "Budget coaching", 0.52, "shopping +80%", "cashflow"),
                  ("STUDENT_LOAN", "Student loan info", 0.20, "", "loans"),
                  ("SAVE_MONEY", "Start small savings", 0.12, "", "savings"),
                  ("CARD_QUESTION", "Card question", 0.10, "", "routine")],
        "buys": [("groceries", "Groceries", "food", 180, 0.97, "recurring"),
                 ("textbooks", "Textbooks", "shopping", 150, 0.44, "academic year started"),
                 ("phone_plan", "Phone plan", "utilities", 20, 0.90, "recurring"),
                 ("public_transport", "Student transport pass", "transport", 50, 0.40, "September"),
                 ("going_out", "Going out", "leisure", 120, 0.60, "")],
        "recommendations": [("budget_coach", "KBC budget coach", 0.52, "3 months of overspending", "services"),
                            ("student_account", "Free student account + alerts", 0.40, "", "payments")],
        "modules": [("budget_coach", "Get your budget back on track", "negative balance"),
                    ("spending_insights", "Where does your money go?", "shopping +80%"),
                    ("savings_goal", "Tiny savings goal", "")],
        "alerts": [("negative_balance", "Your current account is negative", -85.0, "EUR", ""),
                   ("spending_spike", "Shopping spending up 80%", 80.0, "pct", "vs previous 3 months")],
    },
]


if __name__ == "__main__":
    rows = write_csv(SPECS, "report/financial_report_mock.csv")
    df = pd.DataFrame(rows)
    print(f"wrote report/financial_report_mock.csv: {len(df)} rows, {df.client_id.nunique()} clients")
    print(df.groupby("section").size().to_string())
