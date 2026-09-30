# config.py -- single place for all simulation + model settings

SEED = 42
DATA_DIR = "data"
MODEL_DIR = "models"
MOCK_FILE = "data/kbc_mock.json"          # KBC-provided sample timelines
START_DATE = "2025-01-01"                 # month 0 of the simulation

N_CUSTOMERS = 3000
N_MONTHS = 24
LEAD = 3                                  # months of "warm-up" signals before a life event
DECOY_RATE = 0.25                         # customers who show signals but never follow through
SPARSE_CONTEXT_RATE = 0.35                # customers with only raw transactions (like the KBC mock)

# ---- monthly spending categories (low-resolution stream, used by both models) ----
CATEGORIES = ["housing", "food", "transport", "utilities", "health", "leisure", "shopping", "savings"]
BASE_SHARE = {"housing": 0.30, "food": 0.12, "transport": 0.08, "utilities": 0.08,
              "health": 0.05, "leisure": 0.08, "shopping": 0.12, "savings": 0.17}

PRODUCTS = ["savings_account", "credit_card", "car_loan", "car_insurance", "home_loan",
            "home_insurance", "life_insurance", "bike_insurance", "investment_account"]

# ---- life events: the latent "why" behind a customer's behaviour ----
# Each event: warm-up signals (searches, spend shift) -> intents -> raw transactions -> products.
# Transaction templates follow the KBC mock schema: (day offset, merchant, mcc, amount range, type, memo)
LIFE_EVENTS = {
    "automotive_lifecycle": {
        "searches": ["autolening", "autoverzekering", "tweedehands wagen"],
        "spend_shift": {"transport": 1.6, "savings": 0.9},
        "intents": ["BUY_CAR", "INSURE_CAR", "ROADSIDE_ASSISTANCE"],
        "txs": [
            (0, "Garage Cardoen", 5511, (8000, 35000), "SEPA Transfer", "Aankoop wagen"),
            (2, "Vlaamse Belastingdienst - BIV", 9311, (100, 900), "SEPA Transfer", "Belasting op inverkeerstelling"),
            (3, "KBC Verzekeringen", 6300, (40, 130), "Direct Debit", "Polis Auto BA + Omnium premie"),
        ],
        "products": {"car_insurance": 3, "car_loan": 0},     # product -> day offset
        "optional": {"car_loan": 0.5},                        # product -> probability it is taken
    },
    "real_estate_acquisition": {
        "searches": ["woonkrediet", "hypotheek simulatie", "notaris kosten"],
        "spend_shift": {"savings": 1.8, "leisure": 0.8},
        "intents": ["BUY_HOME", "INSURE_HOME", "RENOVATION_LOAN"],
        "txs": [
            (0, "Notariskantoor De Smet", 8111, (6000, 25000), "SEPA Transfer", "Provisie aktekosten aankoop woning"),
            (13, "KBC Bank NV", 6012, (700, 2200), "Direct Debit", "Maandelijkse aflossing hypothecair krediet"),
            (14, "KBC Verzekeringen", 6300, (25, 80), "Direct Debit", "Woningpolis brand en waterschade"),
            (14, "KBC Verzekeringen", 6300, (15, 60), "Direct Debit", "Schuldsaldoverzekering periodieke premie"),
        ],
        "products": {"home_loan": 13, "home_insurance": 14, "life_insurance": 14},
        "optional": {},
    },
    "micro_mobility_acquisition": {
        "searches": ["fietsverzekering", "e-bike lease", "fiets diefstal"],
        "spend_shift": {"leisure": 1.4},
        "intents": ["BUY_BIKE", "INSURE_BIKE"],
        "txs": [
            (0, "Fietsen De Geus", 5940, (900, 6000), "Bancontact", "E-Bike aankoop"),
            (1, "KBC Verzekeringen", 6300, (8, 25), "Direct Debit", "KBC Fietsverzekering Omnium + Diefstal"),
        ],
        "products": {"bike_insurance": 1},
        "optional": {},
    },
    "family_expansion": {                 # "baby on the way": pregnancy signals months before birth
        "lead_months": 6,
        "searches": ["kinderbijslag", "zwangerschapsverlof", "spaarrekening kind", "kinderopvang"],
        "spend_shift": {"health": 1.8, "shopping": 1.3},
        "intents": ["PREPARE_FOR_BABY", "PROTECT_FAMILY", "SAVE_FOR_CHILD"],
        "txs": [
            (-170, "Dr. Peeters Gynaecoloog", 8011, (60, 150), "Bancontact", "Consultatie gynaecologie"),
            (-110, "Dr. Peeters Gynaecoloog", 8011, (60, 150), "Bancontact", "Echografie 20 weken"),
            (-80, "Prenatal Gent", 5621, (80, 400), "Bancontact", "Zwangerschapskleding"),
            (-45, "Dr. Peeters Gynaecoloog", 8011, (60, 150), "Bancontact", "Consultatie gynaecologie"),
            (-35, "Baby-Dump", 5641, (400, 1800), "Bancontact", "Kinderwagen en babykamer"),
            (0, "AZ Sint-Jan Ziekenhuis", 8062, (300, 2500), "Bancontact", "Ziekenhuisfactuur bevalling"),
            (20, "KBC Verzekeringen", 6300, (15, 60), "Direct Debit", "Levensverzekering premie"),
        ],
        "products": {"life_insurance": 20, "savings_account": 25},
        "optional": {"savings_account": 0.6},
    },
    "wealth_building": {
        "searches": ["beleggen", "fondsen", "pensioensparen"],
        "spend_shift": {"savings": 1.8, "shopping": 0.9},
        "intents": ["INVEST", "PENSION_SAVING"],
        "txs": [
            (0, "KBC Asset Management", 6211, (1000, 20000), "SEPA Transfer", "Instap beleggingsfonds"),
        ],
        "products": {"investment_account": 0},
        "optional": {},
    },
}

# ---- balances & investments ----
# persona -> (savings in months of income, share of customers investing, portfolio in months of income)
WEALTH = {"student": ((0.2, 2), 0.10, (0.5, 3)), "young_professional": ((1, 6), 0.35, (1, 8)),
          "family": ((2, 10), 0.40, (2, 12)), "affluent": ((6, 24), 0.90, (12, 60))}
# investable universe: item, name, asset class, yearly drift, yearly volatility
INSTRUMENTS = [
    ("stock:ASML", "ASML Holding", "stocks", 0.12, 0.35), ("stock:KBC", "KBC Group", "stocks", 0.08, 0.25),
    ("stock:UCB", "UCB", "stocks", 0.10, 0.30), ("stock:ABI", "AB InBev", "stocks", 0.05, 0.25),
    ("stock:SOLB", "Solvay", "stocks", 0.04, 0.30), ("stock:AAPL", "Apple", "stocks", 0.12, 0.30),
    ("stock:MSFT", "Microsoft", "stocks", 0.12, 0.28), ("stock:NVDA", "Nvidia", "stocks", 0.20, 0.50),
    ("etf:IWDA", "iShares MSCI World ETF", "etf", 0.07, 0.15), ("etf:VWCE", "Vanguard FTSE All-World ETF", "etf", 0.07, 0.15),
    ("fund:KBC_ECO", "KBC Eco Fund", "funds", 0.06, 0.18), ("fund:KBC_TECH", "KBC Equity Fund Technology", "funds", 0.10, 0.28),
    ("fund:KBC_PENSION", "KBC Pension Savings Fund", "funds", 0.04, 0.10),
    ("bond:OLO_2034", "Belgian OLO 2034", "bonds", 0.03, 0.05), ("bond:EU_2030", "EU Bond 2030", "bonds", 0.025, 0.04),
]

# background, non-life-event intents (the "Make a payment" type of intents in the paper)
ROUTINE_INTENTS = ["MAKE_PAYMENT", "CHECK_BALANCE", "CARD_QUESTION", "UPDATE_DETAILS",
                   "MANAGE_CASHFLOW", "REPORT_FRAUD"]

PERSONAS = {
    "student":            {"age": (18, 26), "income": (900, 1800),
                           "events": {"micro_mobility_acquisition": 0.35, "automotive_lifecycle": 0.10,
                                      "wealth_building": 0.05},
                           "base_products": {"savings_account": 0.4, "credit_card": 0.2}},
    "young_professional": {"age": (25, 38), "income": (2200, 5000),
                           "events": {"automotive_lifecycle": 0.35, "real_estate_acquisition": 0.25,
                                      "micro_mobility_acquisition": 0.25, "wealth_building": 0.20,
                                      "family_expansion": 0.15},
                           "base_products": {"savings_account": 0.6, "credit_card": 0.5}},
    "family":             {"age": (30, 55), "income": (3500, 8500),
                           "events": {"automotive_lifecycle": 0.30, "real_estate_acquisition": 0.30,
                                      "family_expansion": 0.35, "micro_mobility_acquisition": 0.20,
                                      "wealth_building": 0.15},
                           "base_products": {"savings_account": 0.8, "credit_card": 0.6}},
    "affluent":           {"age": (35, 70), "income": (7000, 18000),
                           "events": {"wealth_building": 0.55, "real_estate_acquisition": 0.25,
                                      "automotive_lifecycle": 0.30, "micro_mobility_acquisition": 0.15},
                           "base_products": {"savings_account": 0.9, "credit_card": 0.8}},
}

# ---- tabular (LightGBM) next-best-product baseline ----
HORIZON = 3                                   # label: product taken in the next HORIZON months
TEST_CUTOFF = N_MONTHS - HORIZON              # test on the last HORIZON months
TRAIN_CUTOFFS = list(range(6, TEST_CUTOFF - HORIZON + 1, 3))   # training labels never reach test period

# ---- TIMeSynC (sequence) model ----
VAL_MONTH = N_MONTHS - 6                      # intents in [VAL_MONTH, TEST_MONTH) -> validation
TEST_MONTH = N_MONTHS - 3                     # intents from TEST_MONTH on -> test
N_BINS = 10                                   # quantile bins for numeric field values
MAX_ENC_LEN = 384                             # most recent context tokens kept per customer
MAX_DEC_LEN = 64                              # most recent intents kept per customer
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 25
BATCH_SIZE = 64
LR = 2e-3
NEED_WINDOW_DAYS = 30                     # target: which needs show up in the next N days
NEED_LOSS_WEIGHT = 3.0                    # weight of the 30-day needs loss vs next-intent loss
