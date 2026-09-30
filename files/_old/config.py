# config.py

SEED = 42

DATA_DIR = "data"

N_CUSTOMERS = 3000
N_MONTHS = 24
LEAD = 3
DECOY_RATE = 0.25

CATEGORIES = [
    "housing",
    "food",
    "transport",
    "utilities",
    "health",
    "leisure",
    "shopping",
    "savings",
]

BASE_SHARE = {
    "housing": 0.30,
    "food": 0.12,
    "transport": 0.08,
    "utilities": 0.08,
    "health": 0.05,
    "leisure": 0.08,
    "shopping": 0.12,
    "savings": 0.17,
}

PRODUCTS = [
    "savings_account",
    "credit_card",
    "home_loan",
    "car_loan",
    "investment_account",
    "insurance",
]

PERSONAS = {
    "student": {
        "age": (18, 26),
        "income": (900, 1800),
        "p": {
            "savings_account": 0.35,
            "credit_card": 0.25,
            "insurance": 0.05,
            "investment_account": 0.05,
        },
    },
    "young_professional": {
        "age": (25, 38),
        "income": (2200, 5000),
        "p": {
            "savings_account": 0.50,
            "credit_card": 0.45,
            "car_loan": 0.20,
            "investment_account": 0.25,
            "insurance": 0.30,
        },
    },
    "family": {
        "age": (30, 55),
        "income": (3500, 8500),
        "p": {
            "savings_account": 0.60,
            "credit_card": 0.55,
            "home_loan": 0.45,
            "car_loan": 0.30,
            "insurance": 0.65,
            "investment_account": 0.30,
        },
    },
    "affluent": {
        "age": (35, 70),
        "income": (7000, 18000),
        "p": {
            "savings_account": 0.70,
            "credit_card": 0.65,
            "home_loan": 0.35,
            "insurance": 0.75,
            "investment_account": 0.70,
        },
    },
}

SIGNALS = {
    "savings_account": {
        "savings": 1.8,
        "leisure": 0.90,
    },
    "credit_card": {
        "shopping": 1.7,
        "leisure": 1.4,
    },
    "home_loan": {
        "housing": 1.8,
        "savings": 0.85,
    },
    "car_loan": {
        "transport": 1.8,
        "savings": 0.90,
    },
    "investment_account": {
        "savings": 1.8,
        "shopping": 0.90,
    },
    "insurance": {
        "health": 1.6,
        "housing": 1.2,
    },
}

# ---- evaluation / model settings (were missing -> features/train/app crashed) ----
MODEL_DIR = "models"
HORIZON = 3                                   # label window: product taken in next HORIZON months
TEST_CUTOFF = N_MONTHS - HORIZON              # test on the last HORIZON months
TRAIN_CUTOFFS = list(range(6, TEST_CUTOFF - HORIZON + 1, 3))  # labels never overlap test period
