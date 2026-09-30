"""Step 4: SHAP explanations turned into short plain-language reasons."""
import numpy as np
import shap

NICE = {"savings_inflow": "savings deposits", "housing_goods": "home-related spending",
        "baby_kids": "spending on children", "car_fuel": "car and fuel spending",
        "travel": "travel spending", "leisure": "leisure spending",
        "groceries": "grocery spending", "pension_inflow": "pension contributions"}


def describe(feature, value, shap_value):
    direction = "raises" if shap_value > 0 else "lowers"
    if feature.startswith("chg_"):
        cat = NICE.get(feature[4:], feature[4:])
        what = f"{cat} {'went up' if value > 0 else 'went down'} recently ({value:+.0%})"
    elif feature.startswith("recent_"):
        what = f"{NICE.get(feature[7:], feature[7:])} averaged {value:,.0f} EUR/month"
    elif feature.startswith("held_"):
        what = f"customer {'already has' if value else 'does not have'} {feature[5:].replace('_', ' ')}"
    elif feature.startswith("persona_"):
        what = f"customer profile: {feature[8:].replace('_', ' ')}" if value else None
    else:
        what = f"{feature} = {value:,.0f}"
    return None if what is None else f"{what} ({direction} the score)"


def top_reasons(model, x_row, n=3):
    """x_row: one-row DataFrame. Returns up to n human-readable reasons."""
    sv = np.asarray(shap.TreeExplainer(model).shap_values(x_row))
    if sv.ndim == 3:
        sv = sv[..., 1]
    sv = sv.reshape(-1)
    out = []
    for i in np.argsort(-np.abs(sv)):
        text = describe(x_row.columns[i], x_row.iloc[0, i], sv[i])
        if text:
            out.append(text)
        if len(out) == n:
            break
    return out
