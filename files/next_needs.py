"""Output: a ranked list of possible needs per customer in the next 30 days (config.NEED_WINDOW_DAYS).

Input is a JSON file in the KBC schema: [{client_id, timeline: [{timestamp, merchant, mcc, amount, type, memo}, ...]}]
Output is JSON: [{client_id, as_of, window_days, needs: [{need, life_event, probability}, ...]}, ...]
probability = P(customer shows this need within the window), independent per need.

CLI:     python next_needs.py data/kbc_mock.json [--top 5] [--at 2026-05-20T12:00:00] [-o needs.json]
Python:  from next_needs import get_next_needs; get_next_needs(clients)
"""
import argparse
import json

import pandas as pd

from config import MOCK_FILE, NEED_WINDOW_DAYS
from flatten import kbc_json_to_events
from timesync_predict import INTENT_TO_EVENT, IntentPredictor, products_from_timeline

_predictor = None


def get_next_needs(clients, top=5, at=None):
    """clients: parsed KBC-schema JSON. at: moment to predict (default: 1h after each client's last transaction)."""
    global _predictor
    _predictor = _predictor or IntentPredictor()
    out = []
    for c in clients:
        ev = kbc_json_to_events([c])
        moment = pd.Timestamp(at) if at else ev.timestamp.max() + pd.Timedelta(hours=1)
        pred = _predictor.predict(ev, moment, products=products_from_timeline(c["timeline"]))
        needs = [{"need": n, "life_event": INTENT_TO_EVENT.get(n), "probability": round(p, 4)}
                 for n, p in pred["needs_30d"][:top]]
        out.append({"client_id": c["client_id"], "as_of": str(moment), "window_days": NEED_WINDOW_DAYS,
                    "needs": needs})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default=MOCK_FILE)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--at", help="predict at this moment instead of after the last transaction")
    ap.add_argument("-o", "--output", help="also write the JSON to this file")
    a = ap.parse_args()
    result = get_next_needs(json.load(open(a.input)), a.top, a.at)
    for r in result:
        print(f"\n{r['client_id']}  (as of {r['as_of']}, next {r['window_days']} days)")
        for n in r["needs"]:
            print(f"  {n['probability']:6.1%}  {n['need']:22s} {n['life_event']}")
    if a.output:
        with open(a.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nwritten to {a.output}")


if __name__ == "__main__":
    main()
