"""Step 4b: use the trained TIMeSynC model on any customer timeline, incl. the KBC mock JSON.

For every KBC mock client we "replay" the timeline: after each transaction we ask the model
"if this customer opened the app one hour later, what would they want?". Intents are grouped
into life events (INSURE_CAR -> automotive_lifecycle, ...) and compared to ground_truth_link.
The cross-attention weights tell which context tokens drove the prediction (the "why").

Run: python timesync_predict.py [path/to/kbc.json]
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from config import *
from flatten import kbc_json_to_events, product_from_text
from timesync_data import collate, make_sequence, to_days
from timesync_model import TIMeSynC

INTENT_TO_EVENT = {i: ev for ev, cfg in LIFE_EVENTS.items() for i in cfg["intents"]}


class IntentPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        with open(os.path.join(model_dir, "tokenizer.pkl"), "rb") as f:
            self.tok = pickle.load(f)
        ck = torch.load(os.path.join(model_dir, "timesync.pt"), weights_only=False)
        self.model = TIMeSynC(**ck["sizes"], **ck["flags"])
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, events, at, intents=None, products=None, top_k=5, n_reasons=5):
        """events: flattened events of ONE customer; at: Timestamp of the moment to predict.
        intents: optional DataFrame(timestamp, intent) of earlier intents; products: [(name, Timestamp)]."""
        events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
        intents = intents if intents is not None else pd.DataFrame(columns=["timestamp", "intent"])
        intents = intents[intents.timestamp < at].sort_values("timestamp")
        prods = [(p, float(to_days(pd.Series([ts]))[0])) for p, ts in (products or [])]
        enc = self.tok.encode_events(events)
        dec = self.tok.encode_intents(intents) if len(intents) else {
            "t": np.zeros(0, np.float32), "cal": np.zeros((0, 4), np.int64), "y": np.zeros(0, np.int64)}
        at_day = float(to_days(pd.Series([at]))[0])
        ex = make_sequence(self.tok, enc, dec, prods, query_t=at_day,
                           hide_history=not (events.domain == "digital").any())
        b = collate([ex])
        logits, need_logits = self.model(b)
        logits, need_prob = logits[0, -1], torch.sigmoid(need_logits[0, -1]).numpy()
        logits[:2] = float("-inf")
        prob = torch.softmax(logits, -1).numpy()

        ranked = np.argsort(-prob)
        top = [(self.tok.intents[i], float(prob[i])) for i in ranked[:top_k]]
        life = {}
        for i, p in enumerate(prob):
            ev = INTENT_TO_EVENT.get(self.tok.intents[i])
            if ev:
                life[ev] = life.get(ev, 0.0) + float(p)
        life = dict(sorted(life.items(), key=lambda kv: -kv[1]))

        # why: last decoder layer cross-attention of the query position, averaged over heads
        attn = self.model.decoder[-1].cross_attn.last_attn[0, :, -1].mean(0).numpy()
        reasons = []
        for j in np.argsort(-attn):
            src = ex["enc"]["src"][j]
            if src < 0 or attn[j] < 1e-3:
                continue
            r = events.iloc[src]
            reasons.append({"time": str(r.timestamp), "token": f"{r.domain}.{r.field}={r.value}",
                            "product": r["product"], "weight": float(attn[j])})
            if len(reasons) == n_reasons:
                break
        needs = sorted(((n, float(p)) for n, p in zip(self.tok.needs, need_prob)), key=lambda x: -x[1])
        return {"at": str(at), "needs_30d": needs, "top_intents": top, "life_events": life, "reasons": reasons}


def products_from_timeline(timeline):
    """Point-in-time product ownership, derived from KBC's own direct debits in the timeline."""
    out = []
    for tx in timeline:
        p = product_from_text(tx["merchant"], tx["memo"])
        if p != "NONE" and tx["merchant"].startswith("KBC"):
            out.append((p, pd.Timestamp(tx["timestamp"]).tz_convert(None)))
    return out


def score_kbc_file(path=MOCK_FILE, predictor=None):
    predictor = predictor or IntentPredictor()
    clients = json.load(open(path))
    results = []
    for c in clients:
        ev = kbc_json_to_events([c])
        prods = products_from_timeline(c["timeline"])
        steps = []
        for tx in sorted(c["timeline"], key=lambda t: t["timestamp"]):
            at = pd.Timestamp(tx["timestamp"]).tz_convert(None) + pd.Timedelta(hours=1)
            pred = predictor.predict(ev, at, products=prods)
            steps.append({"after_tx": tx["tx_id"], "merchant": tx["merchant"], "memo": tx["memo"], **pred})
        first_life = next(iter(steps[0]["life_events"]))
        results.append({"client_id": c["client_id"], "ground_truth": c.get("ground_truth_link"),
                        "predicted_life_event_first_tx": first_life,
                        "predicted_life_event_final": next(iter(steps[-1]["life_events"])),
                        "steps": steps})
    return results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else MOCK_FILE
    results = score_kbc_file(path)
    for r in results:
        print(f"\n=== {r['client_id']}  (ground truth: {r['ground_truth']})")
        for s in r["steps"]:
            intents = ", ".join(f"{i} {p:.0%}" for i, p in s["top_intents"][:3])
            ev, p = next(iter(s["life_events"].items()))
            print(f"  after {s['merchant'][:30]:30s} -> {intents:55s} | life event: {ev} ({p:.0%})")
        print("  why (last step):", "; ".join(f"{x['token']} ({x['weight']:.0%})" for x in r["steps"][-1]["reasons"][:3]))
    hit_first = np.mean([r["predicted_life_event_first_tx"] == r["ground_truth"] for r in results])
    hit_final = np.mean([r["predicted_life_event_final"] == r["ground_truth"] for r in results])
    print(f"\nLife-event accuracy on {len(results)} KBC mock clients: after 1st tx {hit_first:.0%}, "
          f"after full timeline {hit_final:.0%}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "mock_predictions.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
