"""Step 3b: train + evaluate TIMeSynC against the paper's baselines on next-intent prediction.

Temporal split per intent (paper 4.1): intents before VAL_MONTH train, [VAL_MONTH, TEST_MONTH)
validate, >= TEST_MONTH test. Context is always restricted to events strictly before each intent.
Two targets, trained jointly:
  next intent   (paper)  : Recall@k, overall and for life-event intents only
  30-day needs  (ours)   : which life-event needs occur in the next NEED_WINDOW_DAYS -> Recall@k, PR-AUC

Run: python train_timesync.py                  # popularity, SASRec, TIMeSynC
     python train_timesync.py --ablations      # + paper Table 4 style ablations
"""
import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from config import *
from timesync_data import Tokenizer, build_examples, collate, load_sim, month_to_day
from timesync_model import TIMeSynC, SASRec

VAL_DAY, TEST_DAY, END_DAY = month_to_day(VAL_MONTH), month_to_day(TEST_MONTH), month_to_day(N_MONTHS)
KS = (1, 5, 10)
NEED_KS = (1, 3, 5)
LIFE_INTENTS = {i for ev in LIFE_EVENTS.values() for i in ev["intents"]}


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def batches(examples, bs, shuffle, rng=None):
    order = rng.permutation(len(examples)) if shuffle else np.arange(len(examples))
    for i in range(0, len(order), bs):
        yield collate([examples[j] for j in order[i:i + bs]])


def split_mask(b, split):
    t, valid = b["dec_t"], ~b["dec_pad"]
    m = {"train": t < VAL_DAY, "val": (t >= VAL_DAY) & (t < TEST_DAY), "test": t >= TEST_DAY}[split]
    return m & valid


def need_mask(b, split):
    """Positions whose full 30-day label window lies inside the split (no label leakage)."""
    t, valid, W = b["dec_t"], ~b["dec_pad"], NEED_WINDOW_DAYS
    m = {"train": t + W < VAL_DAY, "val": (t >= VAL_DAY) & (t + W < TEST_DAY),
         "test": (t >= TEST_DAY) & (t + W <= END_DAY)}[split]
    return m & valid


def joint_loss(out, b, split):
    lg, need_lg = out
    m, nm = split_mask(b, split), need_mask(b, split)
    loss = F.cross_entropy(masked_logits(lg)[m], b["y"][m], reduction="sum") / m.sum().clamp(min=1)
    if nm.any():
        loss = loss + NEED_LOSS_WEIGHT * F.binary_cross_entropy_with_logits(need_lg[nm], b["needs"][nm])
    return loss


def need_metrics(scores, labels):
    """scores/labels: (N, n_needs) numpy. Recall@k over positions with >=1 need, micro PR-AUC."""
    out = {"needs_PR-AUC": float(average_precision_score(labels.ravel(), scores.ravel()))}
    has = labels.sum(1) > 0
    order = np.argsort(-scores[has], axis=1)
    lab = labels[has]
    for k in NEED_KS:
        got = np.take_along_axis(lab, order[:, :k], axis=1).sum(1)
        out[f"needs_R@{k}"] = float((got / lab.sum(1)).mean())
    out["n_need_pos"] = int(has.sum())
    return out


def masked_logits(logits):
    logits = logits.clone()
    logits[..., :2] = float("-inf")          # never predict [PAD] / [BOS]
    return logits


def to_dev(b, dev):
    return {k: v.to(dev) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, examples, split, tok, dev):
    model.eval()
    life_ids = torch.tensor([tok.intent_vocab[i] for i in LIFE_INTENTS if i in tok.intent_vocab])
    hits = {k: [] for k in KS}
    is_life, losses, n_scores, n_labels = [], [], [], []
    for b in batches(examples, BATCH_SIZE, False):
        b = to_dev(b, dev)
        out = model(b)
        nm = need_mask(b, split)
        n_scores.append(torch.sigmoid(out[1][nm]).cpu().numpy())
        n_labels.append(b["needs"][nm].cpu().numpy())
        m = split_mask(b, split)
        if m.sum() == 0:
            continue
        lg = masked_logits(out[0])[m]
        y = b["y"][m]
        losses.append(F.cross_entropy(lg, y, reduction="sum").item())
        top = lg.topk(max(KS), dim=-1).indices
        for k in KS:
            hits[k].append((top[:, :k] == y[:, None]).any(-1).cpu())
        is_life.append(torch.isin(y.cpu(), life_ids))
    life = torch.cat(is_life)
    out = {"loss": sum(losses) / len(life)}
    for k in KS:
        h = torch.cat(hits[k]).float()
        out[f"R@{k}"] = h.mean().item()
        out[f"life_R@{k}"] = h[life].mean().item()
    out["n"], out["n_life"] = len(life), int(life.sum())
    out.update(need_metrics(np.concatenate(n_scores), np.concatenate(n_labels)))
    return out


def train_model(name, model, train_ex, val_ex, tok, dev, epochs=EPOCHS):
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * (len(train_ex) // BATCH_SIZE + 1))
    rng = np.random.default_rng(SEED)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in batches(train_ex, BATCH_SIZE, True, rng):
            b = to_dev(b, dev)
            loss = joint_loss(model(b), b, "train")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, n = tot + loss.item(), n + 1
        v = evaluate(model, val_ex, "val", tok, dev)
        if -v["needs_PR-AUC"] < best:           # model selection on the 30-day needs objective
            best, best_state = -v["needs_PR-AUC"], {k: x.detach().clone() for k, x in model.state_dict().items()}
        print(f"  [{name}] ep {ep + 1:2d} train {tot / n:.3f} | val next-intent R@1 {v['R@1']:.3f} "
              f"| 30d needs PR-AUC {v['needs_PR-AUC']:.3f} R@3 {v['needs_R@3']:.3f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model


def popularity_eval(tok, it):
    """Most-popular-intent ranking from the training period."""
    tr = it[it.timestamp < pd.Timestamp(START_DATE) + pd.Timedelta(days=VAL_DAY)]
    te = it[it.timestamp >= pd.Timestamp(START_DATE) + pd.Timedelta(days=TEST_DAY)]
    rank = list(tr.intent.value_counts().index)
    out = {}
    life = te.intent.isin(LIFE_INTENTS).to_numpy()
    pos = te.intent.map({x: i for i, x in enumerate(rank)}).fillna(1e9).to_numpy()
    for k in KS:
        out[f"R@{k}"] = float((pos < k).mean())
        out[f"life_R@{k}"] = float((pos[life] < k).mean())
    out["n"], out["n_life"] = len(te), int(life.sum())
    return out


def popularity_needs(examples, tok):
    """Constant need scores = training-period frequency of each need."""
    tr, te = [], []
    for b in batches(examples, 512, False):
        tr.append(b["needs"][need_mask(b, "train")].numpy())
        te.append(b["needs"][need_mask(b, "test")].numpy())
    prior = np.concatenate(tr).mean(0)
    labels = np.concatenate(te)
    return need_metrics(np.tile(prior, (len(labels), 1)), labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    dev = pick_device()
    print(f"device: {dev}")

    ev, it, pr = load_sim()
    tok = Tokenizer().fit(ev, it, fit_until_day=VAL_DAY)
    print(f"vocab: {len(tok.fn_vocab)} field names, {len(tok.fv_vocab)} field values, "
          f"{len(tok.intents)} intents")
    ids, examples = build_examples(tok, ev, it, pr)
    # all customers are used; the split is temporal (per decoder position), not per customer
    train_ex = val_ex = test_ex = examples

    sizes = dict(n_fn=len(tok.fn_vocab), n_fv=len(tok.fv_vocab), n_prod=len(tok.prod_vocab),
                 n_intents=len(tok.intents), n_products=len(tok.products), n_needs=len(tok.needs),
                 d=D_MODEL, heads=N_HEADS, layers=N_LAYERS, dropout=DROPOUT)
    runs = {"SASRec": (SASRec, {}), "TIMeSynC": (TIMeSynC, {})}
    if args.ablations:
        for flag in ["dec_time_enc", "dec_alibi_self", "dec_alibi_cross", "product_fusion",
                     "field_emb", "enc_time_enc", "enc_alibi", "enc_product_emb"]:
            runs[f"TIMeSynC w/o {flag}"] = (TIMeSynC, {flag: False})

    results = {"Popularity": {**popularity_eval(tok, it), **popularity_needs(examples, tok)}}
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, (cls, flags) in runs.items():
        print(f"\n== {name}")
        model = train_model(name, cls(**sizes, **flags), train_ex, val_ex, tok, dev, args.epochs)
        results[name] = evaluate(model, test_ex, "test", tok, dev)
        if name in ("TIMeSynC", "SASRec"):
            torch.save({"state": model.state_dict(), "sizes": sizes, "flags": flags, "cls": name},
                       os.path.join(MODEL_DIR, f"{name.lower()}.pt"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tok, f)

    res = pd.DataFrame(results).T
    print("\nTest set (last 3 months)")
    print("Next intent: R@k = Recall@k over all intents, life_R@k = life-event intents only")
    print(res[[f"R@{k}" for k in KS] + [f"life_R@{k}" for k in KS]].round(3).to_string())
    print(f"\nNeeds in the next {NEED_WINDOW_DAYS} days: share of true needs in the top-k, PR-AUC over all scores")
    print(res[["needs_PR-AUC"] + [f"needs_R@{k}" for k in NEED_KS] + ["n_need_pos"]].round(3).to_string())
    res.to_csv(os.path.join(MODEL_DIR, "timesync_results.csv"))


if __name__ == "__main__":
    main()
