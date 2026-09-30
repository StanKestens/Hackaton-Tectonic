"""Tokenization + sequence building for TIMeSynC (paper section 4.2.2).

Encoder context X(u) = [(t_i..t_n), (fn_i..fn_n), (fv_i..fv_n), (p_i..p_n)]
  t  = timestamp (days since START_DATE, plus calendar features for the time encoder)
  fn = field name token   "<domain>.<field>"                 (Field Name embedding)
  fv = field value token  "<domain>.<field>=<value>" or "#b<quantile bin>" for numbers
  p  = product token      (Product embedding)
Decoder Y(u) = [(t_i..t_n), (y_i..y_n)]  timestamped intents.
"""
import numpy as np
import pandas as pd
import torch

from config import *
from flatten import STATIC_DOMAINS, is_numeric

PAD, UNK, CLS, BOS = "[PAD]", "[UNK]", "[CLS]", "[BOS]"
T0 = pd.Timestamp(START_DATE)


def to_days(ts):
    return ((pd.to_datetime(ts) - T0) / pd.Timedelta(days=1)).to_numpy(dtype=np.float32)


def month_to_day(m):
    return float((T0 + pd.DateOffset(months=int(m)) - T0) / pd.Timedelta(days=1))


def calendar(ts):
    """Multi-dimensional time signal (day of week, week of month, hour, month) for the time encoder."""
    ts = pd.DatetimeIndex(pd.to_datetime(ts))
    return np.stack([ts.dayofweek, (ts.day - 1) // 7, ts.hour, ts.month - 1], axis=1).astype(np.int64)


CAL_SIZES = [7, 5, 24, 12]


class Tokenizer:
    """Holds vocabularies and quantile bin edges. Fit on training-period data only."""

    def fit(self, events, intents, fit_until_day):
        ev = events[to_days(events.timestamp) < fit_until_day]
        fn = ev.domain + "." + ev.field
        num = np.array([is_numeric(d, f) for d, f in zip(ev.domain, ev.field)])
        self.edges = {}
        for name in fn[num].unique():
            vals = pd.to_numeric(ev.value[num & (fn == name).to_numpy()], errors="coerce").dropna()
            qs = np.unique(np.quantile(vals, np.linspace(0, 1, N_BINS + 1)[1:-1]))
            self.edges[name] = qs
        fv = self._values(ev.domain, ev.field, ev.value)
        self.fn_vocab = self._vocab(fn.unique(), extra=[CLS])
        self.fv_vocab = self._vocab(pd.unique(fv), extra=[CLS])
        self.prod_vocab = self._vocab(["NONE"] + PRODUCTS)
        self.intents = [PAD, BOS] + sorted(intents.intent.unique())
        self.intent_vocab = {k: i for i, k in enumerate(self.intents)}
        self.products = PRODUCTS
        # "needs" = life-event intents (not routine ones); multi-label target over the next N days
        life = {i for ev in LIFE_EVENTS.values() for i in ev["intents"]}
        self.needs = [i for i in self.intents if i in life]
        self.need_index = {k: j for j, k in enumerate(self.needs)}
        return self

    @staticmethod
    def _vocab(items, extra=()):
        v = {PAD: 0, UNK: 1}
        for it in list(extra) + list(items):
            v.setdefault(str(it), len(v))
        return v

    def _values(self, domain, field, value):
        out = []
        for d, f, v in zip(domain, field, value):
            name = f"{d}.{f}"
            if is_numeric(d, f):
                try:
                    b = int(np.searchsorted(self.edges.get(name, []), float(v)))
                except ValueError:
                    b = "nan"
                out.append(f"{name}#b{b}")
            else:
                out.append(f"{name}={v}")
        return np.array(out, dtype=object)

    def encode_events(self, ev):
        """events DataFrame (one customer or many) -> dict of numpy arrays."""
        fn = (ev.domain + "." + ev.field).to_numpy()
        fv = self._values(ev.domain, ev.field, ev.value)
        return {
            "t": to_days(ev.timestamp),
            "cal": calendar(ev.timestamp),
            "fn": np.array([self.fn_vocab.get(x, 1) for x in fn], dtype=np.int64),
            "fv": np.array([self.fv_vocab.get(x, 1) for x in fv], dtype=np.int64),
            "prod": np.array([self.prod_vocab.get(str(x), 1) for x in ev["product"]], dtype=np.int64),
            "static": ev.domain.isin(STATIC_DOMAINS).to_numpy(),
        }

    def encode_intents(self, it):
        return {"t": to_days(it.timestamp), "cal": calendar(it.timestamp),
                "y": np.array([self.intent_vocab.get(x, 0) for x in it.intent], dtype=np.int64)}


def make_sequence(tok, enc, dec, products, query_t=None, hide_history=False):
    """Build one training/inference example.

    enc: encoded events of one customer (sorted by time); dec: encoded intents (sorted).
    products: list of (product, start_day) for point-in-time ownership.
    query_t: if given, append one extra decoder position that asks "what is the intent at query_t?".
    hide_history: decoder never sees earlier intents (only [BOS]) -> the model must predict from
                  context alone, like for the KBC mock clients where no intent history exists.
    """
    # encoder: [CLS] (always visible) + static profile tokens + most recent dynamic tokens
    stat = np.where(enc["static"])[0]
    dyn = np.where(~enc["static"])[0][-(MAX_ENC_LEN - 1 - len(stat)):]
    idx = np.concatenate([stat, dyn]).astype(np.int64)
    e = {k: v[idx] for k, v in enc.items()}
    first_t = e["t"][0] if len(idx) else 0.0
    e = {
        "t": np.concatenate([[first_t - 1], e["t"]]).astype(np.float32),
        "cal": np.concatenate([np.zeros((1, 4), np.int64), e["cal"].reshape(-1, 4)]),
        "fn": np.concatenate([[tok.fn_vocab[CLS]], e["fn"]]),
        "fv": np.concatenate([[tok.fv_vocab[CLS]], e["fv"]]),
        "prod": np.concatenate([[tok.prod_vocab["NONE"]], e["prod"]]),
        "static": np.concatenate([[True], e["static"]]),
        "src": np.concatenate([[-1], idx]),      # row index into the customer's events (-1 = CLS)
    }
    # decoder: input = [BOS] + y[:-1], target = y, query time = time of the target intent
    t, cal, y = dec["t"], dec["cal"].reshape(-1, 4), dec["y"]
    # 30-day needs target: for each query time t_i, which needs occur in [t_i, t_i + window)
    need_col = np.array([tok.need_index.get(tok.intents[v], -1) for v in y], dtype=np.int64)
    needs = np.zeros((len(t), len(tok.needs)), np.float32)
    for i in range(len(t)):
        w = (t >= t[i]) & (t < t[i] + NEED_WINDOW_DAYS) & (need_col >= 0)
        needs[i, need_col[w]] = 1.0
    if query_t is not None:
        qt = pd.Timestamp(T0 + pd.Timedelta(days=float(query_t)))
        t = np.concatenate([t, [query_t]]).astype(np.float32)
        cal = np.concatenate([cal, calendar([qt])])
        y = np.concatenate([y, [0]])
        needs = np.concatenate([needs, np.zeros((1, len(tok.needs)), np.float32)])
    t, cal, y, needs = t[-MAX_DEC_LEN:], cal[-MAX_DEC_LEN:], y[-MAX_DEC_LEN:], needs[-MAX_DEC_LEN:]
    y_in = np.concatenate([[tok.intent_vocab[BOS]], y[:-1]])
    if hide_history:
        y_in = np.full_like(y, tok.intent_vocab[BOS])
    # point-in-time product ownership at each decoder query time
    own = np.zeros((len(t), len(tok.products)), np.float32)
    for p, start in products:
        if p in tok.products:
            own[:, tok.products.index(p)] = (start < t)
    return {"enc": e, "dec": {"t": t.astype(np.float32), "cal": cal, "y_in": y_in, "y": y, "own": own, "needs": needs}}


def collate(batch):
    """Pad a list of make_sequence() outputs into tensors."""
    B = len(batch)
    Le = max(len(b["enc"]["t"]) for b in batch)
    Ld = max(len(b["dec"]["t"]) for b in batch)
    n_prod = batch[0]["dec"]["own"].shape[1]
    n_needs = batch[0]["dec"]["needs"].shape[1]
    out = {
        "enc_t": torch.zeros(B, Le), "enc_cal": torch.zeros(B, Le, 4, dtype=torch.long),
        "enc_fn": torch.zeros(B, Le, dtype=torch.long), "enc_fv": torch.zeros(B, Le, dtype=torch.long),
        "enc_prod": torch.zeros(B, Le, dtype=torch.long),
        "enc_static": torch.zeros(B, Le, dtype=torch.bool), "enc_pad": torch.ones(B, Le, dtype=torch.bool),
        "dec_t": torch.zeros(B, Ld), "dec_cal": torch.zeros(B, Ld, 4, dtype=torch.long),
        "dec_in": torch.zeros(B, Ld, dtype=torch.long), "y": torch.zeros(B, Ld, dtype=torch.long),
        "own": torch.zeros(B, Ld, n_prod), "needs": torch.zeros(B, Ld, n_needs), "dec_pad": torch.ones(B, Ld, dtype=torch.bool),
    }
    for i, b in enumerate(batch):
        e, d = b["enc"], b["dec"]
        n, m = len(e["t"]), len(d["t"])
        out["enc_t"][i, :n] = torch.from_numpy(e["t"])
        out["enc_cal"][i, :n] = torch.from_numpy(e["cal"])
        out["enc_fn"][i, :n] = torch.from_numpy(e["fn"])
        out["enc_fv"][i, :n] = torch.from_numpy(e["fv"])
        out["enc_prod"][i, :n] = torch.from_numpy(e["prod"])
        out["enc_static"][i, :n] = torch.from_numpy(e["static"])
        out["enc_pad"][i, :n] = False
        out["dec_t"][i, :m] = torch.from_numpy(d["t"])
        out["dec_cal"][i, :m] = torch.from_numpy(d["cal"])
        out["dec_in"][i, :m] = torch.from_numpy(d["y_in"])
        out["y"][i, :m] = torch.from_numpy(d["y"])
        out["own"][i, :m] = torch.from_numpy(d["own"])
        out["needs"][i, :m] = torch.from_numpy(d["needs"])
        out["dec_pad"][i, :m] = False
    return out


def load_sim(data_dir=DATA_DIR):
    ev = pd.read_csv(f"{data_dir}/events.csv", parse_dates=["timestamp"])
    it = pd.read_csv(f"{data_dir}/intents.csv", parse_dates=["timestamp"])
    pr = pd.read_csv(f"{data_dir}/products.csv", parse_dates=["timestamp"])
    return ev, it, pr


def build_examples(tok, ev, it, pr):
    """One example per customer: full timeline; train/val/test is decided per decoder position by time."""
    ev_g = dict(tuple(ev.groupby("customer", sort=False)))
    it_g = dict(tuple(it.groupby("customer", sort=False)))
    pr_g = {c: list(zip(g["product"], to_days(g.timestamp))) for c, g in pr.groupby("customer")}
    examples, ids = [], []
    for c, ev_c in ev_g.items():
        if c not in it_g:
            continue
        # customers with only transactions (no app/digital data) are trained without intent history
        sparse = not (ev_c.domain == "digital").any()
        ex = make_sequence(tok, tok.encode_events(ev_c), tok.encode_intents(it_g[c]), pr_g.get(c, []),
                           hide_history=sparse)
        examples.append(ex)
        ids.append(c)
    return ids, examples
