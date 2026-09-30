"""TIMeSynC (Katariya et al., RecSys'24 RecTemp) + SASRec baseline, in plain PyTorch.

TIMeSynC = encoder-decoder transformer:
  Encoder  : flattened multi-domain context. token = FieldValue + FieldName + Product + TimeEncoder
             self-attention with TimeAliBi bias and a time-causal mask (key time <= query time)
  Decoder  : sequence of intents. token = PrevIntent + TimeEncoder(query time)
             self-attention: position-causal + TimeAliBi
             cross-attention to encoder: TimeAliBi + time-causal mask (context strictly before query)
  Head     : decoder output + Dense(point-in-time product one-hot) -> intent logits

TimeAliBi: softmax((q K^T + s * -(|t_q - t_k|)) / sqrt(d)) V with a learnable slope s per head.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timesync_data import CAL_SIZES

NEG = -1e9


class TimeEncoder(nn.Module):
    """Absolute time: day-of-week, week-of-month, hour-of-day, month-of-year embeddings."""

    def __init__(self, d):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(n, d) for n in CAL_SIZES)

    def forward(self, cal):
        return sum(emb(cal[..., i]) for i, emb in enumerate(self.embs))


class TimeAliBiAttention(nn.Module):
    def __init__(self, d, heads, dropout, use_alibi=True):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.drop = nn.Dropout(dropout)
        self.use_alibi = use_alibi
        # ALiBi-style geometric slopes (per month of time difference), learnable
        init = torch.tensor([2.0 ** -i for i in range(heads)])
        self.raw_slope = nn.Parameter(torch.log(torch.expm1(init)))
        self.last_attn = None          # kept for explanations

    def forward(self, xq, xk, tq, tk, allowed, zero_bias_keys=None):
        """xq (B,Lq,D) xk (B,Lk,D) tq (B,Lq) tk (B,Lk) allowed (B,Lq,Lk) bool."""
        B, Lq, _ = xq.shape
        Lk = xk.shape[1]
        q = self.q(xq).view(B, Lq, self.h, self.dk).transpose(1, 2)
        k = self.k(xk).view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = self.v(xk).view(B, Lk, self.h, self.dk).transpose(1, 2)
        scores = q @ k.transpose(-1, -2)                                   # (B,H,Lq,Lk)
        if self.use_alibi:
            dt = (tq[:, :, None] - tk[:, None, :]).abs() / 30.0            # months
            if zero_bias_keys is not None:                                 # static tokens: no decay
                dt = dt.masked_fill(zero_bias_keys[:, None, :], 0.0)
            scores = scores - F.softplus(self.raw_slope)[None, :, None, None] * dt[:, None]
        scores = scores / math.sqrt(self.dk)
        scores = scores.masked_fill(~allowed[:, None], NEG)
        attn = self.drop(torch.softmax(scores, dim=-1))
        self.last_attn = attn.detach()
        out = (attn @ v).transpose(1, 2).reshape(B, Lq, -1)
        return self.o(out)


def ffn(d, dropout):
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))


class EncoderLayer(nn.Module):
    def __init__(self, d, h, p, use_alibi):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = TimeAliBiAttention(d, h, p, use_alibi)
        self.ff, self.drop = ffn(d, p), nn.Dropout(p)

    def forward(self, x, t, allowed, static):
        y = self.ln1(x)
        x = x + self.drop(self.attn(y, y, t, t, allowed, static))
        return x + self.drop(self.ff(self.ln2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, h, p, alibi_self, alibi_cross):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = TimeAliBiAttention(d, h, p, alibi_self)
        self.cross_attn = TimeAliBiAttention(d, h, p, alibi_cross)
        self.ff, self.drop = ffn(d, p), nn.Dropout(p)

    def forward(self, x, t, self_allowed, mem, mem_t, cross_allowed, mem_static):
        y = self.ln1(x)
        x = x + self.drop(self.self_attn(y, y, t, t, self_allowed))
        if mem is not None:
            x = x + self.drop(self.cross_attn(self.ln2(x), mem, t, mem_t, cross_allowed, mem_static))
        return x + self.drop(self.ff(self.ln3(x)))


class TIMeSynC(nn.Module):
    DEFAULT_FLAGS = dict(use_context=True, enc_time_enc=True, enc_alibi=True, dec_time_enc=True,
                         dec_alibi_self=True, dec_alibi_cross=True, field_emb=True,
                         enc_product_emb=True, product_fusion=True)

    def __init__(self, n_fn, n_fv, n_prod, n_intents, n_products, n_needs, d=64, heads=4, layers=2,
                 dropout=0.1, **flags):
        super().__init__()
        self.flags = {**self.DEFAULT_FLAGS, **flags}
        f = self.flags
        self.fv_emb = nn.Embedding(n_fv, d, padding_idx=0)
        self.fn_emb = nn.Embedding(n_fn, d, padding_idx=0)
        self.prod_emb = nn.Embedding(n_prod, d, padding_idx=0)
        self.enc_time, self.dec_time = TimeEncoder(d), TimeEncoder(d)
        self.intent_emb = nn.Embedding(n_intents, d, padding_idx=0)
        self.encoder = nn.ModuleList(EncoderLayer(d, heads, dropout, f["enc_alibi"]) for _ in range(layers))
        self.decoder = nn.ModuleList(DecoderLayer(d, heads, dropout, f["dec_alibi_self"], f["dec_alibi_cross"])
                                     for _ in range(layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(d), nn.LayerNorm(d)
        self.product_dense = nn.Sequential(nn.Linear(n_products, d), nn.GELU(), nn.Linear(d, d))
        self.head = nn.Linear(d, n_intents)
        self.need_head = nn.Linear(d, n_needs)          # multi-label: needs in the next N days
        self.drop = nn.Dropout(dropout)

    def encode(self, b):
        f = self.flags
        x = self.fv_emb(b["enc_fv"])
        if f["field_emb"]:
            x = x + self.fn_emb(b["enc_fn"])
        if f["enc_product_emb"]:
            x = x + self.prod_emb(b["enc_prod"])
        if f["enc_time_enc"]:
            x = x + self.enc_time(b["enc_cal"])
        x = self.drop(x)
        t, static, valid = b["enc_t"], b["enc_static"], ~b["enc_pad"]
        # time-causal: a token sees tokens at or before its own time; static tokens always visible
        allowed = ((t[:, None, :] <= t[:, :, None]) | static[:, None, :]) & valid[:, None, :]
        for layer in self.encoder:
            x = layer(x, t, allowed, static)
        return self.enc_ln(x)

    def forward(self, b):
        f = self.flags
        x = self.intent_emb(b["dec_in"])
        if f["dec_time_enc"]:
            x = x + self.dec_time(b["dec_cal"])
        x = self.drop(x)
        t, Ld = b["dec_t"], b["dec_in"].shape[1]
        causal = torch.tril(torch.ones(Ld, Ld, dtype=torch.bool, device=x.device))
        self_allowed = causal[None] & ~b["dec_pad"][:, None, :]
        self_allowed = self_allowed | torch.eye(Ld, dtype=torch.bool, device=x.device)[None]
        mem = mem_t = cross_allowed = static = None
        if f["use_context"]:
            mem, mem_t, static = self.encode(b), b["enc_t"], b["enc_static"]
            # cross-attention: only context strictly BEFORE the moment we predict (no leakage)
            cross_allowed = ((mem_t[:, None, :] < t[:, :, None]) | static[:, None, :]) & ~b["enc_pad"][:, None, :]
        for layer in self.decoder:
            x = layer(x, t, self_allowed, mem, mem_t, cross_allowed, static)
        h = self.dec_ln(x)
        if f["product_fusion"]:
            h = h + self.product_dense(b["own"])
        return self.head(h), self.need_head(h)


class SASRec(nn.Module):
    """Baseline from the paper: causal self-attention over the intent sequence only."""

    def __init__(self, n_intents, n_needs, d=64, heads=4, layers=2, dropout=0.1, max_len=512, **_):
        super().__init__()
        self.intent_emb = nn.Embedding(n_intents, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, n_intents)
        self.need_head = nn.Linear(d, n_needs)

    def forward(self, b):
        x = b["dec_in"]
        L = x.shape[1]
        h = self.intent_emb(x) + self.pos_emb(torch.arange(L, device=x.device))[None]
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        h = self.ln(self.blocks(h, mask=mask, is_causal=True))
        return self.head(h), self.need_head(h)
