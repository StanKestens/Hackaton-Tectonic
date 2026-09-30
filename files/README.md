# KBC hackathon PoC: understanding what a customer needs, right now

Two algorithms on the same synthetic multi-channel data:

| | Algorithm | Question it answers | Input |
|---|---|---|---|
| A | **TIMeSynC** (Katariya et al., 2024, arXiv:2410.12825) encoder-decoder transformer | *What will this customer want next, at this moment?* (next intent + life event) | flattened raw event stream (transactions, logins, searches, monthly spend, products) |
| B | LightGBM next-best-product (tabular baseline) | *Which product will they take in the next 3 months?* | hand-made monthly features |

Setup (use `venv`, which has torch):

    source venv/bin/activate
    pip install -r requirements.txt

    python simulate.py              # 1. synthetic data -> data/  (~15s)
    python train.py                 # 2. LightGBM next-best-product baseline (~40s)
    python train_timesync.py        # 3. TIMeSynC vs SASRec vs popularity (~10 min, MPS/CPU)
    python next_needs.py data/kbc_mock.json --top 5 -o needs.json   # 4. OUTPUT: ranked next needs per client

`next_needs.get_next_needs(clients)` does the same from Python (for the financial-analysis code).
Options: `--include-routine` (also CHECK_BALANCE etc.), `--at <timestamp>` (predict at a given moment).

`python train_timesync.py --ablations` also reruns the paper's Table 4 ablations (slow).

## Files
- `config.py`: every setting: personas, life events (with KBC-style merchants/MCCs/memos), model hyperparameters
- `flatten.py`: turns raw records (KBC JSON transactions etc.) into `(time, domain.field, value, product)` rows (paper Fig. 3)
- `simulate.py`: latent life events → warm-up signals → intents → raw transactions → products
- `timesync_data.py`: tokenizer (quantile bins, vocabularies) and encoder/decoder sequence building
- `timesync_model.py`: TIMeSynC (TimeAliBi, time encoder, field-name/product embeddings, product fusion) and SASRec
- `train_timesync.py`: temporal split, Recall@1/5/10 (paper metric), overall and for life-event intents only
- `next_needs.py`: **the deliverable**, KBC-schema JSON in, ranked list of next needs out
- `timesync_predict.py`: `IntentPredictor` for any timeline + replay of the KBC mock clients, with attention-based "why"
- `features.py`, `train.py`, `explain.py`: LightGBM baseline with SHAP reasons
- `_old/`: the original versions of the files that were rewritten

## Data
`data/kbc_mock.json` is the KBC sample (3 clients). The simulator emits the same transaction schema
(`merchant, mcc, amount, type, memo`), and `data/sample_timelines.json` shows simulated clients in that format.
To plug in new KBC data, add loaders that output the `flatten.EVENT_COLUMNS` rows. Nothing else changes.
