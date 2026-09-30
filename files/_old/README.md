# KBC hackathon PoC: next-best-product on synthetic data

pip install -r requirements.txt
python simulate.py        # 1. generate data
python train.py           # 2. train + evaluate (time-based split)
streamlit run app.py      # 3. demo dashboard
