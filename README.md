# Dataset — Home Credit Default Risk

This folder holds the Kaggle CSV files. They are NOT committed to git (too large).

## Download Instructions

### Option A — Kaggle CLI (Recommended)
```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Place your kaggle.json API token in ~/.kaggle/
#    Get it from: https://www.kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Download competition data (run from project root)
kaggle competitions download -c home-credit-default-risk -p data/
cd data && unzip home-credit-default-risk.zip && cd ..
```

### Option B — Manual Download
1. Go to https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Accept competition rules
3. Download all files and place them in this `data/` folder

## Required Files
After download, this folder must contain at least:

| File | Size | Description |
|------|------|-------------|
| `application_train.csv` | ~166 MB | Main training set — 307K applicants |
| `application_test.csv`  | ~74 MB  | Test set for inference |
| `bureau.csv`            | ~224 MB | Credit bureau history |
| `previous_application.csv` | ~138 MB | Previous loan applications |

Optional (used for richer features):
- `bureau_balance.csv`
- `credit_card_balance.csv`
- `installments_payments.csv`
- `POS_CASH_balance.csv`

## After Download — Initialize the Database
```bash
python scripts/init_db.py
```
This loads `application_train.csv` into SQLite for the Talk-to-Data chatbot.

## After Download — Train the Model
```bash
python -m src.ml.train
```
This trains the LightGBM model and saves artifacts to `models/`.
