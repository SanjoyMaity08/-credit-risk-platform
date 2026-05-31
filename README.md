# 🏦 NeoStats — AI-Powered Credit Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Docker](https://img.shields.io/badge/Deploy-Docker-blue)
![Claude API](https://img.shields.io/badge/LLM-Claude%20API-purple)

> **Candidate Assignment · AI Engineer Role · NeoStats**
> Built on the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) dataset from Kaggle.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Architecture](#-architecture)
3. [Project Structure](#-project-structure)
4. [Prerequisites](#-prerequisites)
5. [Setup & Run Instructions](#-setup--run-instructions)
   - [Option A — Docker (Recommended)](#option-a--docker-recommended)
   - [Option B — Local Python](#option-b--local-python)
6. [Environment Variables](#-environment-variables)
7. [Model Design & Class Imbalance Strategy](#-model-design--class-imbalance-strategy)
8. [Evaluation Metrics & Results](#-evaluation-metrics--results)
9. [Prompt Engineering & Hallucination Control](#-prompt-engineering--hallucination-control)
10. [Decision Rules Logic](#-decision-rules-logic)
11. [Known Limitations & Improvements](#-known-limitations--improvements)

---

## 🎯 Project Overview

NeoStats is a full-stack AI platform for credit risk assessment. It addresses six core banking needs:

| Module | What it Does |
|--------|-------------|
| **EDA & Insights** | Dataset analysis — demographics, financials, credit history, data quality, 6 business insights |
| **ML Risk Scoring** | LightGBM model predicts default probability → risk score (0–1000) → risk band (Low / Medium / High) |
| **Explainable AI** | SHAP feature importance — shows which features drove each prediction |
| **Decision Rules** | ML-derived IF-THEN credit policy rules with confidence scores and support counts |
| **Talk-to-Data** | Natural language → SQL chatbot powered by Claude API — ask questions in plain English |
| **Docker Deployment** | Single `docker-compose up` command launches the full platform |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI  (app.py)                     │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌───────┐ ┌───────┐  │
│  │   EDA    │ │   Risk     │ │  Explai- │ │ Rules │ │ Chat  │  │
│  │Insights  │ │Prediction  │ │ nability │ │Engine │ │  Bot  │  │
│  └──────────┘ └────────────┘ └──────────┘ └───────┘ └───┬───┘  │
└──────────────────────────────────────────────────────────┼──────┘
                                                           │
              ┌────────────────────────┐                   │ NL Question
              │   ML Layer (src/ml/)   │         ┌─────────▼──────────┐
              │  LightGBM · 5-Fold CV  │         │   Talk-to-Data     │
              │  SHAP explainability   │         │  src/talk_to_data/ │
              └────────────┬───────────┘         │  Claude API        │
                           │                     │  NL → SQL → Result │
              ┌────────────▼───────────┐         └─────────┬──────────┘
              │  Data Layer (src/data/)│                   │
              │  loader · preprocessor │         ┌─────────▼──────────┐
              └────────────┬───────────┘         │  SQLite Database   │
                           │                     │  (credit_risk.db)  │
              ┌────────────▼───────────┐         └────────────────────┘
              │  Kaggle CSV Files      │
              │  data/*.csv            │
              └────────────────────────┘
```

---

## 🗂 Project Structure

```
credit_risk_platform/
├── data/
│   └── README.md                      ← Dataset download instructions
├── documents/
│   └── project_presentation.pdf       ← Project presentation (PDF)
├── models/                            ← Saved model artifacts (auto-created)
├── notebooks/
│   └── eda.py                         ← Exploratory Data Analysis script
├── scripts/
│   └── init_db.py                     ← Initialize SQLite from CSV
├── sql/
│   └── schema.sql                     ← Database schema reference
├── src/
│   ├── data/
│   │   ├── loader.py                  ← Load and join dataset tables
│   │   └── preprocessor.py            ← Cleaning, encoding, imputation
│   ├── ml/
│   │   ├── train.py                   ← Model training pipeline
│   │   ├── predict.py                 ← Inference and risk scoring
│   │   └── evaluate.py                ← Metrics: ROC-AUC, PR-AUC, KS
│   ├── talk_to_data/
│   │   ├── nl_to_sql.py               ← NL → SQL via Claude API
│   │   ├── query_runner.py            ← Safe SQL execution on SQLite
│   │   └── prompt_templates.py        ← Versioned prompt templates
│   └── utils/
│       ├── config.py                  ← Environment configuration
│       ├── logger.py                  ← Structured logging
│       └── helpers.py                 ← Shared utility functions
├── app.py                             ← Streamlit multi-section UI
├── Dockerfile                         ← Docker image definition
├── docker-compose.yml                 ← Multi-container orchestration
├── requirements.txt                   ← Python dependencies
├── .env.example                       ← Required environment variables
├── .gitignore
└── README.md
```

---

## ✅ Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Docker | 24+ | [docs.docker.com](https://docs.docker.com/get-docker/) |
| Docker Compose | 2.0+ | Included with Docker Desktop |
| Git | any | [git-scm.com](https://git-scm.com) |
| Kaggle account | — | [kaggle.com](https://www.kaggle.com) — to download dataset |
| Anthropic API key | — | [console.anthropic.com](https://console.anthropic.com) — for Talk-to-Data |

---

## 🚀 Setup & Run Instructions

### Step 1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-platform.git
cd credit-risk-platform
```

---

### Step 2 — Download the Kaggle Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Place your kaggle.json token in ~/.kaggle/
# Get it from: https://www.kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download competition data into data/ folder
kaggle competitions download -c home-credit-default-risk -p data/
cd data && unzip home-credit-default-risk.zip && cd ..
```

> **Manual alternative:** Go to https://www.kaggle.com/competitions/home-credit-default-risk/data, accept the rules, download all files, and place them in the `data/` folder.

**Required files in `data/`:**
```
data/
├── application_train.csv   (~166 MB)  ← required
├── application_test.csv    (~74 MB)   ← required
├── bureau.csv              (~224 MB)  ← optional but recommended
└── previous_application.csv (~138 MB) ← optional but recommended
```

---

### Step 3 — Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here   # Required for Talk-to-Data chatbot
DATA_DIR=/app/data                        # Leave as-is for Docker
MODEL_DIR=/app/models                     # Leave as-is for Docker
DB_PATH=/app/data/credit_risk.db          # Leave as-is for Docker
```

---

### Option A — Docker (Recommended)

#### Step 4A — Build and Launch

```bash
docker-compose up --build
```

This single command will:
- Build the Docker image with all Python dependencies
- Mount your `data/` and `models/` folders into the container
- Start the Streamlit app on port 8501

#### Step 5A — Open the App

```
http://localhost:8501
```

#### Step 6A — Train the Model (first time only)

Open a second terminal and run:

```bash
# Train the LightGBM model
docker-compose exec credit-risk-app python -m src.ml.train

# Initialize the SQLite database for Talk-to-Data
docker-compose exec credit-risk-app python scripts/init_db.py
```

> After training, model artifacts are saved to `models/` and will persist across container restarts.

#### Stop the App

```bash
docker-compose down
```

---

### Option B — Local Python

#### Step 4B — Create Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

#### Step 5B — Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 6B — Set Environment Variables

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-your-key-here
export DATA_DIR=./data
export MODEL_DIR=./models
export DB_PATH=./data/credit_risk.db

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-your-key-here
set DATA_DIR=./data
set MODEL_DIR=./models
set DB_PATH=./data/credit_risk.db
```

#### Step 7B — Train the Model

```bash
python -m src.ml.train
```

Expected output:
```
Training fold 1/5 ...
Training fold 2/5 ...
...
OOF AUC: 0.7721
Training complete.
```

#### Step 8B — Initialize the Database

```bash
python scripts/init_db.py
```

#### Step 9B — Launch the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | **Yes** | — | Claude API key for NL→SQL chatbot |
| `DATA_DIR` | No | `./data` | Path to Kaggle CSV files |
| `MODEL_DIR` | No | `./models` | Path to save/load model artifacts |
| `DB_PATH` | No | `./data/credit_risk.db` | SQLite database path |

---

## 🤖 Model Design & Class Imbalance Strategy

**Algorithm: LightGBM** was chosen because:
- 10–50× faster training than XGBoost on 300K+ rows
- Native handling of missing values (no manual imputation needed for tree splits)
- Histogram-based binning — memory efficient on wide datasets (300+ features)
- Consistently top-performing on credit risk Kaggle leaderboards

**Class Imbalance Strategy** (dataset has ~8% default rate):

| Strategy | Implementation | Reason |
|----------|---------------|--------|
| `scale_pos_weight=11` | LightGBM native | Upweights minority class during gradient computation |
| Stratified K-Fold (k=5) | `StratifiedKFold` | Each fold preserves 8:92 class ratio |
| PR-AUC as metric | `average_precision_score` | More informative than ROC-AUC for imbalanced data |
| Early stopping (50 rounds) | `lgb.early_stopping` | Prevents overfitting to majority class |

**Feature Engineering:**
- `CREDIT_INCOME_RATIO` = AMT_CREDIT / AMT_INCOME_TOTAL
- `ANNUITY_INCOME_RATIO` = AMT_ANNUITY / AMT_INCOME_TOTAL
- `AGE_YEARS` = -DAYS_BIRTH / 365
- `EMPLOYMENT_RATIO` = DAYS_EMPLOYED / DAYS_BIRTH
- Bureau aggregates: loan count, max credit sum, avg overdue days
- Previous application aggregates: count, avg credit, refused count

---

## 📊 Evaluation Metrics & Results

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **ROC-AUC** | 0.7721 | Strong — Kaggle public baseline ~0.75 |
| **PR-AUC** | 0.3142 | Good for 8% minority class |
| **KS Statistic** | 0.4318 | Score distributions well separated |
| **CV Folds** | 5 | Out-of-fold estimate — unbiased |
| **Threshold** | 0.30 / 0.60 | Low / Medium / High band cutoffs |

---

## 💬 Prompt Engineering & Hallucination Control

**NL → SQL Pipeline Design:**

```
User Question
     ↓
System Prompt (schema + strict rules injected)
     ↓
Claude API (claude-sonnet-4-20250514)
     ↓
SQL string OR "UNSUPPORTED"
     ↓
Blocklist Validator (blocks DROP/DELETE/INSERT/UPDATE)
     ↓
SQLite Execution
     ↓
Result JSON (capped at 20 rows)
     ↓
Claude Summarization Call → Plain English Business Insight
```

**Hallucination Controls:**

| Control | Implementation |
|---------|---------------|
| Schema injection | Full column list + types in system prompt |
| `UNSUPPORTED` fallback | Model must return this if question can't be answered |
| SQL blocklist | `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER` blocked before execution |
| Result cap | Only 20 rows sent to summarization — prevents token overflow |
| Temperature default (0) | Deterministic SQL generation |
| Separate summarization call | Keeps SQL generation clean; summary is a distinct task |

**Token Optimization:**
- Schema kept under 800 tokens using column aliases and grouped descriptions
- Summarization prompt capped at 256 max_tokens
- SQL generation capped at 512 max_tokens

---

## 📋 Decision Rules Logic

Rules are derived by training a shallow **Decision Tree** (max_depth=5) on the same feature set as LightGBM. Each leaf node where default rate exceeds a threshold becomes a rule candidate. Rules are then:

1. Extracted as IF-THEN conditions from tree leaf paths
2. Translated into plain-English credit policy language
3. Annotated with **confidence** (leaf default rate) and **support** (leaf sample count)
4. Reviewed against domain knowledge and adjusted for business practicality

**Sample Output:**
```
RULE-004: IF EXT_SOURCE_2 > 0.7 AND CREDIT_INCOME_RATIO < 3.0 AND DAYS_EMPLOYED < -730
          THEN → AUTO-APPROVE
          Confidence: 94% | Support: 18,902 applicants
```

---

## ⚠️ Known Limitations & Improvements

| Limitation | Suggested Improvement |
|------------|----------------------|
| SHAP values are illustrative in UI | Integrate `shap.TreeExplainer` for live per-prediction SHAP |
| Talk-to-Data limited to `applications` table | Add `bureau`, `previous_application` to SQLite |
| No model retraining pipeline | Add MLflow for experiment tracking + model registry |
| Single Docker container | Separate FastAPI model server from Streamlit UI for production |
| No authentication on UI | Add Streamlit auth or OAuth proxy for production |
| Rules are semi-manual | Automate full rule extraction pipeline from Decision Tree |
| Bureau/installment features not in prediction form | Add aggregated bureau features to the UI input form |

---

## 📚 References

- [Home Credit Default Risk — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- [LightGBM Documentation](https://lightgbm.readthedocs.io)
- [Anthropic Claude API](https://docs.anthropic.com)
- [SHAP — Explainability for ML](https://shap.readthedocs.io)
- [Streamlit Documentation](https://docs.streamlit.io)
