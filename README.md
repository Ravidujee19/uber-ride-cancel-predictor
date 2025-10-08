# 🚕 Uber Ride Cancellation Predictor

Predict the probability that a ride will be **cancelled** using **pre-booking** information only (to avoid post-ride leakage).  
The project includes data prep, modeling, **multi-model ensembles** (soft-voting + stacking), packaging as a single sklearn **pipeline**, a CLI for batch/single predictions, and a Streamlit demo.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-%E2%89%A51.3-FF9900?logo=scikitlearn&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-red?logo=streamlit">
</p>

---

## ✨ Highlights

- **Leakage-aware**: uses only pre-booking features (`hour`, `dayofweek`, `is_weekend`, `Vehicle Type`, `Payment Method`)
- **Multiple models**: Logistic Regression, Random Forest, Gradient Boosting, and optional XGBoost / LightGBM
- **Ensembles**:
  - **Soft Voting** (averages model probabilities)
  - **Stacking** (meta-learner = Logistic Regression)
- **Auto model selection**: picks the **best** by ROC-AUC and saves a single artifact: `artifacts/models/pipeline.pkl`
- **Streamlit app**: interactive demo with single and batch predictions
- **CLI**: predict from JSON (file or inline)

---

## 🗂️ Repo Structure

```
uber-ride-cancel-predictor/
├─ artifacts/
│  ├─ models/
│  │  ├─ pipeline.pkl            # BEST model/ensemble + preprocessing
│  │  └─ metrics.json            # per-model metrics + winner
│  └─ feature_spec.json          # feature schema for reference
├─ data/
│  └─                            # dataset (https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)
├─ scripts/
│  └─ train.py                   # trains all models + ensembles
├─ src/
│  └─ predict.py                 # CLI predictions via saved pipeline
├─ .streamlit/
│  └─ config.toml                # UI theme for Streamlit
├─ inputs.json                   # example request for CLI
├─ app.py                        # Streamlit demo
├─ requirements.txt
├─ Dockerfile
└─ .gitignore
```

---

## 📦 Installation

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> If `xgboost` / `lightgbm` fail to install on your machine, remove them from `requirements.txt`. The training script will skip them automatically.

---

## 📄 Dataset

Default CSV path: **`data/ncr_ride_booking.csv`**

Required columns (case-sensitive):

- `Booking Status` → target mapping:
  - `Completed` → 0
  - `Cancelled by Driver` / `Cancelled by Customer` / `No Driver Found` → 1
  - `Incomplete` → 0
- **Pre-booking** features (at least one of each group):
  - Time: `Date` (`YYYY-MM-DD`) and/or `Time` (`HH:MM` or `HH:MM:SS`)
  - Categorical: `Vehicle Type`, `Payment Method`

The pipeline derives:
- `hour` (from `Time`)
- `dayofweek` + `is_weekend` (from `Date`)

---

## 🏋️‍♂️ Training

**With default dataset path**
```bash
python scripts/train.py
```

**Custom dataset**
```bash
python scripts/train.py --csv /path/to/your.csv
```

Outputs:
- `artifacts/models/pipeline.pkl` ← **use this everywhere**
- `artifacts/models/metrics.json` ← which model won + ROC-AUC
- `artifacts/feature_spec.json`


## ▶️ Streamlit App

```bash
streamlit run app.py
```

Features:
- Single prediction form (Date/Time, Vehicle Type, Payment Method)
- Batch upload (CSV) → download predictions
- Best effort **feature importance** view (tree/linear models; ensembles averaged when possible)

---

## 🧾 CLI Prediction

**Batch (from file)**
```bash
python src/predict.py --json inputs.json --proba
```

**Single row (inline JSON)**
```bash
python src/predict.py --json '{"Date":"2024-11-29","Time":"18:05:00","Vehicle Type":"Go Sedan","Payment Method":"UPI"}' --proba
```

- `--proba` returns **probabilities**; omit to get class labels (0/1).
- The CLI auto-derives `hour`, `dayofweek`, `is_weekend` from `Date` / `Time`.

**Example `inputs.json`:**
```json
[
  {"Date":"2024-11-29","Time":"18:05:00","Vehicle Type":"Go Sedan","Payment Method":"UPI"},
  {"Date":"2024-11-30","Time":"22:10:00","Vehicle Type":"eBike","Payment Method":"Cash"}
]
```

---

## 🤖 Models & Ensembles

**Base models**
- Logistic Regression
- Random Forest
- Gradient Boosting (sklearn)
- XGBoost *(optional)*
- LightGBM *(optional)*

**Ensembles**
- Soft Voting (probability averaging)
- Stacking (meta-learner = Logistic Regression)

**Selection**
- Evaluate all (ROC-AUC on validation split)
- Save the **best** to `artifacts/models/pipeline.pkl`  
  *(includes preprocessing + final estimator/ensemble)*

To see which model won:
```bash
cat artifacts/models/metrics.json
# or on Windows:
Get-Content artifacts\models\metrics.json
```

---

## 🐳 Docker (optional)

```bash
docker build -t ride-cancel-app .
docker run -p 8501:8501 ride-cancel-app
```

Open http://localhost:8501

---

## ⚠️ Notes on Leakage & Realism

- The original public datasets for ride bookings can be **deterministic** (e.g., post-ride fields accidentally reveal the label).
- This repo **intentionally** trains on **pre-booking** features only to simulate real-world deployment and avoid leakage.

---

## 🧰 Troubleshooting

- **ImportError for XGBoost/LightGBM**  
  Remove them from `requirements.txt` and rerun `pip install -r requirements.txt`. Training will continue without them.

- **“Specifying columns using strings is only supported for dataframes.”**  
  You’re probably preprocessing twice. This repo uses a **single, shared** preprocessor at the top of each pipeline.

- **Artifacts missing**  
  Run training: `python scripts/train.py`

- **Different column names**  
  Update the mappings in `scripts/train.py` and `src/predict.py`.

---


## 📜 License

MIT — feel free to use, modify, and share with attribution.

---
