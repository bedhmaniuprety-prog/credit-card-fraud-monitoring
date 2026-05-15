# Credit Card Fraud Detection with Continuous Monitoring

A machine learning pipeline for credit card fraud detection, with a focus on
**post-deployment monitoring and retraining strategies**. Built as part of a
thesis project (Chapter 4).

## What this project does

1. Loads and explores the [Kaggle creditcard dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate)
2. Engineers temporal features and applies a time-ordered train/val/test split to avoid leakage
3. Trains and compares three models: Logistic Regression, Random Forest, and XGBoost
4. Selects decision thresholds on validation data by maximising F1-score
5. Simulates continuous monitoring across sliding windows of test data, comparing three strategies:
   - **No retraining** — fixed model evaluated over time
   - **Rolling retraining** — model retrained every N windows on accumulated data
   - **Event-based retraining** — retraining triggered by recall drop or PSI drift alert

## Repository Structure

```text
credit-card-fraud-monitoring/
├── data/                  # Place creditcard.csv here (not tracked by Git)
├── notebooks/
│   └── fraud_detection.py # Main pipeline script
├── outputs/               # Generated CSVs and plots (not tracked by Git)
├── src/
│   ├── config.py          # Constants and hyperparameters
│   ├── data.py            # Data loading, feature engineering, splitting
│   ├── metrics.py         # Evaluation, threshold search, PSI
│   ├── models.py          # Baseline models and XGBoost grid search
│   ├── monitoring.py      # Monitoring strategies
│   └── visualization.py   # ROC, PR curves, and confusion matrix plots
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-monitoring.git
cd credit-card-fraud-monitoring
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

## 3. Add the Dataset

Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Place the file inside the `data/` folder:

```text
data/
└── creditcard.csv
```

> Note: The dataset is not tracked by Git because it is large. Make sure `data/` is included in `.gitignore`.

**4. Run the pipeline**
```bash
python notebooks/fraud_detection.py
```

All outputs (CSVs and plots) are saved to `outputs/`.

## Key design decisions

- **Temporal split** instead of random split — transactions are ordered by time,
  so a random split would leak future data into training and inflate all metrics.
- **PR-AUC for model selection** — with 0.17% fraud rate, ROC-AUC is misleading;
  PR-AUC better reflects performance on the minority class.
- **Threshold search on validation data** — default 0.5 threshold is wrong for
  imbalanced problems; we sweep thresholds and pick the one maximising F1.
- **PSI for drift detection** — Population Stability Index flags when feature
  distributions shift significantly from the training reference.

## Dataset

[ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Features V1–V28 are PCA-transformed for confidentiality. Raw features are `Time` and `Amount`.