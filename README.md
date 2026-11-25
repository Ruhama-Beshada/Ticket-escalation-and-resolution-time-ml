# Ticket-escalation-and-resolution-time-ml
Leakage-safe classification and regression pipelines for support tickets, built with scikit-learn.

- **Experiment A (classification):** predict whether a ticket will be escalated (`escalated`).
- **Experiment B (regression):** predict how long a ticket takes to resolve (`resolution_time_hours`).

The notebook compares models with 5-fold cross-validation, tunes the best candidate, evaluates the final model on a held-out test set once, and includes a written report with reflections.

## Repository structure

```text
ticket-escalation-and-resolution-time-ml/
├── ticket_analysis.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── classification_escalation.csv   (900 rows)
    └── resolution_time_regression.csv  (800 rows)
```

## How to run

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook ticket_analysis.ipynb
```

Keep the `data` folder in the same folder as the notebook.

## Method

- `ticket_id` is excluded (identifier only).
- Test set (20%) is held out first. Classification uses a stratified split.
- Preprocessing inside a `Pipeline`: median imputation + scaling for numeric columns, most-frequent imputation + one-hot encoding for categorical columns.
- Identical 5-fold CV for all models (`StratifiedKFold` for classification, `KFold` for regression).
- Primary metrics: F1 (classification) and MAE (regression).
- Tuning with `GridSearchCV`. Test set evaluated once per experiment.

## Results

### Experiment A: classification (F1)

| Model | CV mean F1 | CV std |
|---|---|---|
| Logistic Regression | 0.393 | 0.063 |
| k-NN | 0.361 | 0.050 |
| Decision Tree | 0.331 | 0.058 |
| **Logistic Regression (tuned: C=0.1, class_weight="balanced")** | **0.520** | **0.042** |

**Test result:** F1 = 0.60, precision = 0.52, recall = 0.71 (25 false positives, 11 false negatives).

### Experiment B: regression (MAE, hours)

| Model | CV mean MAE | CV std |
|---|---|---|
| Baseline (median) | 3.195 | 0.083 |
| Linear Regression | 2.035 | 0.172 |
| **Ridge (tuned, alpha=0.1)** | **2.035** | **0.172** |
| Decision Tree (default) | 3.269 | 0.144 |
| Decision Tree (tuned: depth 6, leaf 20) | 2.364 | 0.089 |

**Test result:** MAE = 2.230 h, RMSE = 3.403 h (median baseline test MAE = 3.273 h).

## Key findings

- Class weighting (`balanced`) drove the classification improvement; `C` hardly mattered.
- Ridge and Linear Regression tie; the tuned tree is steadier across folds but about 0.33 h worse.
- Two extreme test tickets explain most of the gap between CV and test MAE; the 5 worst predictions make up 56.5% of the squared error, which is why RMSE is much larger than MAE.

## Limitations

- Only Logistic Regression was tuned in Experiment A.
- Features are weakly predictive for escalation.
- Small test sets (38 escalated tickets; only 2 outlier tickets in regression), so test numbers are noisy.
- Decision threshold left at 0.5.

The full report and reflection answers are in the last section of the notebook.
