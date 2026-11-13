# Credit Risk Intelligence System with Explainable AI (XAI) & Historical Cohort Support

A production-grade credit risk underwriting and decision-support web application integrating **Explainable AI (SHAP & LIME)**, **Historical Cohort Analysis (K-Nearest Neighbors)**, **Interactive What-If Counterfactual Simulation**, and **Responsible AI / Fair Lending Auditing**.

---

## 📌 Executive Summary & Architecture

This system upgrades traditional credit scoring into an explainable, transparent, and defensible decision-support platform built on the **Statlog German Credit Dataset** (1,000 real borrower records across 20 financial & demographic features).

```
ds_PROJECT/
├── app.py                      # Main Streamlit dashboard application
├── credit_data.csv             # Curated dataset (1,000 records, 21 columns)
├── requirements.txt            # System dependencies
├── test_system.py              # Automated 9-point unit & integration test suite
├── README.md                   # System documentation & operating guide
├── src/
│   ├── __init__.py             # Package init
│   ├── config.py               # Schemas, feature specifications, risk thresholds
│   ├── data_processor.py       # Preprocessing pipeline, column transformer, train-test split
│   ├── model_trainer.py        # ML training, cross-validation, 4-model evaluation, serialization
│   ├── xai_engine.py           # SHAP & LIME local/global explainability & narrative generation
│   ├── historical_matcher.py   # Historical cohort retrieval & leakage-free neighbor matching
│   ├── fairness_evaluator.py   # Disparate impact & demographic parity checks
│   └── history_store.py        # Audit logging with privacy safeguards
├── models/
│   ├── best_model.joblib       # Primary serialized model (Random Forest)
│   ├── all_models.joblib       # All 4 trained classifiers
│   ├── preprocessor.joblib     # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── model_metrics.json      # Evaluation metrics across all 4 models
│   ├── confusion_matrices.joblib # Confusion matrices
│   ├── roc_curves.joblib       # ROC curve coordinates
│   ├── train_data.joblib       # Saved training split & feature mappings
│   └── prediction_history.json # Local audit log of assessments
└── extracted_data/             # Raw UCI German credit source files
```

---

## 📊 1. Dataset Analysis & Preprocessing

- **Dataset Source:** Statlog German Credit Data (1,000 borrower records).
- **Target Variable (`credit_risk`):**
  - `0`: Good / Repaid Loan (700 records, 70.0%)
  - `1`: Bad / Defaulted Loan (300 records, 30.0%)
- **Data Integrity:** 0 missing values, 0 duplicate records.
- **Features (20 total):**
  - **7 Numerical:** `duration` (months), `credit_amount` (DM), `installment_rate` (% of income), `residence_since` (years), `age` (years), `existing_credits` (count), `num_dependents`.
  - **13 Categorical:** `checking_status`, `credit_history`, `purpose`, `savings_status`, `employment`, `personal_status_sex`, `other_parties`, `property_magnitude`, `other_payment_plans`, `housing`, `job`, `telephone`, `foreign_worker`.
- **Leakage Prevention:** Features are split via `train_test_split(test_size=0.2, stratify=y, random_state=42)`. All scalers (`StandardScaler`) and encoders (`OneHotEncoder(handle_unknown='ignore')`) are strictly fitted on the training split only and applied downstream.

---

## 🤖 2. Machine Learning Model Performance

Four classification models were trained and benchmarked on the out-of-sample test split ($N=200$, 60 default cases, 140 non-default cases) using balanced class weights to address the 5:1 cost asymmetry of default:

| Model | Accuracy | Precision | Recall (Default Detection) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Primary)** | **73.0%** | **53.8%** | **71.7%** | **0.614** | **0.797** |
| **Logistic Regression** | 75.0% | 55.8% | 80.0% | 0.658 | 0.806 |
| **Support Vector Machine (Calibrated)** | 78.0% | 69.1% | 48.3% | 0.569 | 0.811 |
| **Decision Tree** | 56.0% | 37.3% | 68.3% | 0.482 | 0.598 |

### Selection Rationale
In retail and commercial credit underwriting, the cost of failing to identify an actual default (False Negative) is approximately 5x more costly than a rejected good borrower (False Positive). **Random Forest** provides an optimal combination of strong ROC-AUC discriminative power (0.797+), robust recall, and native exact TreeExplainer compatibility for SHAP attribution.

---

## 🔍 3. Explainable AI (XAI) Architecture

1. **SHAP (SHapley Additive exPlanations):**
   - Employs `shap.TreeExplainer` for exact Shapley value decomposition of tree ensembles.
   - Decomposes applicant predictions into additive positive (risk-elevating) and negative (risk-mitigating) components relative to baseline base risk (~50.2%).
   - Generates local feature impact horizontal bar charts and global feature importance rankings.
2. **Dynamic Natural Language Generation:**
   - Translates numerical SHAP values and specific borrower characteristics into plain-language financial narratives explaining *why* the credit score was assigned.
3. **LIME (Local Interpretable Model-agnostic Explanations):**
   - Employs `LimeTabularExplainer` as an independent verification method using local perturbations to confirm primary feature drivers.

---

## 🏛️ 4. Historical Cohort Analysis

- Uses `sklearn.neighbors.NearestNeighbors(metric='euclidean')` fitted on the historical borrower feature space.
- Given a prospective applicant, dynamically retrieves the $k$ nearest historical borrower records ($k=3$ to $15$).
- Aggregates cohort outcomes:
  - Count and percentage of borrowers who fully repaid vs defaulted.
  - Average loan amount and duration of the matched cohort.
- **Zero Target Leakage:** Neighbor queries operate solely across independent financial features without visibility into ground-truth outcomes until displaying historical performance.

---

## 🎯 5. Defensible Risk Classification

Because default risk is probabilistic, continuous predicted probabilities are mapped into three actionable risk categories:
- **Low Risk ($P(\text{Default}) < 28\%$):** Below baseline population default rate (30%). Eligible for standard automated or fast-track approval.
- **Medium Risk ($28\% \le P(\text{Default}) < 58\%$):** Concentrated around population base rate. Mandates manual underwriter review, debt-to-income verification, or additional guarantor.
- **High Risk ($P(\text{Default}) \ge 58\%$):** Elevated default probability. Recommend adverse action or substantial secured collateral.

---

## 🔄 6. Interactive What-If Scenario Simulator

- Allows credit analysts to adjust key financial levers:
  - Loan amount and duration
  - Checking account liquidity and savings balances
  - Co-signers / Guarantors and installment ratios
- Instant delta recalculation ($\Delta P$ and risk category transition).
- Dynamic narrative explaining which adjustment contributed most to the risk reduction.
- Explicit regulatory disclosure that statistical associations do not guarantee approval.

---

## 🛡️ 7. Responsible AI & Fair Lending Audit

- Disparate Impact and Demographic Parity audits across sensitive attributes:
  - **Gender:** Female vs Male applicants.
  - **Age:** Young borrowers ($< 25$) vs Mature borrowers ($\ge 25$).
  - **Foreign Worker Status:** Foreign vs Domestic applicants.
- Benchmarked against the **Four-Fifths (80%) Rule** standard.
- Enforces decision-support governance: automated models never hold unilateral rejection authority without human review.

---

## 🚀 Installation & Execution

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Unit & Integration Tests
```bash
python test_system.py
```
*(All 9 tests should pass in under 5 seconds).*

### 4. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Or with custom port and host:
```bash
streamlit run app.py --server.port 8501 --server.headless true
```
Open **`http://localhost:8501`** in your browser.

---

## ⚖️ Legal & Regulatory Disclaimers
1. **Decision Support Only:** This software is intended strictly as a decision-support advisory tool for authorized credit analysts and underwriters.
2. **Non-Causality:** SHAP and LIME values quantify statistical attributions within the trained model and do not constitute formal economic or real-world causal proof.
3. **Historical Data:** The German Credit dataset reflects historical lending conventions. Institutions must re-train models on localized, compliant, and debiased portfolio data.
