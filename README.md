# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

**End-to-end machine learning pipeline for detecting credit card fraud on 1.85M transactions with 0.52% fraud rate**

---

## 📋 Quick Overview

| Aspect | Details |
|--------|---------|
| **Dataset** | 1,852,394 transactions (0.52% fraud) |
| **Model** | XGBoost |
| **PR-AUC Score** | 0.784 |
| **Precision** | 75% |
| **Recall** | 68% |
| **Final Features** | 13 engineered features |
| **Data Reduction** | 94.6% (1.85M → 100K, no signal loss) |

---

## 🎯 Business Problem

Detect fraudulent credit card transactions when fraud represents only 0.52% of data. Standard accuracy metrics fail on this imbalanced data - a model predicting all transactions as non-fraud achieves 99.5% accuracy but catches zero frauds.

---

## 📊 Dataset

- **Total Transactions**: 1,852,394
- **Fraud Cases**: 9,651 (0.52%)
- **Legitimate**: 1,842,743 (99.48%)
- **Time Period**: 2 years (2012-2013)
- **Data Quality**: 0 missing values, 0 duplicates
- **Original Features**: 23
- **Engineered Features**: 13

---

## 🔧 What I Did

### 1. Data Wrangling (`Data_wrangle_credit_fraud.ipynb`)

- Merged train (1.3M) + test (0.5M) datasets
- Extracted temporal features: year, month, day, hour, minute
- Calculated cardholder age from DOB
- Created age groups (teen, adult, senior)
- Removed high-cardinality columns (merchant, city, job, etc.)

### 2. Exploratory Data Analysis (`Credit_card_EDA_final_version_last.ipynb`)

- Analyzed fraud distribution: **0.52% fraud rate**
- Transaction amounts: right-skewed distribution
  - Mean: $70.06, Median: $47.45, Max: $28,948.90
  - Applied log transformation for normalization
- Detected 41,386 outlier transactions using IQR method
- Identified fraud patterns by category, city, merchant

### 3. Feature Engineering (`V1_final_model_mentor_reviewe.ipynb`)

Created 13 predictive features:

| Feature | Description | Correlation with Fraud |
|---------|-------------|------------------------|
| `amt_mean_20` | Rolling mean of last 20 transactions | 0.246 |
| `amt` | Transaction amount | 0.209 |
| `amt_vs_mean_ratio` | Current amount / average amount | 0.118 |
| `amt_log` | Log-transformed amount | 0.114 |
| `is_night` | 1 if transaction between 10 PM-5 AM | 0.089 |
| `is_weekend` | 1 if weekend transaction | - |
| `txns_last_24h` | Transaction count in last 24 hours | - |
| `distance_cust_merchant_km` | Geographic distance | - |
| `tr_hour` | Hour of transaction (0-23) | - |
| `category` | Merchant category | - |
| `state` | Geographic location | - |
| `gender` | Cardholder gender | - |
| `age` | Cardholder age | - |

**Feature Selection**: Dropped high-cardinality identifiers (cc_num, merchant, city, job) to reduce overfitting.

### 4. Handling Class Imbalance

**Problem**: 99.48% legitimate vs 0.52% fraudulent

**Solutions Applied**:

1. **Stratified Sampling**: Reduced 1.85M → 100K while preserving 0.52% fraud ratio
   - Result: 90% faster training with no signal loss

2. **Stratified Train-Test Split** (80-20):
   - Train: 80K records (0.521% fraud)
   - Test: 20K records (0.52% fraud)

3. **SMOTE During Training**: Created synthetic fraud samples for balanced training

### 5. Preprocessing

- **Encoding**: OneHotEncoder for categorical features, StandardScaler for numerical
- **Handling Missing Values**: 1,998 nulls from rolling window filled with global median
- **No Data Leakage**: Scaler fit only on training data, applied to test

### 6. Model Development

**Models Compared**:
- Logistic Regression: PR-AUC = 0.70
- Random Forest: PR-AUC = 0.75
- **XGBoost: PR-AUC = 0.784** ✅ (Selected)

**Why XGBoost**: Best performance, fast training, handles imbalanced data well.

**Cross-Validation**: StratifiedKFold (5 folds) to maintain fraud ratio in each split.

---

## 📈 Results

### Test Set Performance (20,000 transactions)

| Metric | Value |
|--------|-------|
| **Precision-Recall AUC** | **0.784** |
| **Precision** | 75% |
| **Recall** | 68% |
| **F1-Score** | 0.71 |
| **ROC-AUC** | 0.962 |

