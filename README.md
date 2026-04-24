# 💳 Credit Card Fraud Detection
### End-to-End Machine Learning Pipeline on 1.85M Transactions

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Feature%20Engineering-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Project Overview

This project implements a **complete end-to-end fraud detection pipeline** for credit card transactions on a highly imbalanced dataset of **1.85 million transactions** where only **0.5% are fraudulent**. The pipeline addresses one of the most challenging real-world classification problems: detecting rare fraudulent transactions without sacrificing performance on the majority class.

Using advanced feature engineering, proper handling of class imbalance, and correct evaluation metrics, this project achieves a **PR-AUC score of 0.784**, demonstrating robust minority-class detection.

---

## 🎯 Business Problem

> How do you reliably detect credit card fraud when fraudulent transactions represent less than 1% of your data—without building a model that ignores fraud or generates excessive false alarms?

Financial institutions lose billions annually to fraud. The challenge is not just accuracy—it's **maximizing fraud detection (recall) while minimizing false alarms (precision)**, and understanding the trade-offs between these metrics based on business costs.

---

## 📊 Dataset Overview

| Property | Detail |
|----------|--------|
| **Total Transactions** | 1,852,394 |
| **Fraud Rate** | ~0.52% (9,651 frauds) |
| **Working Sample** | 100,000 (stratified) |
| **Features Engineered** | 13 core features |
| **Feature Categories** | Behavioral, Temporal, Geospatial, Financial |
| **Data Period** | 2012-2013 (2 years) |

---

## 🔧 Technical Approach

### **1. Data Wrangling** (`Data_wrangle_credit_fraud.ipynb`)

**Steps:**
- **Merged** train and test datasets (1,296,675 + 555,719 records)
- **Validated** data integrity: 0 missing values, 0 duplicates
- **Extracted** temporal features:
  - `year`, `month`, `day`, `hour`, `minute` from transaction timestamp
  - `tr_day_name` (day of week)
  - Day of week extracted using `dayofweek()` method
- **Calculated** cardholder age at transaction time
- **Created** age groups: teen (≤18), adult (19-50), senior (>50)
- **Removed** unnecessary columns (Unnamed: 0, high-cardinality identifiers)
- **Output:** Clean dataset saved as CSV for analysis

**Key Findings:**
- No missing data or duplicates
- Transaction amounts are **right-skewed** (log transformation needed)
- Age distribution roughly normal with mean ~44 years
- Fraud transactions are rare and scattered across all categories

---

### **2. Exploratory Data Analysis** (`Credit_card_EDA_final_version_last.ipynb`)

**Analysis Performed:**

#### **Fraud Distribution Analysis:**
- Fraud rate: **0.52%** (9,651 out of 1.85M transactions)
- **Severe class imbalance** → Standard accuracy is misleading
- Frauds concentrated in specific merchant categories and geographic regions

#### **Transaction Amount Analysis:**
- **Mean:** $70.06
- **Median:** $47.45
- **Range:** $1 - $28,948.90
- Right-skewed distribution with outliers at high-value transactions
- **Log transformation applied** to normalize distribution

#### **Outlier Analysis:**
- Used IQR method: Q1=$9.64, Q3=$83.10
- **41,386 outlier transactions** detected
- Merchants with highest outlier amounts:
  - fraud_Boyer PLC (198K+)
  - fraud_Gislason Group (194K+)
  - fraud_Kuhn LLC (189K+)
- **Top fraud categories:** shopping_net, misc_net
- Geographic concentrations in Meridian, Brandon, Houston

#### **Statistical Tests:**
- Applied SMOTE for class imbalance handling
- Correlation analysis: Amount features most predictive of fraud
- Temporal patterns examined (hour, day of week, month)

---

### **3. Feature Engineering & Preprocessing** (`V1_final_model_mentor_reviewe.ipynb`)

**Advanced Features Created:**

| Feature | Description | Business Logic |
|---------|-------------|-----------------|
| `amt_log` | Log-transformed amount | Normalizes right-skewed distribution |
| `is_weekend` | Binary (0=weekday, 1=weekend) | Fraud patterns differ by day type |
| `is_night` | Binary (1=10pm-5am) | Suspicious transactions occur at night |
| `txns_last_24h` | 24-hour rolling transaction count per card | High velocity = suspicious |
| `amt_mean_20` | Rolling mean of last 20 transactions | Baseline spending behavior |
| `amt_vs_mean_ratio` | Current amt / recent average | Detects unusual spend amounts |
| `distance_cust_merchant_km` | Geographic distance traveled | Impossible travel detection |

**Feature Selection Strategy:**
- Dropped **high-cardinality identifiers**: cc_num, merchant, city, first, last, street, dob, job
- Dropped **raw calendar splits** (tr_day, tr_month) to reduce memorization
- **Correlation with fraud** (top features):
  - `amt_mean_20`: 0.246
  - `amt`: 0.209
  - `amt_vs_mean_ratio`: 0.118
  - `amt_log`: 0.114
  - `is_night`: 0.089

**Final Features (13 features):**