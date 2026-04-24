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

<<<<<<< HEAD
**Key Findings:**
- No missing data or duplicates
- Transaction amounts are **right-skewed** (log transformation needed)
- Age distribution roughly normal with mean ~44 years
- Fraud transactions are rare and scattered across all categories
=======
| Category | Examples |
|----------|---------|
| Behavioral | Transaction frequency, average spend deviation, velocity |
| Temporal | Hour of day, day of week, time since last transaction |
| Geospatial | Distance from home location, cross-border flag |

### 3. Exploratory Data Analysis
- Analyzed fraud vs non-fraud transaction distributions
- Identified key behavioral patterns in fraudulent transactions
- Visualized class imbalance and feature correlations

### 4. Model Development
Compared multiple classifiers using **stratified cross-validation**:
- Logistic Regression (baseline)
- Random Forest
- **XGBoost** (selected as final model)

### 5. Model Review & Iteration
- Mentor review incorporated into final model version
- Annotated notebook created for clear walkthrough of methodology

### 6. Evaluation Metric
Used **Precision-Recall AUC (PR-AUC)** — the correct metric for imbalanced fraud detection, where missing fraud is far more costly than a false alarm.
>>>>>>> 1106fdc7462ed54f4395cf7c57043a0feec10737

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

<<<<<<< HEAD
**Advanced Features Created:**
=======
```
Capstone2/
│
├── Data_wrangle_credit_fraud.ipynb               # Data wrangling and cleaning
├── Credit_card_EDA_final_version_last.ipynb      # Final EDA notebook
├── Credit_card_EDA_final_version_last_annotated.ipynb  # Annotated EDA walkthrough
├── preprocessing_v1.ipynb                        # Initial data preprocessing
├── prepro_v2_wo_amnt_cap.ipynb                   # Preprocessing v2 — no amount cap
├── V1_final_model_mentor_reviewe.ipynb           # Final model with mentor review
├── Final Credit card fraud detection report.pdf  # Full project report
├── Final credit card fraud detection.pptx        # Presentation slides
└── README.md
```
>>>>>>> 1106fdc7462ed54f4395cf7c57043a0feec10737

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

<<<<<<< HEAD
**Final Features (13 features):**
=======
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data manipulation and preprocessing |
| Scikit-learn | Preprocessing, cross-validation, baseline models |
| XGBoost | Final classification model |
| Matplotlib / Seaborn | EDA and evaluation plots |
| Jupyter Notebook | Development environment |

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/patidarpuja/Capstone2.git
cd Capstone2

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

# Start with Data Wrangle
Data_wrangle_credit_fraud.ipynb

# Then run EDA
jupyter notebook Credit_card_EDA_final_version_last.ipynb

# Then run the final model
jupyter notebook V1_final_model_mentor_reviewe.ipynb
```

---

## 💡 Key Learnings

- Standard accuracy is a **misleading metric** for imbalanced datasets
- **Stratified sampling** is critical to preserve minority class distribution
- **Feature engineering** contributed more to model performance than hyperparameter tuning
- **PR-AUC** and  **Recal** is the right evaluation metric when false negatives are costly
- XGBoost outperformed Random Forest on this dataset

---

## 👩‍💻 Author

**Puja Patidar**
Data Scientist | Data Analyst
📍 Jersey City, NJ | H4 EAD — No Sponsorship Required

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/puja-patidar-960037212)
[![GitHub](https://img.shields.io/badge/GitHub-patidarpuja-black?style=flat-square&logo=github)](https://github.com/patidarpuja)
>>>>>>> 1106fdc7462ed54f4395cf7c57043a0feec10737
