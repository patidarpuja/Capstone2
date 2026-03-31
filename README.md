# 💳 Credit Card Fraud Detection
### End-to-End Machine Learning Pipeline on 1.85M Transactions

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Feature%20Engineering-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## 📌 Project Overview

Credit card fraud is rare — only **0.5% of all transactions** are fraudulent. This makes it one of the hardest classification problems in real-world data science because standard accuracy metrics are completely misleading. A model that predicts everything as "not fraud" achieves 99.5% accuracy but catches **zero** actual fraud.

This project builds a full end-to-end fraud detection pipeline designed specifically for highly imbalanced data, with a focus on the metrics that actually matter in a fraud detection context.

---

## 🎯 Business Problem

> How do you detect fraud reliably when fraudulent transactions represent less than 1% of your data — without building a model that simply ignores the minority class?

Financial institutions lose billions annually to credit card fraud. The challenge is not just accuracy — it is catching fraud cases (recall) without generating too many false alarms (precision). This project addresses both.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| Total transactions | 1,850,000+ |
| Fraud rate | ~0.5% |
| Working sample | 100,000 (stratified) |
| Features engineered | 15+ |
| Feature types | Behavioral, Temporal, Geospatial |

---

## 🔧 Technical Approach

### 1. Data Preprocessing
- Handled missing values and outliers
- Removed transaction amount cap to preserve real spend patterns
- Applied **stratified sampling** to reduce 1.85M records to 100K
- Preserved the original fraud distribution (0.5%) throughout
- Result: **90% reduction in training time** with no loss of fraud signal

### 2. Feature Engineering
Engineered **15+ features** across three categories:

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
Used **Precision-Recall AUC (PR-AUC)** — the correct metric for imbalanced fraud detection where missing fraud is far more costly than a false alarm.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Final Model | XGBoost |
| PR-AUC Score | **0.784** |
| Training Time Reduction | **90%** |
| Fraud Distribution Preserved | ✅ Yes |

> A PR-AUC of 0.784 on a dataset with only 0.5% fraud rate demonstrates robust minority-class detection — far more meaningful than an accuracy score.

---

## 📁 Repository Structure

```
Capstone2/
│
├── preprocessing_v1.ipynb                        # Initial data preprocessing
├── prepro_v2_wo_amnt_cap.ipynb                   # Preprocessing v2 — no amount cap
├── Data_wrangle_credit_fraud.ipynb               # Data wrangling and cleaning
├── Credit_card_EDA_final_version_last.ipynb      # Final EDA notebook
├── Credit_card_EDA_final_version_last_annotated.ipynb  # Annotated EDA walkthrough
├── V1_final_model_mentor_reviewe.ipynb           # Final model with mentor review
├── Final Credit card fraud detection report.pdf  # Full project report
├── Final credit card fraud detection.pptx        # Presentation slides
└── README.md
```

---

## 🛠️ Tech Stack

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

# Start with preprocessing
jupyter notebook preprocessing_v1.ipynb

# Then run EDA
jupyter notebook Credit_card_EDA_final_version_last.ipynb

# Then run the final model
jupyter notebook V1_final_model_mentor_reviewe.ipynb
```

---

## 💡 Key Learnings

- Standard accuracy is a **misleading metric** for imbalanced datasets
- **Stratified sampling** is critical to preserve minority class distribution
- Removing the **amount cap** in preprocessing improved model performance
- **Feature engineering** contributed more to model performance than hyperparameter tuning
- **PR-AUC** is the right evaluation metric when false negatives are costly
- XGBoost outperformed Random Forest on this dataset

---

## 👩‍💻 Author

**Puja Patidar**
Data Scientist | Data Analyst
📍 Jersey City, NJ | H4 EAD — No Sponsorship Required

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/puja-patidar-960037212)
[![GitHub](https://img.shields.io/badge/GitHub-patidarpuja-black?style=flat-square&logo=github)](https://github.com/patidarpuja)
