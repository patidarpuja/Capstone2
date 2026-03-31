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

### 1. Data Sampling Strategy
- Original dataset: **1.85M transactions**
- Applied **stratified sampling** to reduce to 100K records
- Preserved the original fraud distribution (0.5%) throughout
- Result: **90% reduction in training time** with no loss of fraud signal

### 2. Feature Engineering
Engineered **15+ features** across three categories:

| Category | Examples |
|----------|---------|
| Behavioral | Transaction frequency, average spend deviation, velocity |
| Temporal | Hour of day, day of week, time since last transaction |
| Geospatial | Distance from home location, cross-border flag |

### 3. Model Selection
Compared multiple classifiers using **stratified cross-validation**:
- Logistic Regression (baseline)
- Random Forest
- **XGBoost** (selected as final model)

### 4. Evaluation Metric
Used **Precision-Recall AUC (PR-AUC)** instead of accuracy — the correct metric for imbalanced fraud detection where the cost of missing fraud far outweighs the cost of a false alarm.

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

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing, cross-validation, baseline models |
| XGBoost | Final classification model |
| Matplotlib / Seaborn | EDA and evaluation plots |
| Jupyter Notebook | Development environment |

---

## 📁 Project Structure

```
Capstone2/
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb  # Feature creation and selection
│   ├── 03_Modeling.ipynb             # Model training and evaluation
│   └── 04_Final_Results.ipynb        # Final model and conclusions
│
├── data/
│   └── README.md                     # Data source description
│
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/patidarpuja/Capstone2.git
cd Capstone2

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

---

## 💡 Key Learnings

- Standard accuracy is a **misleading metric** for imbalanced datasets
- **Stratified sampling** is critical to preserve minority class distribution
- **Feature engineering** contributed more to model performance than hyperparameter tuning
- **PR-AUC** is the right evaluation metric when false negatives are costly
- XGBoost outperformed Random Forest on this dataset due to its handling of class imbalance

---

## 👩‍💻 Author

**Puja Patidar**
Data Scientist | Data Analyst
📍 Jersey City, NJ | H4 EAD — No Sponsorship Required

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/puja-patidar-960037212)
[![GitHub](https://img.shields.io/badge/GitHub-patidarpuja-black?style=flat-square&logo=github)](https://github.com/patidarpuja)
