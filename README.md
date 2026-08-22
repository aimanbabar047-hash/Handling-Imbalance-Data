# Handling Imbalanced Data — Credit Card Fraud Detection

## 📌 Project Overview

This project focuses on handling **highly imbalanced real-world data** using a Credit Card Fraud Detection dataset.

Fraud detection is a classic imbalanced classification problem because fraudulent transactions are extremely rare compared with normal transactions. In this project, different techniques were evaluated to understand how class imbalance affects machine learning performance.

The project compares:

- Baseline Logistic Regression
- Logistic Regression with SMOTE
- Logistic Regression with Class Weighting

The models were evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and Precision-Recall Curve.

---

## 🎯 Objectives

- Identify class imbalance in a real-world dataset
- Understand why accuracy can be misleading for imbalanced classification
- Build a baseline classification model
- Apply SMOTE to balance the training data
- Apply class weighting
- Compare different approaches using appropriate evaluation metrics
- Analyze the trade-off between fraud detection and false alarms

---

## 📊 Dataset

The **Credit Card Fraud Detection** dataset was obtained using KaggleHub.

Dataset size:

- Total transactions: **284,807**
- Normal transactions: **284,315 (99.83%)**
- Fraudulent transactions: **492 (0.17%)**

### Target Variable

| Class | Meaning |
|------:|---------|
| 0 | Normal transaction |
| 1 | Fraudulent transaction |

The dataset is therefore **highly imbalanced**.

---

## 🛠️ Technologies & Libraries

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- KaggleHub

### Machine Learning Techniques

- Logistic Regression
- StandardScaler
- SMOTE
- Class Weighting

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

---

## 🔬 Methodology

### 1. Data Loading

The dataset was downloaded using KaggleHub and loaded using Pandas.

### 2. Class Distribution Analysis

The target variable was analyzed to identify the degree of class imbalance.

### 3. Train-Test Split

The dataset was divided into training and testing sets using an 80/20 split with stratification.

### 4. Baseline Model

A Logistic Regression model was trained without applying an imbalance-handling technique.

### 5. SMOTE

SMOTE (Synthetic Minority Over-sampling Technique) was applied only to the training data to generate synthetic minority-class samples.

Before SMOTE:

- Normal: 227,451
- Fraud: 394

After SMOTE:

- Normal: 227,451
- Fraud: 227,451

### 6. Class Weighting

A second approach used Logistic Regression with:

`class_weight="balanced"`

This gives more importance to the minority fraud class during training without generating synthetic samples.

### 7. Model Evaluation

All models were evaluated on the same original test set.

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 99.91% | 82.67% | 63.27% | 71.68% | 0.9605 |
| SMOTE | 98.99% | 13.41% | 89.80% | 23.34% | **0.9765** |
| Class Weighted | 97.55% | 6.10% | **91.84%** | 11.44% | 0.9721 |

---

## 🔍 Key Findings

### Baseline

The baseline model achieved very high accuracy (99.91%) and high precision (82.67%), but its recall was only 63.27%.

This means that a significant number of fraudulent transactions were missed.

### SMOTE

SMOTE increased recall from **63.27% to 89.80%** and achieved the highest ROC-AUC of **0.9765**.

However, precision decreased to 13.41% because the model generated more false positives.

### Class Weighting

Class weighting achieved the highest recall of **91.84%**, detecting more fraudulent transactions than the other approaches.

However, it produced a large number of false positives, resulting in a precision of only 6.10%.

---

## ⚖️ Important Insight

The experiment demonstrates that there is no single metric that is sufficient for evaluating highly imbalanced classification problems.

A model can achieve extremely high accuracy while still missing a considerable number of minority-class cases.

For fraud detection, **Recall, Precision, F1-score, ROC-AUC, and the Confusion Matrix** provide more meaningful insights than accuracy alone.

---

## 🏆 Conclusion

Among the imbalance-handling techniques tested, **SMOTE provided the most favorable overall trade-off** in this experiment.

It significantly improved fraud detection recall while achieving the highest ROC-AUC score.

However, the appropriate model ultimately depends on the relative cost of:

- Missing fraudulent transactions
- Generating false fraud alerts

This project demonstrates the importance of properly identifying and handling class imbalance in real-world machine learning problems.

---

## 📂 Project Structure

```text
credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb
├── README.md
