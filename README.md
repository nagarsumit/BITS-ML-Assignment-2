# Mobile Price Classification - ML Assignment 2

## 1. Problem Statement
This project implements a machine learning pipeline to classify mobile phones into price ranges (0, 1, 2, 3) based on their technical specifications. The goal is to build an interactive Streamlit web application that allows users to compare the performance of six different classification algorithms on the chosen dataset.

## 2. Dataset Description
**Source:** Kaggle (Mobile Price Classification)

**Dataset Name:** `mobile_price_classification.csv`

**Description:**
The dataset contains **2000 instances** and **21 features**. It involves a **Multi-class** classification problem.

* **Target Variable:** `price_range` (0: Low Cost, 1: Medium Cost, 2: High Cost, 3: Very High Cost)
* **Key Features:**
    * `ram`: Random Access Memory in Megabytes
    * `battery_power`: Total energy a battery can store in one time measured in mAh
    * `px_height` / `px_width`: Pixel Resolution Height/Width
    * `mobile_wt`: Weight of mobile phone

## 3. Models Used
The following six machine learning models were implemented and evaluated:
1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbor (KNN)
4.  Naive Bayes (Gaussian)
5.  Random Forest (Ensemble)
6.  XGBoost (Ensemble)

## 4. Evaluation Metrics & Comparison Table
The models were evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC Score.

| ML Model Name             | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|---------------------------|----------|-----------|-----------|--------|----------|-----------|
| Logistic Regression       | 0.9525   | 0.9975    | 0.9529    | 0.9525 | 0.9521   | 0.9371    |
| Decision Tree             | 0.8200   | 0.8798    | 0.8177    | 0.8200 | 0.8180   | 0.7606    |
| KNN                       | 0.5100   | 0.7565    | 0.5160    | 0.5100 | 0.5090   | 0.3486    |
| Naive Bayes               | 0.8175   | 0.9544    | 0.8194    | 0.8175 | 0.8183   | 0.7567    |
| Random Forest (Ensemble)  | 0.8575   | 0.9802    | 0.8555    | 0.8575 | 0.8556   | 0.8106    |
| XGBoost (Ensemble)        | 0.9225   | 0.9889    | 0.9227    | 0.9225 | 0.9226   | 0.8967    |

## 5. Observations
Below are the observations on the performance of each model on the chosen dataset:

| ML Model Name             | Observation about model performance |
|---------------------------|-------------------------------------|
| Logistic Regression       | **Best Performer.** Achieved the highest accuracy (95.25%) and MCC score, indicating that the relationship between features (like RAM) and price is highly linear. |
| Decision Tree             | Moderate performance (82.00%). Likely suffered from slight overfitting compared to the ensemble methods, resulting in lower generalization on the test set. |
| KNN                       | Lowest performance (51.00%). The model struggled significantly, suggesting that even with scaling, the distance-based separation was not distinct enough for this high-dimensional data. |
| Naive Bayes               | Good performance (81.75%) with a very high AUC (0.95). It ranked classes well despite the assumption of feature independence being violated. |
| Random Forest (Ensemble)  | Strong performance (85.75%). It improved upon the single Decision Tree by reducing variance, though it trailed behind XGBoost and Logistic Regression. |
| XGBoost (Ensemble)        | **Second Best Performer.** Achieved high accuracy (92.25%) and an excellent AUC score, demonstrating the power of gradient boosting on structured tabular data. |

## 6. How to Run the App Locally

### Prerequisites
* Python 3.8 or higher installed.

### Installation Steps

**Step 1: Clone the repository**
```bash
git clone [YOUR_GITHUB_REPO_LINK]
cd [YOUR_PROJECT_FOLDER]
```

**Step 2: Install Dependencies**
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Step 3: Run the Application**
```bash
streamlit run app.py
```