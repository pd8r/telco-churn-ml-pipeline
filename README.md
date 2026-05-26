# 📊 Telco Customer Churn Prediction Pipeline

An end-to-end supervised machine learning project aimed at predicting customer churn for a telecommunications company. This repository contains a robust classification pipeline that prioritizes data integrity (preventing data leakage), handles class imbalance, and translates complex model metrics into actionable business strategies.

## 🗂️ Dataset
The project utilizes the **Telco Customer Churn** dataset.
* **Size:** 7,043 customers | 21 features
* **Target Variable:** `Churn` (Imbalanced: ~73.5% Retained / ~26.5% Churned)
* **Key Feature Categories:** Demographics, Account Information, and Services Subscribed.
* 📥 **Source:** [Telco Customer Churn Dataset (Kaggle/IBM)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 🛠️ Machine Learning Pipeline

The notebook is structured into a rigorous, production-minded pipeline:

1. **Exploratory Data Analysis (EDA):** Statistical profiling, outlier analysis (IQR), and visual mapping of numerical/categorical features against churn rates.
2. **Data Preprocessing:** 
   * Handled missing values in `TotalCharges`.
   * One-Hot Encoding for categorical variables (avoiding the dummy variable trap).
   * **Data Leakage Prevention:** `StandardScaler` was fitted *strictly* on the training data before transforming validation/test sets.
3. **Stratified Data Splitting:** 80% Train / 10% Validation / 10% Test. Stratification was enforced to maintain the 73/27 class imbalance across all subsets.
4. **Model Training:** Initialized 8 distinct algorithms across 5 families (Linear, Tree-based, Ensembles, SVM, Instance-based).
5. **Hyperparameter Tuning:** Utilized `GridSearchCV` optimizing for **ROC-AUC** to properly evaluate models on imbalanced data.
6. **Evaluation & Critical Analysis:** Assessed via Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices, and Overfitting checks (Train vs. Val gaps).

## 📈 Key EDA Findings
* **Tenure** is the strongest numerical predictor; new customers are vastly more likely to churn.
* **Month-to-month contracts** and **Electronic check** payment methods exhibit the highest churn rates (~43% and ~45% respectively).
* Customers lacking **Tech Support** or **Online Security** services churn at nearly triple the rate of those who have them.
* *Multicollinearity Note:* `TotalCharges` and `tenure` are highly correlated (0.83), but `TotalCharges` was retained as tree-based models are immune to multicollinearity and it holds independent billing magnitude value.

## 🏆 Model Performance

After rigorous GridSearch tuning, the models were ranked by **ROC-AUC** (the preferred metric for imbalanced datasets).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting** 🥇 | **0.7966** | **0.6507** | 0.5080 | 0.5706 | **0.8226** |
| AdaBoost | 0.7824 | 0.6232 | 0.4599 | 0.5292 | 0.8213 |
| Random Forest | 0.7795 | 0.6127 | 0.4652 | 0.5289 | 0.8195 |
| Logistic Regression | 0.7895 | 0.6226 | 0.5294 | **0.5723** | 0.8151 |
| SVM (Linear) | 0.7866 | 0.6178 | 0.5187 | 0.5640 | 0.8065 |

* **Overfitting Check:** Random Forest and KNN showed signs of overfitting (Train/Val accuracy gaps > 0.05). Gradient Boosting proved to be the most stable with a negligible generalization gap.

## 💼 Business Perspective & Recommendations

In a churn scenario, **False Negatives** (failing to identify a customer who leaves) are far more expensive than **False Positives** (giving a retention discount to a loyal customer). Therefore, **Recall** must be prioritized.

**Strategic Recommendations:**
1. **Deploy Gradient Boosting** as the core predictive engine due to its superior ROC-AUC and stability.
2. **Threshold Tuning:** Lower the default classification threshold from `0.50` to ~`0.35` to intentionally boost Recall and cast a wider net over at-risk customers.
3. **Targeted Interventions:** Route customers flagged with high churn probability to specialized retention campaigns, specifically targeting those on month-to-month contracts lacking security/support add-ons.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/pd8r/telco-churn-ml-pipeline.git
   cd telco-churn-ml-pipeline

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Open the Jupyter Notebook and run the cells sequentially:
   ```bash
   jupyter notebook "Supervised Learning (Classification Project).ipynb"
