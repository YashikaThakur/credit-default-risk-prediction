# Credit Card Default Risk Prediction Studio

## a. Problem Statement
Credit card default prediction is a critical financial risk problem. Lending institutions need to accurately identify borrowers who are likely to miss upcoming payments so that credit limits can be proactively adjusted, minimizing credit losses and non-performing assets.

## b. Dataset Description
* **Dataset**: UCI Default of Credit Card Clients (UCI ID: 350)
* **Instances**: 30,000 customer records (10,000 stratified instances sampled for training and evaluation; 80% train / 20% test).
* **Features (23 attributes)**:
  * Demographic factors: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`.
  * Repayment status across six months: `PAY_0` through `PAY_6`.
  * Statement amounts: `BILL_AMT1` through `BILL_AMT6`.
  * Prior payments: `PAY_AMT1` through `PAY_AMT6`.
* **Target**: `default_payment_next_month` (0 = Non-default, 1 = Default).

## c. GitHub Repository Link
https://github.com/<your-username>/credit-default-risk-prediction

## d. Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8075 | 0.7417 | 0.7209 | 0.2104 | 0.3257 | 0.3164 |
| **Decision Tree** | 0.8235 | 0.7239 | 0.6894 | 0.3665 | 0.4786 | 0.4119 |
| **K-Nearest Neighbors** | 0.8070 | 0.7053 | 0.6443 | 0.2828 | 0.3931 | 0.3344 |
| **Gaussian Naive Bayes** | 0.7760 | 0.7322 | 0.4941 | 0.5679 | 0.5284 | 0.3841 |
| **Support Vector Machine** | 0.8180 | 0.7315 | 0.6990 | 0.3100 | 0.4295 | 0.3797 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivers the highest AUC (0.7417) and strong precision (72.09%), but demonstrates low recall (21.04%) due to conservative linear probability calibration on the minority default class. |
| **Decision Tree** | Yields the highest Accuracy (82.35%) and the highest MCC (0.4119). Its non-linear threshold splits on sequential repayment delay variables capture defaulting risk patterns cleanly without excessive overfitting. |
| **K-Nearest Neighbors** | Moderate overall accuracy (80.70%), but its Euclidean distance metric struggles in 23-dimensional space to reliably isolate sparse default clusters, resulting in a modest recall of 28.28%. |
| **Gaussian Naive Bayes** | Achieves the highest Recall (56.79%) and highest F1 Score (0.5284) by detecting more positive default cases, though correlated financial statement balances lower its precision to 49.41%. |
| **Support Vector Machine** | RBF kernel separation yields balanced metrics with 81.80% accuracy and 69.90% precision, serving as a reliable non-linear margin classifier. |
| **Overall Winner for your dataset?** | **Decision Tree Classifier** is the overall winner for this benchmark. It achieves the top Accuracy (82.35%) and the highest MCC (0.4119), effectively separating credit defaults while maintaining high precision (68.94%). |
