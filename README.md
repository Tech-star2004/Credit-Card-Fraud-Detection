# Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview
This project aims to build a robust machine learning system to identify fraudulent credit card transactions. Credit card fraud detection is a classic **imbalanced classification** problem, as fraudulent transactions represent only a tiny fraction of total activity. The project explores data scaling, synthetic oversampling (SMOTE), and ensemble learning to achieve high detection accuracy while minimizing false alarms.



## 📊 Dataset Overview & Access
This project utilizes the **Credit Card Fraud Detection dataset** hosted on Kaggle. Because the dataset file size is approximately **143MB**, it is not stored directly in this GitHub repository. Instead, it is downloaded dynamically using the Kaggle API (`kagglehub`).

* **Access**: The data is automatically fetched from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
* **Features**: The dataset contains **30 features**:
    * `V1` through `V28`: Anonymized variables resulting from a Principal Component Analysis (PCA) transformation.
    * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
    * `Amount`: The transaction amount.
* **Target**: The `Class` variable is the response variable (1 for **Fraud**, 0 for **Legitimate**).
* **Imbalance**: The dataset is highly imbalanced, with the minority class (Fraud) accounting for only **492** out of **284,807** total transactions (~0.172%).

## 🛠️ Project Workflow
1.  **Exploratory Data Analysis (EDA)**: Analyzing feature correlations and class distributions.
2.  **Data Preprocessing**: Using `StandardScaler` to normalize the `Amount` and `Time` features.
3.  **Handling Imbalance**: Implementing **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training set.
4.  **Modeling**:
    * **Logistic Regression**: Established as a baseline model.
    * **Logistic Regression with SMOTE**: Focused on improving recall.
    * **Random Forest Classifier**: Utilized as a high-performance ensemble method.
5.  **Evaluation**: Comparative analysis of Precision, Recall, and F1-Score.

## 🚀 Key Results & Insights



| Model | Recall (Detection) | Precision (Reliability) | Business Insight |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.92 | 0.06 | High sensitivity but high false alarm rate. |
| **Random Forest** | **0.84** | **0.90** | **Best Balance.** High detection with very few false positives. |

### 🔍 Key Takeaways
* **Accuracy is Misleading**: Due to class imbalance, accuracy does not reflect a model's ability to catch fraud.
* **Recall vs. Precision**: While Logistic Regression caught more fraud, it flagged over 1,400 legitimate transactions incorrectly. **Random Forest** only produced 9 false alarms.
* **Critical Features**: Features $V14$, $V10$, and $V12$ were identified as the most significant predictors of fraud.

## 💻 Tech Stack
* **Language**: Python
* **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE), Matplotlib, Seaborn

## 🚀 How to Run
1.  Clone this repository.
2.  Install the required libraries: `pip install kagglehub pandas scikit-learn imbalanced-learn seaborn matplotlib`.
3.  Open the `.ipynb` file in VS Code or Jupyter.
4.  When running the data loading cell, enter your **Kaggle Username** and **Kaggle Key (KGAT token)** when prompted.****

## 🏁 Conclusion
The **Random Forest Classifier** combined with SMOTE oversampling provides the best balance for financial institutions. It successfully identifies the majority of fraudulent transactions (84% recall) while maintaining high precision to ensure legitimate customers are not inconvenienced.
