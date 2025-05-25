# 📊 Credit Risk Analysis & Modeling

A data-driven credit risk assessment project that combines exploratory data analysis (EDA) with machine learning to identify high-risk loan applicants. The project uses advanced ensemble models (XGBoost, LightGBM, CatBoost) to predict defaults with high recall and interpretable results.

---

## 🧠 Objective

To analyze customer credit data, extract actionable insights, and build robust machine learning models that help financial institutions reduce loan default risk and improve lending decisions.

---

## 🔧 Tech Stack

- **Languages**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **ML Models**: XGBoost, LightGBM, CatBoost  
- **Explainability**: SHAP (SHapley Additive Explanations)  
- **Environment**: Jupyter Notebook

---

## 📂 Dataset

- **Source**: [[Dataset Link]](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- **Size**: ~45,000 client records  
- **Features**: Demographic info, financial history, credit behavior, loan status  

---

## 🔍 Exploratory Data Analysis (EDA)

- Missing value imputation and detection  
- Outlier identification using boxplots  
- Correlation heatmaps for feature selection  
- Distribution plots of key financial indicators

---

## 🧪 Modeling & Evaluation

- **Models Used**:  
  - `XGBoostClassifier`  
  - `LGBMClassifier`  
  - `CatBoostClassifier`  

- **Preprocessing**:  
  - Label encoding for categorical variables  
  - Train/test split (80/20)  
  - Standardization for numerical features

- **Evaluation Metrics**:  
  - Recall (optimized to 81% for high-risk identification)  
  - Precision, F1 Score, ROC-AUC  
  - Confusion Matrix & Classification Report

- **Hyperparameter Tuning**:  
  - Used `GridSearchCV` to tune model parameters  
  - Reduced false negatives, achieving ~20% risk mitigation

---

## 💡 Model Explainability

- Applied **SHAP values** to interpret model decisions  
- Identified top predictors: `credit_history`, `loan_amount`, `income`, `age`, etc.  
- Visualized SHAP summary and force plots to explain predictions to non-technical stakeholders

---

## 🧩 Key Outcomes

- Achieved **81% recall** in identifying defaulters using XGBoost  
- Cleaned and prepared a high-quality modeling dataset from raw data  
- Delivered interpretable insights to support data-driven credit policy improvements  
- Simulated business impact by reducing risk exposure by ~20%

---

## 📌 Future Work

- Deploy model via REST API or Streamlit dashboard  
- Add support for real-time credit scoring  
- Train on larger or multi-source datasets

---

## 🤝 Contributions

Contributions and suggestions are welcome. Please open an issue or submit a pull request.

---

## 📃 License

This project is open-sourced under the [MIT License](LICENSE).
