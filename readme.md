# 🏥 Medical Insurance Cost Prediction

## 📌 Project Overview

This project predicts an individual's annual medical insurance charges using Machine Learning. A **Linear Regression** model is trained on personal information such as age, BMI, number of children, smoking status, sex, and region to estimate medical insurance costs.

The project covers the complete Machine Learning workflow including data cleaning, exploratory data analysis (EDA), feature engineering, model building, evaluation, and feature impact analysis.

---

# 📖 Problem Statement

An insurance company wants to estimate a customer's annual medical insurance charges based on personal attributes such as age, BMI, number of children, smoking status, and region.

---

# 🎯 Business Objective

The objective of this project is to build a regression model that predicts insurance charges accurately. The model helps insurance companies:

* Estimate medical insurance charges.
* Understand the major factors affecting insurance costs.
* Set fair insurance premiums.
* Analyze the impact of lifestyle choices such as smoking.

---

# 📂 Dataset Information

* **Dataset Name:** Medical Cost Personal Dataset
* **Source:** Kaggle
* **Rows:** 1338 (1337 after removing duplicates)
* **Columns:** 7

### Features

| Feature  | Description                                        |
| -------- | -------------------------------------------------- |
| age      | Age of the insured person                          |
| sex      | Gender                                             |
| bmi      | Body Mass Index                                    |
| children | Number of dependent children                       |
| smoker   | Smoking status                                     |
| region   | Residential region                                 |
| charges  | Annual medical insurance charges (Target Variable) |

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

# 🔄 Project Workflow

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Train-Test Split
6. Linear Regression Model
7. Model Evaluation
8. Feature Impact Analysis
9. Save Trained Model

---

# 🧹 Data Cleaning

The following preprocessing steps were performed:

* Checked for missing values.
* Removed duplicate records.
* Verified that age, BMI, and insurance charges contain valid positive values.
* Standardized categorical values.
* Converted categorical features into numerical format using One-Hot Encoding.

---

# 📊 Exploratory Data Analysis

The following visualizations were created:

* Average Insurance Charges (Smokers vs Non-Smokers)
* BMI vs Insurance Charges
* Correlation Heatmap
* Age vs Insurance Charges
* Children vs Insurance Charges

---

# ⚙ Feature Engineering

The following features were created:

* BMI Category
* Smoker Binary Feature
* Smoker × BMI Interaction Feature
* One-Hot Encoding for categorical variables

---

# 🤖 Machine Learning Model

**Algorithm Used**

* Linear Regression

### Train-Test Split

* Training Data: 80%
* Testing Data: 20%

Random State = 42

---

# 📈 Model Evaluation

| Metric                         | Value       |
| ------------------------------ | ----------- |
| R² Score                       | **0.8868**  |
| Mean Absolute Error (MAE)      | **2813.07** |
| Root Mean Squared Error (RMSE) | **4560.55** |

### Interpretation

The model explains approximately **88.68%** of the variation in insurance charges. The average prediction error is around **2813**, indicating that the model performs well on unseen data.

---

# 🔍 Feature Impact Analysis

The Linear Regression coefficients indicate that some features contribute more strongly to predicted insurance charges than others.

### Smoking Status

Smoking is the most influential factor in predicting insurance charges. Smokers are predicted to have significantly higher insurance costs compared to non-smokers, even after accounting for age, BMI, and other personal attributes.

### BMI

BMI has a positive relationship with insurance charges. Individuals with higher BMI values generally have higher predicted medical expenses, and the effect becomes stronger when combined with smoking.

### Age

Insurance charges generally increase with age because older individuals are more likely to require medical care.

### Children

The number of children has a relatively small positive influence on insurance charges.

### Region

Regional differences exist but have a smaller impact than smoking, BMI, and age.

---

# 📷 Project Images

Include screenshots of the following plots inside the **Images** folder.

* Smoker vs Non-Smoker Charges
* BMI vs Charges
* Correlation Heatmap
* Age vs Charges
* Predicted vs Actual Charges

Example:

```
Images/
│
├── smoker_barplot.png
├── bmi_vs_charges.png
├── correlation_heatmap.png
├── age_vs_charges.png
└── predicted_vs_actual.png
```

---

# 📁 Project Structure

```
Medical-Insurance-Cost-Prediction/
│
├── Dataset/
│   └── insurance.csv
│
├── Notebook/
│   └── Medical_Insurance_Cost_Prediction.ipynb
│
├── Images/
│   ├── smoker_barplot.png
│   ├── bmi_vs_charges.png
│   ├── correlation_heatmap.png
│   ├── age_vs_charges.png
│   └── predicted_vs_actual.png
│
├── Model/
│   └── insurance_model.pkl
│
├── README.md
│
└── requirements.txt
```

---

# 🚀 Future Improvements

* Apply Random Forest Regression.
* Try XGBoost Regressor.
* Perform Hyperparameter Tuning.
* Use Cross Validation.
* Build a Streamlit web application for insurance charge prediction.

---

# ✅ Conclusion

A Linear Regression model was successfully developed to predict medical insurance charges. The model achieved an **R² Score of 0.8868**, demonstrating strong predictive performance. Feature analysis showed that **smoking status** is the most influential factor affecting insurance charges, followed by BMI and age. This project demonstrates the complete end-to-end Machine Learning workflow and provides valuable insights for insurance pricing.

---

# 👨‍💻 Author

**Name:** Your Name

**Roll Number:** XXXXX

**Course:** B.Tech (CSE)

**Project:** Medical Insurance Cost Prediction

**GitHub:** https://github.com/your-username
