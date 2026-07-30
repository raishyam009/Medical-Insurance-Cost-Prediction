# ============================================
# Step 1: Import Libraries & Load Dataset
# ============================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns # pyright: ignore[reportMissingModuleSource]
except ImportError:
    sns = None
    print("Warning: seaborn is not installed. Install it with 'pip install seaborn' to enable seaborn-based plots.")

# Load the dataset
# If your notebook is inside the 'Notebook' folder:
df = pd.read_csv("insurance.csv")

# If your notebook and insurance.csv are in the same folder, use:
# df = pd.read_csv("insurance.csv")

# Display the first 5 rows
print("First 5 Rows of the Dataset:")
print(df.head())

# Display the shape of the dataset
print("\nDataset Shape (Rows, Columns):")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types
print("\nData Types:")
print(df.dtypes)

# ============================================
# Step 2: Inspect the Dataset
# ============================================

# Display complete information about the dataset
print("Dataset Information:")
print("-" * 50)
df.info()

# Display statistical summary of numerical columns
print("\nStatistical Summary:")
print("-" * 50)
print(df.describe())

# Display statistical summary of categorical columns
print("\nCategorical Columns Summary:")
print("-" * 50)
print(df.describe(include='object'))

# Display column names
print("\nColumn Names:")
print("-" * 50)
print(df.columns.tolist())

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

# ============================================
# Step 3: Data Cleaning
# ============================================

# 1. Check for missing values
print("=" * 50)
print("Missing Values")
print("=" * 50)
print(df.isnull().sum())

# 2. Check for duplicate rows
print("\n" + "=" * 50)
print("Duplicate Rows")
print("=" * 50)
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

# Remove duplicates if any
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed successfully.")
else:
    print("No duplicate rows found.")

# 3. Check for unrealistic values
print("\n" + "=" * 50)
print("Checking for Invalid Values")
print("=" * 50)

print("Age <= 0      :", (df["age"] <= 0).sum())
print("BMI <= 0      :", (df["bmi"] <= 0).sum())
print("Charges <= 0  :", (df["charges"] <= 0).sum())

# 4. Standardize categorical text
df["sex"] = df["sex"].str.lower().str.strip()
df["smoker"] = df["smoker"].str.lower().str.strip()
df["region"] = df["region"].str.lower().str.strip()

print("\nCategorical values standardized successfully.")

# 5. Display unique values
print("\nUnique values after standardization:")
print("Sex     :", df["sex"].unique())
print("Smoker  :", df["smoker"].unique())
print("Region  :", df["region"].unique())

# 6. Final dataset shape
print("\nFinal Dataset Shape:", df.shape)

# ============================================
# EDA 1: Average Charges by Smoking Status
# ============================================

plt.figure(figsize=(6,5))

sns.barplot(
    data=df,
    x="smoker",
    y="charges",
    estimator=np.mean
)

plt.title("Average Insurance Charges: Smokers vs Non-Smokers")
plt.xlabel("Smoking Status")
plt.ylabel("Average Charges")

plt.show()

# ============================================
# EDA 2: BMI vs Charges
# ============================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="bmi",
    y="charges",
    hue="smoker"
)

plt.title("BMI vs Insurance Charges")
plt.xlabel("BMI")
plt.ylabel("Charges")

plt.show()

# ============================================
# EDA 3: Correlation Heatmap
# ============================================

plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=["int64", "float64"])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# ============================================
# EDA 4: Age vs Charges
# ============================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="age",
    y="charges"
)

plt.title("Age vs Insurance Charges")
plt.xlabel("Age")
plt.ylabel("Charges")

plt.show()

# ============================================
# EDA 5: Children vs Charges
# ============================================

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="children",
    y="charges"
)

plt.title("Insurance Charges by Number of Children")
plt.xlabel("Number of Children")
plt.ylabel("Charges")

plt.show()

# ============================================
# Step 5: Feature Engineering
# ============================================

# 1. Create BMI Category
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

df["bmi_category"] = df["bmi"].apply(bmi_category)

# 2. Convert smoker to numeric
# (0 = Non-Smoker, 1 = Smoker)
df["smoker_binary"] = df["smoker"].map({
    "no": 0,
    "yes": 1
})

# 3. Create Smoker × BMI Interaction Feature
df["smoker_bmi"] = df["smoker_binary"] * df["bmi"]

# 4. Remove original smoker column
# (We already have smoker_binary, so we don't need smoker anymore.)
df = df.drop(columns=["smoker"])

# 5. One-Hot Encode Remaining Categorical Columns
df_encoded = pd.get_dummies(
    df,
    columns=["sex", "region", "bmi_category"],
    drop_first=True
)

# 6. Convert Boolean Columns to Integer (0/1)
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# ============================================
# Display Results
# ============================================

print("=" * 50)
print("Feature Engineering Completed Successfully")
print("=" * 50)

print("\nFirst 5 Rows of Encoded Dataset:")
print(df_encoded.head())

print("\nEncoded Dataset Shape:")
print(df_encoded.shape)

print("\nEncoded Columns:")
print(df_encoded.columns.tolist())

print("\nData Types:")
print(df_encoded.dtypes)
# ============================================
# Step 6: Train-Test Split & Model Building
# ============================================

# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Separate Features (X) and Target (y)
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# 2. Split the dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# 3. Create the Linear Regression model
model = LinearRegression()

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on the test data
y_pred = model.predict(X_test)

# Display information
print("=" * 50)
print("Model Training Completed Successfully!")
print("=" * 50)

print(f"Training Data Shape : {X_train.shape}")
print(f"Testing Data Shape  : {X_test.shape}")

print("\nFirst 10 Predictions:")
for actual, predicted in zip(y_test.head(10), y_pred[:10]):
    print(f"Actual: {actual:.2f} | Predicted: {predicted:.2f}")

# ============================================
# Step 7: Model Evaluation
# ============================================

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Calculate Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=" * 50)
print("Model Evaluation Metrics")
print("=" * 50)
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

# ============================================
# 2. Predicted vs Actual Plot
# ============================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linestyle='--'
)

plt.title("Actual vs Predicted Insurance Charges")
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")

plt.grid(True)

plt.show()

# ============================================
# 3. Feature Importance (Model Coefficients)
# ============================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

feature_importance["Absolute Coefficient"] = feature_importance["Coefficient"].abs()

feature_importance = feature_importance.sort_values(
    by="Absolute Coefficient",
    ascending=False
)

print("\nTop Feature Coefficients")
print("=" * 50)
print(feature_importance)

# ============================================
# 4. Top 3 Most Important Features
# ============================================

print("\nTop 3 Most Influential Features")
print("=" * 50)

for i in range(3):
    feature = feature_importance.iloc[i]["Feature"]
    coef = feature_importance.iloc[i]["Coefficient"]

    direction = "increases" if coef > 0 else "decreases"

    print(f"{i+1}. {feature}")
    print(f"   Coefficient: {coef:.2f}")
    print(f"   Interpretation: As '{feature}' increases, predicted insurance charges {direction}.")
    print()
    # ============================================
# Step 8: Save the Trained Model
# ============================================

import joblib
import os

# Create the Model folder if it doesn't exist
os.makedirs("Model", exist_ok=True)

# Save the trained Linear Regression model
joblib.dump(model, "Model/insurance_model.pkl")

print("=" * 50)
print("Model Saved Successfully!")
print("=" * 50)
print("Model File: Model/insurance_model.pkl")

# ============================================
# Verify Saved Model
# ============================================

loaded_model = joblib.load("Model/insurance_model.pkl")

print("\nSaved Model Loaded Successfully!")
print("Loaded Model:", loaded_model)

# ============================================
# Make a Sample Prediction
# ============================================

sample_prediction = loaded_model.predict(X_test.iloc[:5])

print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f} | Predicted: {sample_prediction[i]:.2f}")