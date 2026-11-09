
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load cleaned dataset
df = pd.read_csv("spotify_cleaned.csv")

# CHANGE THIS if needed.
# Example: LINEAR_TARGET = "listening_time"
LINEAR_TARGET = None

possible_targets = [
    "listening_time",
    "listening_hours",
    "total_listening_time",
    "minutes_listened",
    "songs_played_per_day",
    "ads_listened_per_week"
]

if LINEAR_TARGET is None:
    for column in possible_targets:
        if column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                LINEAR_TARGET = column
                break

if LINEAR_TARGET is None:
    raise ValueError(
        "No suitable numerical target was detected. "
        "Set LINEAR_TARGET manually to a continuous numerical column."
    )

print("Linear Regression target:", LINEAR_TARGET)

# Convert target into numerical form
df[LINEAR_TARGET] = pd.to_numeric(
    df[LINEAR_TARGET],
    errors="coerce"
)

# Remove rows where target is missing
df = df.dropna(subset=[LINEAR_TARGET]).copy()

# Separate input features and target
X = df.drop(columns=[LINEAR_TARGET])
y = df[LINEAR_TARGET]

# Remove identifier columns
id_columns = [
    column for column in X.columns
    if column.lower() in ["id", "user_id", "customer_id", "index"]
]

X = X.drop(columns=id_columns)

# Identify numerical and categorical features
numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

if len(numeric_columns) + len(categorical_columns) == 0:
    raise ValueError("No usable input features were found.")

# Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Linear Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results")
print("-------------------------")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Save evaluation results
results = pd.DataFrame({
    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        r2
    ]
})

results.to_csv("linear_results.csv", index=False)

# Save actual and predicted values
predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Residual": y_test.values - y_pred
})

predictions.to_csv("linear_predictions.csv", index=False)

# Graph 1: Actual vs predicted values
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.6)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.tight_layout()
plt.savefig("linear_actual_vs_predicted.png")
plt.show()

# Graph 2: Residual distribution
residuals = y_test.values - y_pred

plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.savefig("linear_residual_distribution.png")
plt.show()

# Graph 3: Residuals vs predicted values
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig("linear_residuals_vs_predicted.png")
plt.show()

print("\nMember 3 work completed.")
