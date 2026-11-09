
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# Load cleaned dataset
df = pd.read_csv("spotify_cleaned.csv")

# CHANGE THIS if your churn column has a different name
CHURN_COLUMN = None

possible_targets = [
    "churn",
    "churned",
    "is_churned",
    "customer_churn",
    "churn_status"
]

if CHURN_COLUMN is None:
    for column in possible_targets:
        if column in df.columns:
            CHURN_COLUMN = column
            break

if CHURN_COLUMN is None:
    raise ValueError(
        "Churn column not detected. Set CHURN_COLUMN manually "
        "to the correct column name."
    )

print("Churn target column:", CHURN_COLUMN)

# Remove rows where target is missing
df = df.dropna(subset=[CHURN_COLUMN]).copy()

# Convert target into binary values
target_values = df[CHURN_COLUMN].astype(str).str.strip().str.lower()

positive_values = [
    "1", "yes", "true", "churn", "churned"
]

negative_values = [
    "0", "no", "false", "not churn", "not churned", "retained"
]

unique_values = target_values.unique().tolist()

if set(unique_values).issubset(set(positive_values + negative_values)):
    y = target_values.isin(positive_values).astype(int)
elif len(unique_values) == 2:
    print("Target categories found:", unique_values)
    print("Assuming the second category is the positive class.")
    y = (target_values == unique_values[1]).astype(int)
else:
    raise ValueError(
        "The churn target must have exactly two categories. "
        "Check the target column and its values."
    )

# Remove the target from input features
X = df.drop(columns=[CHURN_COLUMN])

# Remove columns that are unlikely to be useful identifiers
id_columns = [
    column for column in X.columns
    if column.lower() in ["id", "user_id", "customer_id", "index"]
]

X = X.drop(columns=id_columns)

# Identify numerical and categorical columns
numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Numerical data processing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical data processing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both pipelines
preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

if len(np.unique(y_test)) == 2:
    roc_auc = roc_auc_score(y_test, y_probability)
else:
    roc_auc = np.nan

print("\nLogistic Regression Results")
print("---------------------------")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save evaluation results in CSV
results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

results.to_csv("logistic_results.csv", index=False)

# Save actual and predicted values
predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability": y_probability
})

predictions.to_csv("logistic_predictions.csv", index=False)

# Graph 1: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("logistic_confusion_matrix.png")
plt.show()

# Graph 2: ROC curve
if len(np.unique(y_test)) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, y_probability)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label="Logistic Regression")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logistic_roc_curve.png")
    plt.show()

# Graph 3: Precision-recall curve
if len(np.unique(y_test)) == 2:
    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_test, y_probability)
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recall_values, precision_values)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig("logistic_precision_recall.png")
    plt.show()

# Graph 4: Predicted churn probabilities
plt.figure(figsize=(7, 5))
plt.hist(y_probability, bins=20, edgecolor="black")
plt.xlabel("Predicted Churn Probability")
plt.ylabel("Number of Users")
plt.title("Distribution of Predicted Churn Probabilities")
plt.tight_layout()
plt.savefig("logistic_probability_distribution.png")
plt.show()

print("\nMember 2 work completed.")
