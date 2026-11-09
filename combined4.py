import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("All libraries imported successfully!")
# ============================================================
#              LINEAR REGRESSION
# ============================================================
sns.set_theme(style="whitegrid")
df = pd.read_csv("spotify_churn_dataset.csv")
print("\nFirst five rows of dataset:")
print(df.head())
print("\nLast five rows of dataset:")
print(df.tail())
print("\nDataset shape (rows, columns):")
print(df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nDataset information:")
df.info()
print("\nStatistical summary:")
print(df.describe(include="all"))
print("\nData types:")
print(df.dtypes)
print("\nNumber of unique values:")
print(df.nunique())
print("\nMissing values in each column:")
print(df.isnull().sum())
missing_values = df.isnull().sum() # returns pandas series 
print("\n")
print("=" * 60)
print(missing_values)
print("\n")
print("=" * 60)
print(missing_values.sum())

if missing_values.sum() > 0:

    plt.figure(figsize=(10, 5))

    missing_values[missing_values > 0].sort_values().plot(
        kind="bar"
    )

    plt.title("Missing Values in Dataset")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()
    print("SUCCESSFULLY DONE!")

print("\nNumber of duplicate rows:")
print(df.duplicated())
print(df.duplicated().sum())
df = df.drop_duplicates()
df = df.dropna(how="all")
print("\nDataset shape after removing duplicates:")
print(df.shape)

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()
categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical columns:")
print(numeric_columns)

print("\nCategorical columns:")
print(categorical_columns)

df.to_csv("spotify_cleaned.csv", index=False)
print("\nCleaned dataset saved successfully!")

# Histogram Plot

if len(numeric_columns) > 0:

    df[numeric_columns].hist(
        figsize=(14, 10),
        bins=20
    )

    plt.suptitle("Distribution of Numerical Features")
    plt.tight_layout()

    plt.show()


# Box Plot

for column in numeric_columns[:8]:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title("Box Plot of " + column)
    plt.xlabel(column)

    plt.tight_layout()

    plt.show()

# Correlation Heatmap 

if len(numeric_columns) >= 2:

    plt.figure(figsize=(12, 8))

    correlation = df[numeric_columns].corr()  # returns pandas dataframe

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig("member1_correlation_heatmap.png")
    plt.show()

# Scatter Plot

if len(numeric_columns) >= 2:

    for i in range(min(3, len(numeric_columns) - 1)):

        x_column = numeric_columns[i]
        y_column = numeric_columns[i + 1]

        plot_data = df[[x_column, y_column]].dropna()

        if len(plot_data) > 0:

            plt.figure(figsize=(7, 5))

            sns.scatterplot(
                data=plot_data,
                x=x_column,
                y=y_column
            )

            plt.title(x_column + " vs " + y_column)

            plt.tight_layout()

            plt.savefig(
                "member1_scatter_" + str(i + 1) + ".png"
            )

            plt.show()



# Load Cleaned Dataset 

df = pd.read_csv("spotify_cleaned.csv")

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
        "Churn column not found. Set CHURN_COLUMN manually."
    )


print("\nSelected churn column:", CHURN_COLUMN)


df = df.dropna(subset=[CHURN_COLUMN]).copy()

target_values = (
    df[CHURN_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

positive_values = [
    "1", "yes", "true", "churn", "churned"
]

negative_values = [
    "0", "no", "false", "not churn", "not churned", "retained"
]

unique_values = target_values.unique().tolist()

if set(unique_values).issubset(
    set(positive_values + negative_values)
):

    y = target_values.isin(positive_values).astype(int)

elif len(unique_values) == 2:

    print("Target categories:", unique_values)
    print("Second category is treated as positive.")

    y = (target_values == unique_values[1]).astype(int)

else:

    raise ValueError(
        "Churn target must have exactly two categories."
    )


if y.nunique() != 2:

    raise ValueError(
        "Both churn classes must be present in the dataset."
    )

# Prepare Input Features 
X = df.drop(columns=[CHURN_COLUMN])

id_columns = [
    column for column in X.columns
    if column.lower() in [
        "id", "user_id", "customer_id", "index"
    ]
]

X = X.drop(columns=id_columns)

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

if len(numeric_columns) + len(categorical_columns) == 0:

    raise ValueError("No usable input features found.")


# Data Preprocessing 

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# Create Logistic Regression Model 

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Split Dataset 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# Train Model 

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")



# Make Predictions

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# Evaluate Model 

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nLOGISTIC REGRESSION RESULTS")
print("---------------------------")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Saving Results

logistic_results = pd.DataFrame({
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

logistic_results.to_csv(
    "logistic_results.csv",
    index=False
)

logistic_predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability": y_probability
})

logistic_predictions.to_csv(
    "logistic_predictions.csv",
    index=False
)


# Confusion Matrix 

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

plt.savefig("member2_confusion_matrix.png")
plt.show()


# ROC Curve 

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.tight_layout()

plt.savefig("member2_roc_curve.png")
plt.show()


# Precision-Recall Curve 

precision_values, recall_values, thresholds = (
    precision_recall_curve(y_test, y_probability)
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.tight_layout()

plt.savefig("member2_precision_recall.png")
plt.show()



# ============================================================
#              LINEAR REGRESSION
# ============================================================

# Load Cleaned Dataset 

df = pd.read_csv("spotify_cleaned.csv")
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
        "Numerical target not detected. "
        "Set LINEAR_TARGET manually."
    )
print("\nSelected Linear Regression target:", LINEAR_TARGET)

df[LINEAR_TARGET] = pd.to_numeric(
    df[LINEAR_TARGET],
    errors="coerce"
)

df = df.dropna(subset=[LINEAR_TARGET]).copy()

X = df.drop(columns=[LINEAR_TARGET])

y = df[LINEAR_TARGET]

id_columns = [
    column for column in X.columns
    if column.lower() in [
        "id", "user_id", "customer_id", "index"
    ]
]

X = X.drop(columns=id_columns)

numeric_columns = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

if len(numeric_columns) + len(categorical_columns) == 0:

    raise ValueError("No usable input features found.")

# Preprocessing Data

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# Create Linear Regression Model 

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])


# Split Dataset 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# Training Model For Linear Regressinon  

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# Make Predictions 

y_pred = model.predict(X_test)


# Evaluating Model 

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\nLINEAR REGRESSION RESULTS")
print("-------------------------")

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)

# Saving Results

linear_results = pd.DataFrame({
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

linear_results.to_csv(
    "linear_results.csv",
    index=False
)

linear_predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Residual": y_test.values - y_pred
})

linear_predictions.to_csv(
    "linear_predictions.csv",
    index=False
)


# Actual Values vs Predicted Values Graph 

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted Values")

plt.tight_layout()

plt.show()

# Loading Model Results 

logistic_results = pd.read_csv(
    "logistic_results.csv"
)

linear_results = pd.read_csv(
    "linear_results.csv"
)

logistic_predictions = pd.read_csv(
    "logistic_predictions.csv"
)

linear_predictions = pd.read_csv(
    "linear_predictions.csv"
)


# Organising both Logistic and linear Regressing results Results 

logistic_metrics = dict(zip(
    logistic_results["Metric"],
    logistic_results["Value"]
))

linear_metrics = dict(zip(
    linear_results["Metric"],
    linear_results["Value"]
))

# Printing Complete Project Results 

print("\nLOGISTIC REGRESSION PERFORMANCE")
print("-" * 40)

for metric, value in logistic_metrics.items():

    print(metric, ":", round(value, 4))


print("\nLINEAR REGRESSION PERFORMANCE")
print("-" * 40)

for metric, value in linear_metrics.items():

    print(metric, ":", round(value, 4))

#R2 Score Graph for Linear Regressing model

plt.figure(figsize=(6, 5))

plt.bar(
    ["Linear Regression"],
    [linear_metrics["R2 Score"]]
)

plt.ylabel("R2 Score")
plt.title("Linear Regression Model Performance")

plt.tight_layout()

plt.show()

# Final Project Conclusion

print("\n")
print("=" * 60)
print("                 PROJECT CONCLUSION")
print("=" * 60)

print(
    "\nLogistic Regression was used to classify Spotify users "
    "according to their churn status."
)

print(
    "\nLinear Regression was used to predict a continuous "
    "numerical variable."
)

print(
    "\nThe classification model was evaluated using Accuracy, "
    "Precision, Recall, F1 Score and ROC-AUC."
)

print(
    "\nThe regression model was evaluated using MAE, MSE, "
    "RMSE and R2 Score."
)

print(
    "\nAll model results and graphs have been saved."
)

print(
    "\nFinal combined results: final_project_results.csv"
)

print(
    "\nFinal written summary: final_project_summary.txt"
)

print("\nTEAM LEADER WORK COMPLETED SUCCESSFULLY!")

print("\n")
print("=" * 60)
print("             COMPLETE PROJECT FINISHED")
print("=" * 60)