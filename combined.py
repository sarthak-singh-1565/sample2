
# ============================================================
#                 SPOTIFY CHURN ANALYSIS
#                       AI101 PROJECT
# ============================================================
#
# MEMBER 1: Data Loading, Cleaning and Data Analysis
# MEMBER 2: Logistic Regression - Churn Classification
# MEMBER 3: Linear Regression - Numerical Prediction
# MEMBER 4: Team Leader - Integration and Final Analysis
#
# ============================================================


# ============================================================
#                  IMPORTING LIBRARIES
# ============================================================

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

sns.set_theme(style="whitegrid")


# ============================================================
#                    MEMBER 1
#          DATA LOADING, CLEANING AND ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("MEMBER 1: DATA LOADING AND DATA ANALYSIS")
print("=" * 60)


# -------------------- 1. Loading Dataset --------------------

df = pd.read_csv("spotify_churn_dataset.csv")

print("\nFirst five rows of dataset:")
print(df.head())

print("\nLast five rows of dataset:")
print(df.tail())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# -------------------- 2. Understanding Dataset --------------------

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nData types:")
print(df.dtypes)

print("\nNumber of unique values:")
print(df.nunique())


# -------------------- 3. Checking Missing Values --------------------

print("\nMissing values in each column:")
print(df.isnull().sum())

missing_values = df.isnull().sum()

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

    plt.savefig("member1_missing_values.png")
    plt.show()


# -------------------- 4. Checking Duplicates --------------------

print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

df = df.dropna(how="all")

print("\nDataset shape after removing duplicates:")
print(df.shape)


# -------------------- 5. Numerical and Categorical Columns --------------------

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


# -------------------- 6. Saving Cleaned Dataset --------------------

df.to_csv("spotify_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")


# -------------------- 7. Histograms --------------------

if len(numeric_columns) > 0:

    df[numeric_columns].hist(
        figsize=(14, 10),
        bins=20
    )

    plt.suptitle("Distribution of Numerical Features")
    plt.tight_layout()

    plt.show()


# -------------------- 8. Box Plots --------------------

for column in numeric_columns[:8]:

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title("Box Plot of " + column)
    plt.xlabel(column)

    plt.tight_layout()

    plt.show()


# -------------------- 9. Correlation Heatmap --------------------

if len(numeric_columns) >= 2:

    plt.figure(figsize=(12, 8))

    correlation = df[numeric_columns].corr()  # returns pandas datafraame

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


# -------------------- 10. Count Plots --------------------

for column in categorical_columns[:6]:

    if df[column].nunique(dropna=True) <= 15:

        plt.figure(figsize=(9, 5))

        sns.countplot(
            data=df,
            x=column,
            order=df[column].value_counts().index
        )

        plt.title("Count Plot of " + column)
        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            "member1_count_" + column.replace(" ", "_") + ".png"
        )

        plt.show()


# -------------------- 11. Scatter Plots --------------------

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


# -------------------- 12. Violin Plots --------------------

for column in numeric_columns[:4]:

    values = df[column].dropna()

    if len(values) > 0:

        plt.figure(figsize=(8, 4))

        sns.violinplot(x=values)

        plt.title("Violin Plot of " + column)

        plt.tight_layout()

        plt.savefig(
            "member1_violin_" + column.replace(" ", "_") + ".png"
        )

        plt.show()


# -------------------- 13. Pair Plot --------------------

if len(numeric_columns) >= 2:

    selected_columns = numeric_columns[:4]

    pair_data = df[selected_columns].dropna()

    if len(pair_data) >= 2:

        pair_data = pair_data.sample(
            n=min(500, len(pair_data)),
            random_state=42
        )

        sns.pairplot(pair_data)

        plt.savefig("member1_pairplot.png")
        plt.show()


print("\nMEMBER 1 COMPLETED SUCCESSFULLY!")


# ============================================================
#                    MEMBER 2
#             LOGISTIC REGRESSION
# ============================================================

print("\n")
print("=" * 60)
print("MEMBER 2: LOGISTIC REGRESSION")
print("=" * 60)


# -------------------- 1. Load Cleaned Dataset --------------------

df = pd.read_csv("spotify_cleaned.csv")


# -------------------- 2. Select Target Column --------------------

# If automatic detection fails, enter the correct column name.
# Example: CHURN_COLUMN = "churn"

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


# -------------------- 3. Prepare Target Variable --------------------

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


# -------------------- 4. Prepare Input Features --------------------

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


# -------------------- 5. Data Preprocessing --------------------

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


# -------------------- 6. Create Logistic Regression Model --------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# -------------------- 7. Split Dataset --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# -------------------- 8. Train Model --------------------

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")


# -------------------- 9. Make Predictions --------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# -------------------- 10. Evaluate Model --------------------

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


# -------------------- 11. Save Results --------------------

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


# -------------------- 12. Confusion Matrix --------------------

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


# -------------------- 13. ROC Curve --------------------

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


# -------------------- 14. Precision-Recall Curve --------------------

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


# -------------------- 15. Probability Distribution --------------------

plt.figure(figsize=(7, 5))

plt.hist(
    y_probability,
    bins=20,
    edgecolor="black"
)

plt.xlabel("Predicted Churn Probability")
plt.ylabel("Number of Users")

plt.title("Predicted Churn Probability Distribution")

plt.tight_layout()

plt.savefig("member2_probability_distribution.png")
plt.show()


print("\nMEMBER 2 COMPLETED SUCCESSFULLY!")


# ============================================================
#                    MEMBER 3
#              LINEAR REGRESSION
# ============================================================

print("\n")
print("=" * 60)
print("MEMBER 3: LINEAR REGRESSION")
print("=" * 60)


# -------------------- 1. Load Cleaned Dataset --------------------

df = pd.read_csv("spotify_cleaned.csv")


# -------------------- 2. Select Numerical Target --------------------

# Change this to a genuine continuous numerical column.
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
        "Numerical target not detected. "
        "Set LINEAR_TARGET manually."
    )


print("\nSelected Linear Regression target:", LINEAR_TARGET)


# -------------------- 3. Prepare Data --------------------

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


# -------------------- 4. Preprocessing --------------------

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


# -------------------- 5. Create Linear Regression Model --------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])


# -------------------- 6. Split Dataset --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# -------------------- 7. Train Model --------------------

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# -------------------- 8. Make Predictions --------------------

y_pred = model.predict(X_test)


# -------------------- 9. Evaluate Model --------------------

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


# -------------------- 10. Save Results --------------------

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


# -------------------- 11. Actual vs Predicted Graph --------------------

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

plt.savefig("member3_actual_vs_predicted.png")
plt.show()


# -------------------- 12. Residual Distribution --------------------

residuals = y_test.values - y_pred

plt.figure(figsize=(7, 5))

sns.histplot(
    residuals,
    kde=True
)

plt.xlabel("Residual")
plt.ylabel("Frequency")

plt.title("Distribution of Residuals")

plt.tight_layout()

plt.savefig("member3_residual_distribution.png")
plt.show()


# -------------------- 13. Residuals vs Predicted --------------------

plt.figure(figsize=(7, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.title("Residuals vs Predicted Values")

plt.tight_layout()

plt.savefig("member3_residuals_vs_predicted.png")
plt.show()


print("\nMEMBER 3 COMPLETED SUCCESSFULLY!")


# ============================================================
#                    MEMBER 4
#           TEAM LEADER - FINAL ANALYSIS
# ============================================================

print("\n")
print("=" * 60)
print("MEMBER 4: TEAM LEADER - FINAL ANALYSIS")
print("=" * 60)


# -------------------- 1. Load Model Results --------------------

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


# -------------------- 2. Organise Results --------------------

logistic_metrics = dict(zip(
    logistic_results["Metric"],
    logistic_results["Value"]
))

linear_metrics = dict(zip(
    linear_results["Metric"],
    linear_results["Value"]
))


# -------------------- 3. Print Complete Project Results --------------------

print("\nLOGISTIC REGRESSION PERFORMANCE")
print("-" * 40)

for metric, value in logistic_metrics.items():

    print(metric, ":", round(value, 4))


print("\nLINEAR REGRESSION PERFORMANCE")
print("-" * 40)

for metric, value in linear_metrics.items():

    print(metric, ":", round(value, 4))


# -------------------- 4. Classification Performance Graph --------------------

classification_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

classification_values = [
    logistic_metrics["Accuracy"],
    logistic_metrics["Precision"],
    logistic_metrics["Recall"],
    logistic_metrics["F1 Score"]
]

plt.figure(figsize=(9, 5))

bars = plt.bar(
    classification_names,
    classification_values
)

plt.ylim(0, 1)

plt.ylabel("Score")
plt.title("Logistic Regression Performance Comparison")

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        round(height, 3),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig("leader_logistic_performance.png")
plt.show()


# -------------------- 5. Linear Regression Error Comparison --------------------

error_names = [
    "MAE",
    "MSE",
    "RMSE"
]

error_values = [
    linear_metrics["MAE"],
    linear_metrics["MSE"],
    linear_metrics["RMSE"]
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    error_names,
    error_values
)

plt.ylabel("Error Value")
plt.title("Linear Regression Error Metrics")

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        round(height, 3),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig("leader_linear_errors.png")
plt.show()


# -------------------- 6. R2 Score Graph --------------------

plt.figure(figsize=(6, 5))

plt.bar(
    ["Linear Regression"],
    [linear_metrics["R2 Score"]]
)

plt.ylabel("R2 Score")
plt.title("Linear Regression Model Performance")

plt.tight_layout()

plt.savefig("leader_r2_score.png")
plt.show()


# -------------------- 7. Actual vs Predicted Churn --------------------

plt.figure(figsize=(9, 5))

plt.scatter(
    range(len(logistic_predictions)),
    logistic_predictions["Actual"],
    label="Actual",
    alpha=0.6
)

plt.scatter(
    range(len(logistic_predictions)),
    logistic_predictions["Predicted"],
    label="Predicted",
    alpha=0.6
)

plt.xlabel("Test Data Row")
plt.ylabel("Churn Class")

plt.title("Actual vs Predicted Churn")

plt.legend()
plt.tight_layout()

plt.savefig("leader_churn_predictions.png")
plt.show()


# -------------------- 8. Actual vs Predicted Numerical Values --------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    linear_predictions["Actual"],
    linear_predictions["Predicted"],
    alpha=0.6
)

minimum = min(
    linear_predictions["Actual"].min(),
    linear_predictions["Predicted"].min()
)

maximum = max(
    linear_predictions["Actual"].max(),
    linear_predictions["Predicted"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Final Linear Regression Predictions")

plt.tight_layout()

plt.savefig("leader_actual_vs_predicted.png")
plt.show()


# -------------------- 9. Churn Class Distribution --------------------

plt.figure(figsize=(7, 5))

churn_counts = (
    logistic_predictions["Actual"]
    .value_counts()
    .sort_index()
)

plt.bar(
    ["Not Churned", "Churned"],
    [
        churn_counts.get(0, 0),
        churn_counts.get(1, 0)
    ]
)

plt.ylabel("Number of Users")
plt.title("Actual Churn Distribution in Test Data")

plt.tight_layout()

plt.savefig("leader_churn_distribution.png")
plt.show()


# -------------------- 10. Prediction Probability Analysis --------------------

plt.figure(figsize=(8, 5))

plt.hist(
    logistic_predictions["Probability"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Churn Probability")
plt.ylabel("Number of Users")

plt.title("Distribution of Churn Prediction Probabilities")

plt.tight_layout()

plt.savefig("leader_probability_analysis.png")
plt.show()


# -------------------- 11. Prediction Error Analysis --------------------

linear_predictions["Absolute Error"] = abs(
    linear_predictions["Actual"] -
    linear_predictions["Predicted"]
)

plt.figure(figsize=(8, 5))

plt.hist(
    linear_predictions["Absolute Error"],
    bins=20,
    edgecolor="black"
)

plt.xlabel("Absolute Prediction Error")
plt.ylabel("Number of Observations")

plt.title("Linear Regression Prediction Error Distribution")

plt.tight_layout()

plt.savefig("leader_prediction_error.png")
plt.show()


# -------------------- 12. Calculate Additional Statistics --------------------

average_actual = linear_predictions["Actual"].mean()

average_predicted = linear_predictions["Predicted"].mean()

average_absolute_error = (
    linear_predictions["Absolute Error"].mean()
)

print("\nADDITIONAL REGRESSION ANALYSIS")
print("-" * 40)

print("Average Actual Value:", average_actual)

print("Average Predicted Value:", average_predicted)

print("Average Absolute Error:", average_absolute_error)


# -------------------- 13. Create Combined Results Table --------------------

combined_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Logistic Regression",
        "Logistic Regression",
        "Logistic Regression",
        "Logistic Regression",
        "Linear Regression",
        "Linear Regression",
        "Linear Regression",
        "Linear Regression"
    ],

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ],

    "Value": [
        logistic_metrics["Accuracy"],
        logistic_metrics["Precision"],
        logistic_metrics["Recall"],
        logistic_metrics["F1 Score"],
        logistic_metrics["ROC AUC"],
        linear_metrics["MAE"],
        linear_metrics["MSE"],
        linear_metrics["RMSE"],
        linear_metrics["R2 Score"]
    ]

})

print("\nFINAL COMBINED RESULTS")
print(combined_results)

combined_results.to_csv(
    "final_project_results.csv",
    index=False
)


# -------------------- 14. Save Final Project Summary --------------------

with open(
    "final_project_summary.txt",
    "w"
) as file:

    file.write("SPOTIFY CHURN ANALYSIS - AI101 PROJECT\n")

    file.write("=" * 45 + "\n\n")

    file.write("MODEL 1: LOGISTIC REGRESSION\n")

    for metric, value in logistic_metrics.items():

        file.write(
            str(metric) + ": " + str(value) + "\n"
        )

    file.write("\nMODEL 2: LINEAR REGRESSION\n")

    for metric, value in linear_metrics.items():

        file.write(
            str(metric) + ": " + str(value) + "\n"
        )

    file.write("\nADDITIONAL ANALYSIS\n")

    file.write(
        "Average Actual Value: " +
        str(average_actual) + "\n"
    )

    file.write(
        "Average Predicted Value: " +
        str(average_predicted) + "\n"
    )

    file.write(
        "Average Absolute Error: " +
        str(average_absolute_error) + "\n"
    )

    file.write(
        "\nLogistic Regression was used for binary churn classification.\n"
    )

    file.write(
        "Linear Regression was used for continuous numerical prediction.\n"
    )


# -------------------- 15. Final Project Conclusion --------------------

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
