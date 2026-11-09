# ============================================================
#              SPOTIFY CHURN ANALYSIS - AI101
# ============================================================
#
# MEMBER 1: Data Loading, Cleaning and Basic Analysis
# MEMBER 2: Logistic Regression - Churn Classification
# MEMBER 3: Linear Regression - Numerical Prediction
# MEMBER 4: Team Leader - Final Results
#
# Graphs are made using Matplotlib only.
# Preprocessing uses SimpleImputer and ColumnTransformer.
# ============================================================


# -------------------- IMPORTING LIBRARIES --------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------- SETTINGS --------------------

# Change these names if your CSV file uses different column names.

CHURN_COLUMN = "is_churned"
LINEAR_TARGET = "listening_time"

# Read the dataset
df = pd.read_csv("spotify_churn_dataset.csv")


# ============================================================
# MEMBER 1: DATA LOADING, CLEANING AND BASIC ANALYSIS
# ============================================================

print("\n========== MEMBER 1: DATA ANALYSIS ==========")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove completely empty rows
df = df.dropna(how="all")

print("\nDataset after removing duplicates:")
print(df.shape)

# Save the dataset. Missing values will be handled during preprocessing.
df.to_csv("spotify_cleaned.csv", index=False)


# GRAPH 1: Churn count bar graph

if CHURN_COLUMN in df.columns:

    churn_counts = df[CHURN_COLUMN].value_counts()

    plt.figure(figsize=(7, 5))
    plt.bar(churn_counts.index.astype(str), churn_counts.values)

    plt.title("Number of Churned and Non-Churned Users")
    plt.xlabel("Churn Status")
    plt.ylabel("Number of Users")

    plt.tight_layout()
    plt.savefig("graph1_churn_count.png")
    plt.show()

else:
    print("\nChurn column not found. Graph 1 was skipped.")


# GRAPH 2: Histogram

if LINEAR_TARGET in df.columns:

    values = pd.to_numeric(df[LINEAR_TARGET], errors="coerce").dropna()

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=15, edgecolor="black")

    plt.title("Distribution of " + LINEAR_TARGET)
    plt.xlabel(LINEAR_TARGET)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("graph2_histogram.png")
    plt.show()

else:
    print("\nLinear target column not found. Graph 2 was skipped.")


# GRAPH 3: Box plot

if LINEAR_TARGET in df.columns:

    values = pd.to_numeric(df[LINEAR_TARGET], errors="coerce").dropna()

    plt.figure(figsize=(7, 5))
    plt.boxplot(values)

    plt.title("Box Plot of " + LINEAR_TARGET)
    plt.ylabel(LINEAR_TARGET)

    plt.tight_layout()
    plt.savefig("graph3_boxplot.png")
    plt.show()


# GRAPH 4: Correlation heatmap using Matplotlib

numeric_data = df.select_dtypes(include=np.number)

if len(numeric_data.columns) >= 2:

    correlation = numeric_data.corr()

    plt.figure(figsize=(9, 6))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Correlation Heatmap")

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            plt.text(
                j, i,
                round(correlation.iloc[i, j], 2),
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig("graph4_correlation_heatmap.png")
    plt.show()

else:
    print("\nNot enough numerical columns for correlation heatmap.")

print("\nMEMBER 1 COMPLETED!")


# ============================================================
# MEMBER 2: LOGISTIC REGRESSION - CHURN CLASSIFICATION
# ============================================================

print("\n========== MEMBER 2: LOGISTIC REGRESSION ==========")

logistic_accuracy = None
logistic_precision = None
logistic_recall = None
logistic_f1 = None

if CHURN_COLUMN in df.columns:

    logistic_data = df.dropna(subset=[CHURN_COLUMN]).copy()

    # Convert churn values into 0 and 1
    churn_text = (
        logistic_data[CHURN_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive_values = ["1", "yes", "true", "churn", "churned"]
    negative_values = ["0", "no", "false", "not churn",
                       "not churned", "retained"]

    unique_values = churn_text.unique().tolist()

    if set(unique_values).issubset(set(positive_values + negative_values)):

        y = churn_text.isin(positive_values).astype(int)

    elif len(unique_values) == 2:

        categories = sorted(unique_values)

        print("\nChurn categories:", categories)
        print("The category", categories[1], "will be treated as churn (1).")

        y = (churn_text == categories[1]).astype(int)

    else:
        y = None
        print("\nChurn column must contain exactly two categories.")

    if y is not None and y.nunique() == 2:

        # Input columns are all columns except the churn column
        X = logistic_data.drop(columns=[CHURN_COLUMN])

        # Remove ID columns
        id_columns = []

        for column in X.columns:
            if column.lower() in ["id", "user_id", "customer_id", "index"]:
                id_columns.append(column)

        X = X.drop(columns=id_columns)

        # Find numerical and categorical columns
        numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Numerical preprocessing:
        # Fill missing values with median and scale the values
        numeric_preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical preprocessing:
        # Fill missing values with most common value and convert text to numbers
        categorical_preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Apply preprocessing to the correct columns
        preprocessor = ColumnTransformer([
            ("numerical", numeric_preprocessing, numeric_columns),
            ("categorical", categorical_preprocessing, categorical_columns)
        ])

        # Make sure the data can be divided into training and testing sets
        if len(X.columns) > 0 and y.value_counts().min() >= 2:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            # First preprocess the data, then train Logistic Regression
            logistic_model = Pipeline([
                ("preprocessing", preprocessor),
                ("model", LogisticRegression(max_iter=1000))
            ])

            logistic_model.fit(X_train, y_train)

            y_pred = logistic_model.predict(X_test)

            logistic_accuracy = accuracy_score(y_test, y_pred)
            logistic_precision = precision_score(
                y_test, y_pred, zero_division=0
            )
            logistic_recall = recall_score(
                y_test, y_pred, zero_division=0
            )
            logistic_f1 = f1_score(
                y_test, y_pred, zero_division=0
            )

            print("\nLogistic Regression Results:")
            print("Accuracy:", round(logistic_accuracy * 100, 2), "%")
            print("Precision:", round(logistic_precision, 3))
            print("Recall:", round(logistic_recall, 3))
            print("F1 Score:", round(logistic_f1, 3))

            # Save prediction results
            logistic_predictions = pd.DataFrame({
                "Actual": y_test.values,
                "Predicted": y_pred
            })

            logistic_predictions.to_csv(
                "logistic_predictions.csv",
                index=False
            )

            # GRAPH 5: Confusion matrix using Matplotlib

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()

            plt.title("Logistic Regression - Confusion Matrix")
            plt.xlabel("Predicted Value")
            plt.ylabel("Actual Value")

            plt.xticks([0, 1], ["Not Churned", "Churned"])
            plt.yticks([0, 1], ["Not Churned", "Churned"])

            for i in range(2):
                for j in range(2):
                    plt.text(
                        j, i, cm[i, j],
                        ha="center",
                        va="center"
                    )

            plt.tight_layout()
            plt.savefig("graph5_confusion_matrix.png")
            plt.show()

        else:
            print("Not enough data to train Logistic Regression.")

else:
    print("Churn column not found. Check CHURN_COLUMN.")

print("\nMEMBER 2 COMPLETED!")


# ============================================================
# MEMBER 3: LINEAR REGRESSION - NUMERICAL PREDICTION
# ============================================================

print("\n========== MEMBER 3: LINEAR REGRESSION ==========")

linear_r2 = None
linear_mae = None
linear_rmse = None

if LINEAR_TARGET in df.columns:

    linear_data = df.copy()

    # Make sure the target column contains numbers
    linear_data[LINEAR_TARGET] = pd.to_numeric(
        linear_data[LINEAR_TARGET],
        errors="coerce"
    )

    linear_data = linear_data.dropna(subset=[LINEAR_TARGET])

    # Do not use the churn target as an input feature
    columns_to_remove = [LINEAR_TARGET]

    if CHURN_COLUMN in linear_data.columns:
        columns_to_remove.append(CHURN_COLUMN)

    X = linear_data.drop(columns=columns_to_remove)
    y = linear_data[LINEAR_TARGET]

    # Remove ID columns
    id_columns = []

    for column in X.columns:
        if column.lower() in ["id", "user_id", "customer_id", "index"]:
            id_columns.append(column)

    X = X.drop(columns=id_columns)

    # Find numerical and categorical columns
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if len(X.columns) > 0 and len(linear_data) >= 5:

        # Numerical preprocessing
        numeric_preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical preprocessing
        categorical_preprocessing = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine both preprocessing steps
        preprocessor = ColumnTransformer([
            ("numerical", numeric_preprocessing, numeric_columns),
            ("categorical", categorical_preprocessing, categorical_columns)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # Preprocess the data and apply Linear Regression
        linear_model = Pipeline([
            ("preprocessing", preprocessor),
            ("model", LinearRegression())
        ])

        linear_model.fit(X_train, y_train)

        y_pred = linear_model.predict(X_test)

        # Calculate model performance
        linear_mae = mean_absolute_error(y_test, y_pred)
        linear_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        linear_r2 = r2_score(y_test, y_pred)

        print("\nLinear Regression Results:")
        print("Mean Absolute Error:", round(linear_mae, 3))
        print("Root Mean Squared Error:", round(linear_rmse, 3))
        print("R2 Score:", round(linear_r2, 3))

        # Save prediction results
        linear_predictions = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        linear_predictions.to_csv(
            "linear_predictions.csv",
            index=False
        )

        # GRAPH 6: Actual vs Predicted values

        plt.figure(figsize=(7, 5))
        plt.scatter(y_test, y_pred)

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")

        plt.tight_layout()
        plt.savefig("graph6_actual_vs_predicted.png")
        plt.show()

    else:
        print("Not enough data for Linear Regression.")

else:
    print("Linear target column not found. Check LINEAR_TARGET.")
    print("Linear Regression was skipped.")

print("\nMEMBER 3 COMPLETED!")


# ============================================================
# MEMBER 4: TEAM LEADER - FINAL ANALYSIS
# ============================================================

print("\n========== MEMBER 4: FINAL PROJECT RESULTS ==========")

print("\nLOGISTIC REGRESSION")

if logistic_accuracy is not None:
    print("Accuracy:", round(logistic_accuracy * 100, 2), "%")
    print("Precision:", round(logistic_precision, 3))
    print("Recall:", round(logistic_recall, 3))
    print("F1 Score:", round(logistic_f1, 3))
else:
    print("Logistic Regression was not completed.")

print("\nLINEAR REGRESSION")

if linear_r2 is not None:
    print("MAE:", round(linear_mae, 3))
    print("RMSE:", round(linear_rmse, 3))
    print("R2 Score:", round(linear_r2, 3))
else:
    print("Linear Regression was not completed.")

# Save all model results in one CSV file
final_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Logistic Regression",
        "Logistic Regression",
        "Logistic Regression",
        "Linear Regression",
        "Linear Regression",
        "Linear Regression"
    ],
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "MAE",
        "RMSE",
        "R2 Score"
    ],
    "Value": [
        logistic_accuracy,
        logistic_precision,
        logistic_recall,
        logistic_f1,
        linear_mae,
        linear_rmse,
        linear_r2
    ]
})

final_results.to_csv("final_project_results.csv", index=False)

print("\nFinal results saved in final_project_results.csv")

# Save a simple project summary
with open("final_project_summary.txt", "w") as file:

    file.write("SPOTIFY CHURN ANALYSIS - AI101 PROJECT\n\n")

    file.write("LOGISTIC REGRESSION RESULTS\n")
    file.write("Accuracy: " + str(logistic_accuracy) + "\n")
    file.write("Precision: " + str(logistic_precision) + "\n")
    file.write("Recall: " + str(logistic_recall) + "\n")
    file.write("F1 Score: " + str(logistic_f1) + "\n\n")

    file.write("LINEAR REGRESSION RESULTS\n")
    file.write("MAE: " + str(linear_mae) + "\n")
    file.write("RMSE: " + str(linear_rmse) + "\n")
    file.write("R2 Score: " + str(linear_r2) + "\n")

print("\nProject summary saved successfully.")
print("\n========== COMPLETE PROJECT FINISHED ==========")
