# ============================================================
#              SPOTIFY CHURN ANALYSIS - AI101
# ============================================================
#
# MEMBER 1: Data Loading, Cleaning and Basic Analysis
# MEMBER 2: Logistic Regression - Churn Classification
# MEMBER 3: Linear Regression - Numerical Prediction
# MEMBER 4: Team Leader - Final Results
#
# This project uses Matplotlib only for graphs.
# Seaborn is NOT used.
# ============================================================


# -------------------- IMPORTING LIBRARIES --------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------- SETTINGS --------------------

# Change these names if your CSV file uses different column names.

CHURN_COLUMN = "churn"
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

# Fill missing numerical values with the column mean
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())

# Fill missing text values with the most common value
text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nDataset after cleaning:")
print(df.shape)

# Save the cleaned dataset
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


# GRAPH 2: Histogram of the numerical target

if LINEAR_TARGET in df.columns:

    plt.figure(figsize=(7, 5))
    plt.hist(df[LINEAR_TARGET].dropna(), bins=15, edgecolor="black")

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

    plt.figure(figsize=(7, 5))
    plt.boxplot(df[LINEAR_TARGET].dropna())

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

    # Show correlation values inside the boxes
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

    # Remove rows where the churn value is missing
    logistic_data = df.dropna(subset=[CHURN_COLUMN]).copy()

    # Convert churn values into 0 and 1
    churn_text = logistic_data[CHURN_COLUMN].astype(str).str.strip().str.lower()

    positive_values = ["1", "yes", "true", "churn", "churned"]
    negative_values = ["0", "no", "false", "not churn", "not churned", "retained"]

    if set(churn_text.unique()).issubset(set(positive_values + negative_values)):

        y = churn_text.isin(positive_values).astype(int)

    elif len(churn_text.unique()) == 2:

        categories = sorted(churn_text.unique())

        print("\nChurn categories:", categories)
        print("The category", categories[1], "will be treated as churn (1).")

        y = (churn_text == categories[1]).astype(int)

    else:
        y = None
        print("\nChurn column must contain exactly two categories.")

    if y is not None and y.nunique() == 2:

        # Select all columns except the churn column
        X = logistic_data.drop(columns=[CHURN_COLUMN])

        # Remove ID columns because they do not help prediction
        id_columns = []

        for column in X.columns:
            if column.lower() in ["id", "user_id", "customer_id", "index"]:
                id_columns.append(column)

        X = X.drop(columns=id_columns)

        # Convert text columns into numbers
        X = pd.get_dummies(X, drop_first=True)

        # Fill any remaining missing values
        X = X.fillna(X.mean())

        if len(X.columns) > 0 and y.value_counts().min() >= 2:

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y
            )

            # Scale the input data
            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Create and train the model
            logistic_model = LogisticRegression(max_iter=1000)
            logistic_model.fit(X_train, y_train)

            # Predict churn
            y_pred = logistic_model.predict(X_test)

            # Calculate accuracy
            logistic_accuracy = accuracy_score(y_test, y_pred)

            print("\nLogistic Regression Accuracy:")
            print(round(logistic_accuracy * 100, 2), "%")

            # Calculate precision, recall and F1 score manually
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

            tn = cm[0, 0]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tp = cm[1, 1]

            if tp + fp > 0:
                logistic_precision = tp / (tp + fp)
            else:
                logistic_precision = 0

            if tp + fn > 0:
                logistic_recall = tp / (tp + fn)
            else:
                logistic_recall = 0

            if logistic_precision + logistic_recall > 0:
                logistic_f1 = (
                    2 * logistic_precision * logistic_recall
                    / (logistic_precision + logistic_recall)
                )
            else:
                logistic_f1 = 0

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

    # Keep only rows with a numerical target
    linear_data = df.copy()
    linear_data[LINEAR_TARGET] = pd.to_numeric(
        linear_data[LINEAR_TARGET],
        errors="coerce"
    )
    linear_data = linear_data.dropna(subset=[LINEAR_TARGET])

    # Use numerical columns as input features
    X = linear_data.select_dtypes(include=np.number).drop(
        columns=[LINEAR_TARGET]
    )

    y = linear_data[LINEAR_TARGET]

    # Remove ID columns
    id_columns = []

    for column in X.columns:
        if column.lower() in ["id", "user_id", "customer_id", "index"]:
            id_columns.append(column)

    X = X.drop(columns=id_columns)

    if len(X.columns) > 0 and len(linear_data) >= 5:

        # Fill missing input values with their mean
        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # Create and train Linear Regression model
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        # Predict numerical values
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
        print("Not enough numerical data for Linear Regression.")

else:
    print("Linear target column not found. Check LINEAR_TARGET.")
    print("Linear Regression was skipped.")

print("\nMEMBER 3 COMPLETED!")


# ============================================================
# MEMBER 4: TEAM LEADER - FINAL ANALYSIS
# ============================================================

print("\n========== MEMBER 4: FINAL PROJECT RESULTS ==========")

# Display the final model results

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
