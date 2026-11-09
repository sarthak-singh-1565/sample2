
# ==========================================================
# SPOTIFY CHURN ANALYSIS
# AI101 GROUP PROJECT
#
# Member 1: Data Analysis
# Member 2: Logistic Regression
# Member 3: Linear Regression
# Member 4: Final Analysis and Comparison
# ==========================================================


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ==========================================================
# COMMON SETTINGS
# ==========================================================

df = pd.read_csv("spotify_churn_dataset.csv")

print("\nDATASET LOADED SUCCESSFULLY")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# Change these names according to your actual dataset.

CHURN_COLUMN = "churn"
LINEAR_TARGET = "listening_time"


# If your dataset uses different column names,
# replace the above two names with the correct names.


# ==========================================================
# MEMBER 1: DATA ANALYSIS
# ==========================================================

print("\n========== MEMBER 1: DATA ANALYSIS ==========")

# Checking missing values

print("\nMissing values:")
print(df.isnull().sum())

# Removing duplicate rows

df = df.drop_duplicates()

print("\nDataset after removing duplicates:")
print(df.shape)

# Basic statistical information

print("\nStatistical information:")
print(df.describe())

# Filling missing numerical values with their mean

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].mean())


# Filling missing text values with the most common value

text_columns = df.select_dtypes(include="object").columns

for column in text_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# Saving cleaned dataset

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


# GRAPH 2: Histogram

if LINEAR_TARGET in df.columns:

    plt.figure(figsize=(7, 5))

    plt.hist(df[LINEAR_TARGET].dropna(), bins=10)

    plt.title("Distribution of Listening Time")
    plt.xlabel(LINEAR_TARGET)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("graph2_histogram.png")
    plt.show()


# GRAPH 3: Box plot

if LINEAR_TARGET in df.columns:

    plt.figure(figsize=(7, 5))

    plt.boxplot(df[LINEAR_TARGET].dropna())

    plt.title("Box Plot of Listening Time")
    plt.ylabel(LINEAR_TARGET)

    plt.tight_layout()
    plt.savefig("graph3_boxplot.png")
    plt.show()


# GRAPH 4: Correlation heatmap

numeric_data = df.select_dtypes(include=np.number)

if len(numeric_data.columns) >= 2:

    correlation = numeric_data.corr()

    plt.figure(figsize=(9, 6))

    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar()

    plt.xticks(range(len(correlation.columns)),
               correlation.columns, rotation=90)

    plt.yticks(range(len(correlation.columns)),
               correlation.columns)

    plt.title("Correlation Heatmap")

    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            plt.text(j, i, round(correlation.iloc[i, j], 2),
                     ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("graph4_correlation.png")
    plt.show()


# ==========================================================
# MEMBER 2: LOGISTIC REGRESSION
# ==========================================================

print("\n========== MEMBER 2: LOGISTIC REGRESSION ==========")

logistic_accuracy = None
confusion = None
logistic_predictions = None


if CHURN_COLUMN in df.columns:

    # Selecting input columns

    model_data = df.select_dtypes(include=np.number).copy()

    if CHURN_COLUMN in model_data.columns:

        model_data = model_data.dropna()

        X = model_data.drop(columns=[CHURN_COLUMN])
        y = model_data[CHURN_COLUMN]

        # Logistic Regression requires two classes

        if len(y.unique()) == 2 and len(X.columns) > 0:

            # Dividing data into training and testing data

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.20,
                random_state=42
            )

            # Scaling the data

            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Creating model

            model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)

            # Making predictions

            y_pred = model.predict(X_test)

            # Checking accuracy

            logistic_accuracy = accuracy_score(y_test, y_pred)

            print("\nLogistic Regression Accuracy:")
            print(logistic_accuracy * 100, "%")

            confusion = confusion_matrix(y_test, y_pred)

            logistic_predictions = y_pred


            # GRAPH 5: Confusion matrix

            plt.figure(figsize=(6, 5))

            plt.imshow(confusion, cmap="Blues")

            plt.colorbar()

            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Values")
            plt.ylabel("Actual Values")

            for i in range(len(confusion)):
                for j in range(len(confusion)):
                    plt.text(j, i, confusion[i, j],
                             ha="center", va="center")

            plt.tight_layout()
            plt.savefig("graph5_confusion_matrix.png")
            plt.show()

        else:
            print("Logistic Regression cannot run.")
            print("Check whether churn has exactly two classes.")

    else:
        print("Churn column must be numerical (0 and 1).")

else:
    print("Churn column not found. Check CHURN_COLUMN.")


# ==========================================================
# MEMBER 3: LINEAR REGRESSION
# ==========================================================

print("\n========== MEMBER 3: LINEAR REGRESSION ==========")

linear_r2 = None
linear_mse = None
actual_values = None
predicted_values = None


if LINEAR_TARGET in df.columns:

    # Selecting numerical columns

    linear_data = df.select_dtypes(include=np.number).copy()

    linear_data = linear_data.dropna()

    if LINEAR_TARGET in linear_data.columns:

        X = linear_data.drop(columns=[LINEAR_TARGET])
        y = linear_data[LINEAR_TARGET]

        if len(X.columns) > 0 and len(linear_data) >= 5:

            # Dividing data

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.20,
                random_state=42
            )

            # Creating Linear Regression model

            model = LinearRegression()

            model.fit(X_train, y_train)

            # Predictions

            y_pred = model.predict(X_test)

            # Model performance

            linear_r2 = r2_score(y_test, y_pred)

            linear_mse = mean_squared_error(y_test, y_pred)

            actual_values = y_test
            predicted_values = y_pred

            print("\nLinear Regression R2 Score:")
            print(linear_r2)

            print("\nMean Squared Error:")
            print(linear_mse)


            # GRAPH 6: Actual vs Predicted

            plt.figure(figsize=(7, 5))

            plt.scatter(y_test, y_pred)

            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")

            plt.title("Actual vs Predicted Values")

            plt.tight_layout()
            plt.savefig("graph6_actual_predicted.png")
            plt.show()

        else:
            print("Not enough data for Linear Regression.")

else:
    print("Linear target column not found.")
    print("Check LINEAR_TARGET.")


# ==========================================================
# MEMBER 4: TEAM LEADER
# FINAL ANALYSIS AND COMPARISON
# ==========================================================

print("\n========== MEMBER 4: FINAL ANALYSIS ==========")

# Comparing model results

model_names = []
model_scores = []

if logistic_accuracy is not None:

    model_names.append("Logistic Regression")
    model_scores.append(logistic_accuracy * 100)

if linear_r2 is not None:

    model_names.append("Linear Regression R2")
    model_scores.append(linear_r2)


# GRAPH 7: Model comparison

if len(model_names) > 0:

    plt.figure(figsize=(8, 5))

    plt.bar(model_names, model_scores)

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig("graph7_model_comparison.png")
    plt.show()


# Final results

print("\n========== FINAL PROJECT RESULTS ==========")

if logistic_accuracy is not None:
    print("Logistic Regression Accuracy:",
          round(logistic_accuracy * 100, 2), "%")
else:
    print("Logistic Regression: Not completed")

if linear_r2 is not None:
    print("Linear Regression R2 Score:",
          round(linear_r2, 3))
else:
    print("Linear Regression: Not completed")


# Saving final results

results = pd.DataFrame({
    "Model": model_names,
    "Score": model_scores
})

results.to_csv("final_project_results.csv", index=False)

print("\nFinal results saved successfully.")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
