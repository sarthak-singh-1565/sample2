
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read results from Member 2
logistic_results = pd.read_csv("logistic_results.csv")

# Read results from Member 3
linear_results = pd.read_csv("linear_results.csv")

# Convert results into simple dictionaries
logistic_metrics = dict(zip(
    logistic_results["Metric"],
    logistic_results["Value"]
))

linear_metrics = dict(zip(
    linear_results["Metric"],
    linear_results["Value"]
))

print("\n================================")
print("       FINAL PROJECT REPORT")
print("================================")

print("\nLOGISTIC REGRESSION")
for metric, value in logistic_metrics.items():
    print(metric, ":", value)

print("\nLINEAR REGRESSION")
for metric, value in linear_metrics.items():
    print(metric, ":", value)

# Graph 1: Classification model performance
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

plt.figure(figsize=(8, 5))
plt.bar(classification_names, classification_values)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Logistic Regression Performance")
plt.tight_layout()
plt.savefig("final_logistic_performance.png")
plt.show()

# Graph 2: Linear Regression error metrics
error_names = ["MAE", "MSE", "RMSE"]

error_values = [
    linear_metrics["MAE"],
    linear_metrics["MSE"],
    linear_metrics["RMSE"]
]

plt.figure(figsize=(8, 5))
plt.bar(error_names, error_values)
plt.ylabel("Error")
plt.title("Linear Regression Error Metrics")
plt.tight_layout()
plt.savefig("final_linear_errors.png")
plt.show()

# Graph 3: Linear Regression R2 score
plt.figure(figsize=(6, 5))
plt.bar(
    ["R2 Score"],
    [linear_metrics["R2 Score"]]
)
plt.ylabel("R2 Score")
plt.title("Linear Regression R2 Performance")
plt.tight_layout()
plt.savefig("final_linear_r2.png")
plt.show()

# Read prediction files
logistic_predictions = pd.read_csv("logistic_predictions.csv")
linear_predictions = pd.read_csv("linear_predictions.csv")

# Graph 4: Actual vs predicted churn
plt.figure(figsize=(7, 5))

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
plt.savefig("final_churn_predictions.png")
plt.show()

# Graph 5: Actual vs predicted numerical values
plt.figure(figsize=(7, 5))

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
plt.savefig("final_linear_predictions.png")
plt.show()

# Save combined model results in one CSV
combined_results = pd.DataFrame({
    "Model": [
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
        linear_metrics["MAE"],
        linear_metrics["MSE"],
        linear_metrics["RMSE"],
        linear_metrics["R2 Score"]
    ]
})

combined_results.to_csv("final_project_results.csv", index=False)

print("\nFinal results saved as final_project_results.csv")

# Save a simple text summary
with open("final_project_summary.txt", "w") as file:
    file.write("SPOTIFY CHURN ANALYSIS - AI101 PROJECT\n\n")

    file.write("LOGISTIC REGRESSION RESULTS\n")
    for metric, value in logistic_metrics.items():
        file.write(str(metric) + ": " + str(value) + "\n")

    file.write("\nLINEAR REGRESSION RESULTS\n")
    for metric, value in linear_metrics.items():
        file.write(str(metric) + ": " + str(value) + "\n")

    file.write(
        "\nThe Logistic Regression model was used for "
        "binary churn classification.\n"
    )

    file.write(
        "The Linear Regression model was used to predict "
        "a continuous numerical variable.\n"
    )

print("\nTeam leader work completed.")
print("All project results have been combined.")
