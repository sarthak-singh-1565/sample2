
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("spotify_churn_dataset.csv")

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe(include="all"))

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Remove completely empty rows
df = df.dropna(how="all")

# Save cleaned dataset
df.to_csv("spotify_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")

# Set graph style
sns.set_theme(style="whitegrid")

# 1. Missing values graph
missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    plt.figure(figsize=(10, 5))
    missing.sort_values().plot(kind="bar")
    plt.title("Missing Values in Each Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("graph_01_missing_values.png")
    plt.show()

# 2. Histograms of numerical columns
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_columns) > 0:
    df[numeric_columns].hist(figsize=(14, 10), bins=20)
    plt.suptitle("Distribution of Numerical Features")
    plt.tight_layout()
    plt.savefig("graph_02_histograms.png")
    plt.show()

# 3. Box plots of numerical columns
if len(numeric_columns) > 0:
    for column in numeric_columns[:8]:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[column])
        plt.title("Box Plot of " + column)
        plt.tight_layout()
        plt.savefig("box_" + column.replace(" ", "_") + ".png")
        plt.show()

# 4. Correlation heatmap
if len(numeric_columns) >= 2:
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df[numeric_columns].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("graph_03_correlation.png")
    plt.show()

# 5. Count plots for categorical columns
categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

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
        plt.savefig("count_" + column.replace(" ", "_") + ".png")
        plt.show()

# 6. Scatter plots between numerical columns
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
                "scatter_" + x_column.replace(" ", "_") + ".png"
            )
            plt.show()

# 7. Violin plots for numerical features
for column in numeric_columns[:4]:
    plt.figure(figsize=(8, 4))
    sns.violinplot(x=df[column].dropna())
    plt.title("Violin Plot of " + column)
    plt.tight_layout()
    plt.savefig("violin_" + column.replace(" ", "_") + ".png")
    plt.show()

# 8. Pair plot using a small sample
if len(numeric_columns) >= 2:
    selected_columns = numeric_columns[:4]
    pair_data = df[selected_columns].dropna()

    if len(pair_data) >= 2:
        pair_data = pair_data.sample(
            n=min(500, len(pair_data)),
            random_state=42
        )

        sns.pairplot(pair_data)
        plt.savefig("graph_04_pairplot.png")
        plt.show()

print("\nMember 1 work completed.")
