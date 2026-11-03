
from pathlib import Path
import pandas as pd

# Find the main project folder
project_folder = Path(__file__).resolve().parents[1]

# Locate the dataset
dataset_path = project_folder / "data" / "cs-training.csv"

# Load the CSV file
df = pd.read_csv(dataset_path)

# Display the first 5 rows
print("First 5 rows of the real dataset:")
print(df.head())

# Display dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Check the data types and missing values
print("\nDataset information:")
df.info()

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["SeriousDlqin2yrs"].value_counts())


# Check duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Statistical summary of numerical columns
print("\nStatistical summary:")
print(df.describe())

# Check target distribution in percentages
print("\nTarget distribution in percentage:")
print(
    df["SeriousDlqin2yrs"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# Create a cleaned working dataset
df_clean = df.copy()

# Remove the unnecessary index column
df_clean = df_clean.drop(columns=["Unnamed: 0"])

# Check missing values in each column
print("\nMissing values before cleaning:")
print(df_clean.isnull().sum())

# Fill missing dependent counts with the median
median_dependents = df_clean["NumberOfDependents"].median()

df_clean["NumberOfDependents"] = (
    df_clean["NumberOfDependents"].fillna(median_dependents)
)

# Verify missing values after cleaning
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

print("\nCleaned dataset shape:")
print(df_clean.shape)



# Find the median monthly income
median_income = df_clean["MonthlyIncome"].median()

# Fill missing monthly income values with the median
df_clean["MonthlyIncome"] = (
    df_clean["MonthlyIncome"].fillna(median_income)
)

# Verify the cleaning results
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

print("\nTotal missing values:")
print(df_clean.isnull().sum().sum())

print("\nFinal cleaned dataset shape:")
print(df_clean.shape)



import matplotlib.pyplot as plt
import seaborn as sns

# Chart 1: Distribution of serious delinquency
plt.figure(figsize=(6, 4))
sns.countplot(data=df_clean, x="SeriousDlqin2yrs")
plt.title("Serious Delinquency Distribution")
plt.xlabel("Serious Delinquency (0 = No, 1 = Yes)")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.show()

# Chart 2: Average monthly income by target
income_by_default = df_clean.groupby(
    "SeriousDlqin2yrs"
)["MonthlyIncome"].mean()

print("\nAverage monthly income by target:")
print(income_by_default)

income_by_default.plot(kind="bar", figsize=(6, 4))
plt.title("Average Monthly Income by Delinquency Status")
plt.xlabel("Serious Delinquency (0 = No, 1 = Yes)")
plt.ylabel("Average Monthly Income")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Chart 3: Age distribution by target
plt.figure(figsize=(8, 4))
sns.histplot(
    data=df_clean,
    x="age",
    hue="SeriousDlqin2yrs",
    bins=30,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Age Distribution by Delinquency Status")
plt.xlabel("Age")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
