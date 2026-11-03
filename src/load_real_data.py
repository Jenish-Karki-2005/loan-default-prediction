
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Find the main project folder
project_folder = Path(__file__).resolve().parents[1]

# Load the real dataset
dataset_path = project_folder / "data" / "cs-training.csv"
df = pd.read_csv(dataset_path)

# Explore the dataset
print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["SeriousDlqin2yrs"].value_counts())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

print("\nTarget distribution (%):")
print(df["SeriousDlqin2yrs"].value_counts(normalize=True).mul(100).round(2))

# Create a cleaned copy for exploration
df_clean = df.drop(columns=["Unnamed: 0"], errors="ignore").copy()

df_clean["NumberOfDependents"] = df_clean["NumberOfDependents"].fillna(
    df_clean["NumberOfDependents"].median()
)

df_clean["MonthlyIncome"] = df_clean["MonthlyIncome"].fillna(
    df_clean["MonthlyIncome"].median()
)

print("\nCleaned dataset shape:")
print(df_clean.shape)

print("\nMissing values after exploratory cleaning:")
print(df_clean.isnull().sum())

# Chart 1: Delinquency distribution
sns.countplot(data=df, x="SeriousDlqin2yrs")
plt.title("Serious Delinquency Distribution")
plt.xlabel("Serious delinquency within 2 years (0 = No, 1 = Yes)")
plt.ylabel("Number of applicants")
plt.tight_layout()
plt.show()

# Chart 2: Average monthly income by target
sns.barplot(data=df_clean, x="SeriousDlqin2yrs", y="MonthlyIncome", errorbar=None)
plt.title("Average Monthly Income by Delinquency Status")
plt.xlabel("Serious delinquency (0 = No, 1 = Yes)")
plt.ylabel("Average monthly income")
plt.tight_layout()
plt.show()

# Chart 3: Age distribution by target
sns.histplot(
    data=df_clean,
    x="age",
    hue="SeriousDlqin2yrs",
    stat="density",
    common_norm=False,
    element="step"
)
plt.title("Age Distribution by Delinquency Status")
plt.xlabel("Age")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
