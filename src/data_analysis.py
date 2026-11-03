
import pandas as pd
import numpy as np

print("Loan Default Prediction & Credit Risk Analysis System")
print("-" * 55)

# Create a sample loan dataset
data = {
    "applicant_id": [101, 102, 103, 104, 105],
    "income": [25000, 50000, 30000, 80000, 40000],
    "loan_amount": [100000, 150000, 120000, 200000, 90000],
    "credit_score": [580, 750, 620, 800, 650],
    "default": [1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

print("\nLoan Dataset:")
print(df)

print("\nAverage Applicant Income:")
print(df["income"].mean())

print("\nAverage Loan Amount:")
print(df["loan_amount"].mean())


# Convert dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Display the complete dataset
print("\nLoan Dataset:")
print(df)

# Display number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names and data types
print("\nDataset Information:")
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Count defaults and non-defaults
print("\nLoan Default Counts:")
print(df["default"].value_counts())



import matplotlib.pyplot as plt
import seaborn as sns

# Visualize default vs non-default applicants
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="default")

plt.title("Loan Default Distribution")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Number of Applicants")

plt.tight_layout()
plt.show()


# Analyze default distribution
print("\nDefault Analysis")

default_counts = df["default"].value_counts().sort_index()

print("\nNumber of applicants:")
print(default_counts)

print("\nPercentage of applicants:")
print(df["default"].value_counts(normalize=True).sort_index() * 100)
