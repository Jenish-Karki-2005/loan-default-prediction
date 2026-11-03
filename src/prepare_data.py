
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Find the project folder
project_folder = Path(__file__).resolve().parents[1]

# Load the real dataset
dataset_path = project_folder / "data" / "cs-training.csv"
df = pd.read_csv(dataset_path)

# Remove the unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Separate features (X) and target (y)
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

# Display the results
print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())




# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the shapes of the split data
print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# Fill missing values using the median
imputer = SimpleImputer(strategy="median")

# Learn median values from training data and fill missing values
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

# Use the same median values to fill missing values in testing data
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Check whether any missing values remain
print("\nMissing values in training data:", X_train.isnull().sum().sum())
print("Missing values in testing data:", X_test.isnull().sum().sum())


# Display the final dataset sizes
print("\nFinal training data shape:", X_train.shape)
print("Final testing data shape:", X_test.shape)

# Check the target distribution in training data
print("\nTraining target distribution:")
print(y_train.value_counts())
