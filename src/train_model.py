from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier



# Find the main project folder
project_folder = Path(__file__).resolve().parents[1]

# Load the dataset
dataset_path = project_folder / "data" / "cs-training.csv"
df = pd.read_csv(dataset_path)

# Remove the unnecessary index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Separate features and target
X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

# Display dataset information
print("Features shape:", X.shape)
print("Target shape:", y.shape)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the split sizes
print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# Create a preprocessing pipeline
preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

print("\nPreprocessing pipeline created successfully.")


# Create a complete pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("\nModel pipeline created successfully.")


# Train the model using training data
model.fit(X_train, y_train)

print("\nModel trained successfully.")


# Evaluate the model on test data
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("Accuracy:", round(accuracy * 100, 2), "%")


# Detailed model evaluation
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Display the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Delinquency (0)", "Delinquency (1)"]
)

display.plot(cmap="Blues")
plt.title("Loan Default Prediction - Confusion Matrix")
plt.show()



# Create the Random Forest model
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

print("\nRandom Forest model created successfully.")


# Train the Random Forest model
rf_model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully.")


# Evaluate Random Forest on test data
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Evaluation")
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")


# Detailed evaluation of Random Forest
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))


# Compare Logistic Regression and Random Forest
print("\n===== MODEL COMPARISON =====")

print("Logistic Regression Accuracy:",
      round(accuracy * 100, 2), "%")

print("Random Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")


# Compare class 1 recall
from sklearn.metrics import recall_score

lr_recall = recall_score(y_test, y_pred, pos_label=1)
rf_recall = recall_score(y_test, rf_pred, pos_label=1)

print("\n===== CLASS 1 RECALL COMPARISON =====")
print("Logistic Regression Recall:",
      round(lr_recall * 100, 2), "%")
print("Random Forest Recall:",
      round(rf_recall * 100, 2), "%")

# Save Random Forest model
import joblib

model_path = project_folder / "models" / "loan_default_model.pkl"

joblib.dump(rf_model, model_path)

print("\nRandom Forest model saved successfully.")

# Save model performance metrics
metrics = {
    "accuracy": rf_accuracy,
    "recall": rf_recall
}

metrics_path = project_folder / "models" / "model_metrics.pkl"

joblib.dump(metrics, metrics_path)

print("Model metrics saved successfully.")