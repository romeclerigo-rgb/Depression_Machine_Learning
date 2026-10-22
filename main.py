import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Find the CSV file in the same folder as this script
folder = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(os.path.join(folder, "*.csv"))
print("CSV files found:", files)

if not files:
    raise SystemExit("No CSV file found. Put the dataset CSV in the same folder as main.py.")

data = pd.read_csv(files[0])

print("Student Depression dataset successfully loaded!")

print("\nFIRST 10 RECORDS:")
print(data.head(10))

print("\nDATASET SIZE:")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\nCOLUMN NAMES:")
print(data.columns)

print("\nDATASET INFORMATION:")
data.info()

print("\nMISSING VALUES:")
print(data.isnull().sum())

print("\nDUPLICATE ROWS:", data.duplicated().sum())

print("\nDESCRIPTIVE STATISTICS:")
print(data.describe())

print("\nDEPRESSION OUTCOME:")
print(data["Depression"].value_counts())

# Check the categorical columns
for col in ["Gender", "Profession", "Sleep Duration", "Dietary Habits",
            "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]:
    print("\n" + col.upper() + ":")
    print(data[col].value_counts())

# NumPy analysis of Academic Pressure
pressure = np.array(data["Academic Pressure"])

print("\nACADEMIC PRESSURE ANALYSIS USING NUMPY:")
print("Mean:", np.mean(pressure))
print("Minimum:", np.min(pressure))
print("Maximum:", np.max(pressure))
print("Standard Deviation:", np.std(pressure))
print("Students with Academic Pressure >= 4:", np.sum(pressure >= 4))

# ---------------- Data Cleaning ----------------
df = data.copy()

# Remove duplicates
df = df.drop_duplicates()

# Fill missing Financial Stress with the median
df["Financial Stress"] = df["Financial Stress"].fillna(df["Financial Stress"].median())

# Remove vague "Others" answers
df = df[(df["Sleep Duration"] != "Others") & (df["Dietary Habits"] != "Others")]

# Keep students only, then drop unhelpful columns
df = df[df["Profession"] == "Student"]
df = df.drop(columns=["id", "Profession", "Work Pressure", "Job Satisfaction", "City"])

print("Rows after cleaning:", df.shape[0])
print("Missing values left:", df.isnull().sum().sum())

# ---------------- Feature Engineering ----------------
# Yes/No and Gender columns -> 0/1
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
df["Family History of Mental Illness"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})

# Ordered columns -> ordered numbers
df["Sleep Duration"] = df["Sleep Duration"].map({
    "Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3})
df["Dietary Habits"] = df["Dietary Habits"].map({
    "Unhealthy": 0, "Moderate": 1, "Healthy": 2})

# Degree has no order -> one-hot encoding
df = pd.get_dummies(df, columns=["Degree"], drop_first=True, dtype=int)

print("Final shape:", df.shape)
print(df.head())

# ---------------- Preprocessing ----------------
# Separate features and target
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training size:", X_train.shape[0])
print("Testing size:", X_test.shape[0])

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Training ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("MODEL ACCURACY:")
print(accuracy)

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
print("\nCONFUSION MATRIX:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print("\nTrue Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Depression", "Depression"]
)
display.plot(cmap="Blues")
plt.title("Confusion Matrix - Depression Prediction")
plt.show()