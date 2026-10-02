import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Create folders if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)


# Load Iris dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(iris.target)


# Save dataset as CSV
df = X.copy()
df["species"] = [iris.target_names[i] for i in y]
df.to_csv("data/iris.csv", index=False)

print("Dataset loaded successfully!")
print("Number of samples:", len(X))
print(X.head())


# -----------------------------
# Graph 1: Class Distribution
# -----------------------------

class_counts = y.value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    iris.target_names,
    class_counts
)

plt.xlabel("Iris Species")
plt.ylabel("Number of Samples")
plt.title("Iris Class Distribution")

plt.tight_layout()
plt.savefig("graphs/class_distribution.png")
plt.close()


# -----------------------------
# Graph 2: Scatter Plot
# -----------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X["sepal length (cm)"],
    y=X["petal length (cm)"],
    hue=[iris.target_names[i] for i in y],
    s=80
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Sepal Length vs Petal Length")

plt.tight_layout()
plt.savefig("graphs/scatter_plot.png")
plt.close()


# -----------------------------
# Split Dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------
# Train Logistic Regression
# -----------------------------

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("\nModel training completed!")


# -----------------------------
# Make Predictions
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)


# -----------------------------
# Classification Report
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "graphs/confusion_matrix.png"
)

plt.close()


# -----------------------------
# Save Model
# -----------------------------

joblib.dump(
    model,
    "models/iris_model.pkl"
)

print("\nModel saved successfully!")

print("Location: models/iris_model.pkl")

print("\nAll graphs generated successfully!")