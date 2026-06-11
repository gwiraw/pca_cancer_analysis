# ==========================================================
# Milestone Assignment 2: Principal Component Analysis
# Author: Wadzanai Gwira
# ==========================================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ==========================================================
# Step 1: Load Dataset
# ==========================================================

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

print("Dataset Shape:", X.shape)

# ==========================================================
# Step 2: Standardize Data
# ==========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================================
# Step 3: Apply PCA
# ==========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA Shape:", X_pca.shape)

# Explained Variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Explained Variance:")
print(sum(pca.explained_variance_ratio_))

# ==========================================================
# Step 4: Create PCA DataFrame
# ==========================================================

pca_df = pd.DataFrame(
    data=X_pca,
    columns=["Principal Component 1",
             "Principal Component 2"]
)

pca_df["Target"] = y

print("\nFirst Five Rows")
print(pca_df.head())

# ==========================================================
# Step 5: Visualize PCA Components
# ==========================================================

plt.figure(figsize=(10, 6))

colors = ['red', 'blue']
labels = ['Malignant', 'Benign']

for target, color, label in zip([0, 1], colors, labels):
    subset = pca_df[pca_df["Target"] == target]

    plt.scatter(
        subset["Principal Component 1"],
        subset["Principal Component 2"],
        c=color,
        label=label,
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Breast Cancer Dataset")
plt.legend()
plt.grid(True)

plt.savefig("pca_scatterplot.png")
plt.show()

# ==========================================================
# BONUS: Logistic Regression
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nLogistic Regression Accuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report")
print(classification_report(y_test, predictions))