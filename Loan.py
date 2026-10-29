
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==================================================
# 1. LOAD DATASET
# ==================================================

df = pd.read_csv("loan_data.csv")


# ==================================================
# 2. DATA PREPROCESSING
# ==================================================

print("FIRST FIVE ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nDATASET INFORMATION:")
df.info()

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATE ROWS:")
print("Total duplicates:", df.duplicated().sum())


# ==================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==================================================

# Outlier Detection using Boxplots

sns.boxplot(y=df["person_income"])
plt.title("Person Income Outliers")
plt.tight_layout()
plt.show()

sns.boxplot(y=df["loan_amnt"])
plt.title("Loan Amount Outliers")
plt.tight_layout()
plt.show()

sns.boxplot(y=df["loan_int_rate"])
plt.title("Loan Interest Rate Outliers")
plt.tight_layout()
plt.show()

sns.boxplot(y=df["credit_score"])
plt.title("Credit Score Outliers")
plt.tight_layout()
plt.show()


# Calculate outliers using IQR

columns = [
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "credit_score"
]

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_limit) |
        (df[col] > upper_limit)
    ]

    print(f"\n{col}")
    print("Total outliers:", len(outliers))
    print(
        "Percentage:",
        round(len(outliers) / len(df) * 100, 2),
        "%"
    )


# Objective 1: Loan Approval Status

print("\nLOAN STATUS COUNTS:")
print(df["loan_status"].value_counts())

df["loan_status"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Loan Approval Status")
plt.xlabel("Loan Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.show()


# Objective 2: Education Levels

print("\nEDUCATION LEVELS:")
print(df["person_education"].value_counts())

df["person_education"].value_counts().plot(
    kind="bar"
)

plt.title("Applicants by Education")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")
plt.tight_layout()
plt.show()


# Objective 3: Loan Purposes

print("\nLOAN PURPOSES:")
print(df["loan_intent"].value_counts())

df["loan_intent"].value_counts().plot(
    kind="bar"
)

plt.title("Loan Purpose Distribution")
plt.xlabel("Loan Purpose")
plt.ylabel("Number of Applicants")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# Objective 4: Average Loan Amount by Age

avg_loan = df.groupby("person_age")["loan_amnt"].mean()

plt.plot(
    avg_loan.index,
    avg_loan.values,
    marker="o"
)

plt.title("Average Loan Amount by Age")
plt.xlabel("Applicant Age")
plt.ylabel("Average Loan Amount")
plt.grid(True)
plt.tight_layout()
plt.show()


# Objective 5: Home Ownership Distribution

df["person_home_ownership"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Distribution of Applicants by Home Ownership")
plt.ylabel("")
plt.tight_layout()
plt.show()


# Objective 6: Correlation Heatmap

numeric_data = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ==================================================
# 4. PREPARE DATA FOR MACHINE LEARNING
# ==================================================

# Target column
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Identify categorical columns
categorical_features = X.select_dtypes(
    include="object"
).columns

# Label Encoding
for col in categorical_features:

    le = LabelEncoder()

    X[col] = le.fit_transform(X[col].astype(str))

print("\nFEATURES AFTER ENCODING:")
print(X.head())


# ==================================================
# 5. TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale features for Logistic Regression, SVM and KNN

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==================================================
# 6. TRAIN FOUR CLASSIFICATION MODELS
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print("MODEL:", name)
    print("=" * 50)

    # Train and predict

    if name in [
        "KNN",
        "Logistic Regression",
        "SVM"
    ]:

        model.fit(X_train_scaled, y_train)

        prediction = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)


    # Calculate accuracy

    accuracy = accuracy_score(y_test, prediction)

    results.append([name, accuracy])

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )


    # Confusion Matrix

    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)


    # Plot Confusion Matrix

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Approved", "Approved"],
        yticklabels=["Not Approved", "Approved"]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()


    # Classification Report

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            prediction,
            labels=[0, 1],
            target_names=["Not Approved", "Approved"],
            zero_division=0
        )
    )


# ==================================================
# 7. PLOT ONE TREE FROM RANDOM FOREST
# ==================================================

rf_model = models["Random Forest"]

plt.figure(figsize=(22, 12))

from sklearn.tree import plot_tree

plot_tree(
    rf_model.estimators_[0],
    feature_names=X_train.columns,
    class_names=["Not Approved", "Approved"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("First Decision Tree in Random Forest")
plt.tight_layout()
plt.show()


# ==================================================
# 8. COMPARE MODEL ACCURACY
# ==================================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
)

print("\nMODEL ACCURACY COMPARISON:")
print(results_df.to_string(index=False))


# Find the best model by accuracy

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBEST MODEL:")
print(best_model["Model"])

print(
    "Best Accuracy:",
    round(best_model["Accuracy"] * 100, 2),
    "%"
)


# Plot model comparison

results_df.plot(
    x="Model",
    y="Accuracy",
    kind="bar",
    legend=False
)

plt.title("Classification Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()




