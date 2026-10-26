

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. LOAD DATASET

df = pd.read_csv("loan_data.csv")


# ==================================================
# 2. DATA PREPROCESSING
# ==================================================

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

df.columns = df.columns.str.strip()

print("COLUMN NAMES")
print(df.columns.tolist())

# 2. Check duplicate rows
print("DUPLICATE ROWS")
print("Total duplicates:", df.duplicated().sum())



# ==================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==================================================




# Outlier Detection using Boxplots


# Outlier Detection using Four Separate Boxplots

sns.boxplot(y=df["person_income"])
plt.title("Person Income Outliers")
plt.show()

sns.boxplot(y=df["loan_amnt"])
plt.title("Loan Amount Outliers")
plt.show()

sns.boxplot(y=df["loan_int_rate"])
plt.title("Loan Interest Rate Outliers")
plt.show()

sns.boxplot(y=df["credit_score"])
plt.title("Credit Score Outliers")
plt.show()


# Objective 1: Check loan approval status

print("\nLoan status counts:")
print(df["loan_status"].value_counts())

df["loan_status"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Loan Approval Status")
plt.xlabel("Loan Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Applicants")
plt.figure()
plt.show()


# Objective 2: Education levels of applicants

print("\nEducation levels:")
print(df["person_education"].value_counts())

df["person_education"].value_counts().plot(
    kind="bar"
)

plt.title("Applicants by Education")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")
plt.figure()
plt.show()


# Objective 3: Most common loan purposes

print("\nLoan purposes:")
print(df["loan_intent"].value_counts())

df["loan_intent"].value_counts().plot(
    kind="bar"
)

plt.title("Loan Purpose Distribution")
plt.xlabel("Loan Purpose")
plt.ylabel("Number of Applicants")
plt.xticks(rotation=30)
plt.figure()
plt.show()



# Objective 4: Average Loan Amount by Age

avg_loan = df.groupby("person_age")["loan_amnt"].mean()

plt.plot(avg_loan.index, avg_loan.values, marker="o")

plt.title("Average Loan Amount by Age")
plt.xlabel("Applicant Age")
plt.ylabel("Average Loan Amount")
plt.grid(True)
plt.show()



# Objective 5: Home Ownership Distribution (Pie Chart)

df["person_home_ownership"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Distribution of Applicants by Home Ownership")
plt.ylabel("")
plt.show()


# Objective 6: Correlation between numerical columns

numeric_data = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.figure()
plt.show()


# ==================================================
# 4. PREPARE DATA FOR MACHINE LEARNING
# ==================================================

# Target column
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Separate numerical and categorical features
numeric_features = X.select_dtypes(
    include=np.number
).columns

categorical_features = X.select_dtypes(
    include="object"
).columns

# Fill missing numerical values
num_imputer = SimpleImputer(strategy="median")

X[numeric_features] = num_imputer.fit_transform(
    X[numeric_features]
)

# Fill missing categorical values
cat_imputer = SimpleImputer(strategy="most_frequent")

X[categorical_features] = cat_imputer.fit_transform(
    X[categorical_features]
)

# Convert text categories into numerical columns
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=int
)

print("\nFeatures after encoding:")
print(X.head())

print("\nTotal rows for machine learning:", len(X))


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

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Scale features for KNN and Logistic Regression
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

    "Decision Tree": DecisionTreeClassifier(
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

    # Train and predict
    if name in ["KNN", "Logistic Regression"]:

        model.fit(X_train_scaled, y_train)
        prediction = model.predict(X_test_scaled)

    else:

        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(
        y_test,
        prediction
    )

    results.append([name, accuracy])

    print("\n================================")
    print("MODEL:", name)
    print("================================")

    print(
        "Accuracy:",
        round(accuracy * 100, 2),
        "%"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Approved", "Approved"],
        yticklabels=["Not Approved", "Approved"]
    )

    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        "output/" + name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.show()

    # Precision, recall and F1-score
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
# 7. COMPARE MODEL ACCURACY
# ==================================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy"]
)

print("\nModel Accuracy Comparison:")
print(results_df)

# Save results
results_df.to_csv(
    "output/model_results.csv",
    index=False
)

# Find best model by accuracy
best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBest Model:")
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
plt.figure()
plt.show()









