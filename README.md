# Loan Approval Prediction Using Machine Learning

A beginner-friendly end-to-end data science and machine learning project
that explores loan applicant data and predicts whether a loan
application is **approved** or **not approved** using classification
models.

## Project Overview

Loan approval can depend on several factors, such as income, loan
amount, interest rate, credit score, employment experience, and previous
loan defaults. In this project, we explore these factors through data
analysis and compare four machine learning classifiers.

> **Note:** This is an educational project. Its predictions should not
> be used as the sole basis for real financial or lending decisions.

## Objectives

-   Inspect and understand the loan applicant dataset.
-   Check missing values and duplicate records.
-   Explore outliers using boxplots and the IQR method.
-   Analyze loan status, education, loan purpose, and home ownership.
-   Study relationships between numerical features using a correlation
    heatmap.
-   Encode categorical features and prepare data for machine learning.
-   Train and evaluate four classification models.
-   Compare model accuracy and identify the best-performing model in
    this experiment.

## Dataset

The project uses a loan applicant dataset with **45,000 records** and
these example fields:

  -----------------------------------------------------------------------
  Column                              Description
  ----------------------------------- -----------------------------------
  `person_age`                        Applicant's age

  `person_gender`                     Applicant's gender

  `person_education`                  Education level

  `person_income`                     Applicant's income

  `person_emp_exp`                    Employment experience

  `person_home_ownership`             Home ownership category

  `loan_amnt`                         Requested loan amount

  `loan_intent`                       Purpose of the loan

  `loan_int_rate`                     Loan interest rate

  `loan_percent_income`               Loan amount as a proportion of
                                      income

  `cb_person_cred_hist_length`        Credit history length

  `credit_score`                      Applicant's credit score

  `previous_loan_defaults_on_file`    Whether previous loan defaults are
                                      recorded

  `loan_status`                       Target: `0` = Not Approved, `1` =
                                      Approved
  -----------------------------------------------------------------------

Place the dataset file (`loan_data.csv`) in the same folder as the
notebook before running it. If your dataset has a different filename or
location, update the file path in the data-loading cell.

## Exploratory Data Analysis

The notebook includes visualizations for:

-   Loan approval status distribution
-   Applicants by education level
-   Loan purpose distribution
-   Average loan amount by age
-   Applicants by home ownership
-   Correlation heatmap
-   Outlier inspection with boxplots

These charts help identify patterns in the dataset before model
training.

## Machine Learning Models

Four classification algorithms are trained and evaluated:

1.  **Logistic Regression**
2.  **Decision Tree Classifier**
3.  **K-Nearest Neighbors (KNN)**
4.  **Random Forest Classifier**

The notebook uses an 80:20 train-test split and evaluates the models
with accuracy, confusion matrices, precision, recall, and F1-score.
Feature scaling is applied for models such as Logistic Regression and
KNN.

## Results

The following are the results from the recorded project run:

  Model                           Accuracy
  --------------------------- ------------
  Logistic Regression               89.01%
  Decision Tree                     89.81%
  K-Nearest Neighbors (KNN)         89.19%
  **Random Forest**             **92.87%**

**Best-performing model in this run: Random Forest, with 92.87% test
accuracy.**

Accuracy is only one measure of performance. The dataset has more
not-approved cases than approved cases, so precision, recall, F1-score,
and the confusion matrix should also be considered---especially for the
approved class.

## Tools and Libraries

-   Python
-   Jupyter Notebook
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn

## How to Run the Project

### 1. Clone or download this repository

``` bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your GitHub repository URL and folder
name.

### 2. Install the required libraries

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Keep the files together

Your project folder should contain at least:

``` text
Loan-Approval-Prediction/
├── Loan_Approval_Prediction_with_Photos.ipynb
├── loan_data.csv
└── README.md
```

Use the actual notebook filename in your repository if it differs.

### 4. Open Jupyter Notebook

``` bash
jupyter notebook
```

Open the notebook, select the correct Python environment, and run the
cells from top to bottom.

## Project Structure

-   **Notebook:** data inspection, cleaning checks, EDA, feature
    encoding, model training, evaluation, and visualizations.
-   **Dataset:** applicant records used for the analysis.
-   **README:** project summary and setup instructions.

## Key Learning Outcomes

-   Understanding a complete beginner-level data science workflow.
-   Performing exploratory data analysis and interpreting charts.
-   Preparing categorical and numerical data for classification.
-   Training and comparing multiple machine learning models.
-   Evaluating a classifier using more than accuracy alone.

## Future Improvements

-   Use a `Pipeline` and `ColumnTransformer` to make preprocessing more
    robust.
-   Use cross-validation and hyperparameter tuning.
-   Evaluate class imbalance with precision-recall curves and additional
    metrics.
-   Compare label encoding with one-hot encoding for nominal categories.
-   Investigate unusual values, including impossible ages, before
    modeling.
-   Test performance on a separate dataset and assess potential fairness
    concerns.

## Disclaimer

This project is for learning and demonstration purposes only. It does
not provide financial advice and should not be used to make actual loan
approval decisions.
