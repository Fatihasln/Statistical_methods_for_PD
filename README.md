Credit Risk Prediction and Customer Segmentation
An end-to-end data science project in R that performs exploratory data analysis, unsupervised customer clustering, and supervised predictive modeling to assess credit loan default risk. The final model, XGBoost, achieves an ROC-AUC of 0.948 on the test set.

Table of Contents
Project Overview

Project Workflow

Key Findings

Model Performance

Feature Importance

Setup and Usage

Libraries Used

Project Overview
This project tackles the critical business problem of credit risk assessment. Using a dataset of loan applicants, it aims to build a robust machine learning model to predict the probability of a loan default. Additionally, the project employs unsupervised learning to identify distinct customer segments, providing valuable insights for targeted marketing and risk management.

The analysis is contained within the Statistical_Learning.ipynb R notebook, which covers all steps from data cleaning to final model evaluation.

Project Workflow
The project follows a structured data science workflow:

Exploratory Data Analysis (EDA): The dataset is loaded, inspected for outliers and anomalies, and visualized to understand variable distributions and correlations. Missing values are imputed using the median.

Feature Engineering:

Categorical variables are converted to a numerical format using one-hot encoding.

A new feature, income_loan_ratio, is created to capture the relationship between an applicant's income and the loan amount.

All features are standardized to ensure they contribute equally to the models.

Unsupervised Learning (Clustering):

Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset, retaining 95% of the variance with 15 components.

K-Means Clustering is performed on the PCA-transformed data to identify natural customer segments. The Elbow Method identified an optimal K=3.

Supervised Learning (Classification):

The data is split into 80% training and 20% testing sets.

Three models are trained:

Logistic Regression (Baseline)

Random Forest

XGBoost

Models are evaluated based on Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

Key Findings
Customer Segmentation
The K-Means clustering revealed three distinct customer profiles:

Cluster 1 (High Risk): Highest default rate (38%), typically rents their home, and has a 'C' loan grade.

Cluster 2 (Low Risk): Lowest default rate (10%), highest income, typically has a mortgage, and an 'A' loan grade.

Cluster 3 (Medium Risk): Moderate default rate (24%), typically rents their home, and has a 'B' loan grade.

Predictive Modeling
The XGBoost model was the top performer across all metrics.

The engineered feature, income_loan_ratio, was the most important predictor of loan default.

Other significant predictors include the loan interest rate and whether the applicant rents their home.

Model Performance
The final models were evaluated on the test set, with the following results:

Model	                    Accuracy	         Precision        	Recall	      F1 Score	     ROC-AUC
Logistic Regression      	0.861	              0.882	            0.950	        0.915          0.871
Random Forest	            0.931	              0.924	            0.994	        0.958	         0.935
XGBoost	                  0.936	              0.929	            0.995	        0.961	         0.948

Feature Importance
The top 5 most predictive features according to the best model (XGBoost) are:

income_loan_ratio

loan_int_rate

person_home_ownershipRENT

person_income

person_emp_length


Install Libraries:
The notebook includes a setup block at the beginning that will automatically install any missing libraries required for the analysis.

Run the Code:
Execute the cells in the notebook sequentially to reproduce the analysis and results. Ensure the credit_risk_dataset.csv file is in the same directory.

Libraries Used
ggplot2

corrplot

dplyr

caret

cluster

caTools

randomForest

xgboost

pROC

reshape2
