# PayPal (Paysim) Fraud Detection using Machine Learning


Link to Assigment: https://drive.google.com/drive/folders/1bpA7EiS6Y1hfCO3l5mpP79qRHU1ad_S9?usp=sharing


This project aims to detect fraudulent financial transactions using the PaySim dataset, a simulated dataset modeled on real financial data. The project applies machine learning techniques to handle class imbalance and build robust classification models.


# Project Structure
Finalised_Dataset_paysim_cleaned.csv
EDA_Notebook (Milestone 1 Assignment).ipynb
Modeling_Notebook (Module 4 assignment).ipynb
README

# Project Goals

- Understand the data and perform Exploratory Data Analysis (EDA)
- Address severe class imbalance using SMOTE
- Train and evaluate Logistic Regression and XGBoost models
- Compare model performance using accuracy, precision, recall, F1-score, and ROC AUC
- Visualize confusion matrix and feature importance

# Dataset Overview

Dataset: `Finalised_Dataset_paysim_cleaned.csv`  
Total Records: ~6.36 million  
Target Variable: `isfraud`  
Class Imbalance: ~0.13% of transactions are fraudulent

# Preprocessing Steps

- Removed irrelevant columns: 'nameorig', 'namedest'
- Encoded transaction `type` into numerical format
- Created custom error features to capture balance inconsistencies
- Handled class imbalance using "SMOTE" to balance fraud vs. non-fraud

# Models Used

1. Logistic Regression
- A simple linear classifier
- Acts as a baseline model

2. XGBoost Classifier
- A gradient-boosted tree-based model
- High performance on tabular data
- Tuned for `eval_metric='logloss'`

# Results Summary

 Metric        | Logistic Regression 	|	 XGBoost 
........................................................... 
Accuracy       |	96%		|	100%
 ......................................................
Precision      | 	 0.95 – 0.97    | 	 1.00 
....................................................... 
Recall         | 	 0.95 – 0.97    |    	 1.00 
.......................................................
 F1-Score      | 	0.96	 	|	1.00 	
.......................................................
 ROC AUC Score | 	0.9919		|	0.99998


Note: Results may have slight overfitting due to the effect of SMOTE while balancing the dataset synthetically


# Visualizations

- Confusion Matrix: Displays classification results
- ROC Curve: Assesses model performance across thresholds
- Feature Importance (XGBoost): Identifies key drivers of fraud

# Requirements

Install dependencies using:

bash
pip install -r requirements.txt
