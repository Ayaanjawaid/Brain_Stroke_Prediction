## Project Overview
This project aims to predict the likelihood of a stroke based on various health parameters using machine learning models. The dataset is preprocessed, analyzed, and multiple models are trained to achieve the best prediction accuracy.

## Libraries Used
Pandas: For data manipulation and analysis.
NumPy: For numerical operations.
Matplotlib and Seaborn: For data visualization.
Scikit-learn: For implementing machine learning models.
XGBoost: For the implementation of the XGBoost model.

## Data Preprocessing
Data Loading: The dataset is loaded using Pandas.
Data Cleaning: Missing values are handled, and unnecessary columns are removed.
Feature Engineering: New features are created to enhance model performance.
Encoding: Categorical variables are encoded using one-hot encoding.

## Exploratory Data Analysis (EDA)
Visualization: Various plots (histograms, bar plots, correlation heatmaps) are used to understand the distribution and relationships of the data.
Statistical Analysis: Summary statistics are computed to gain insights into the dataset.

## Model Training
Multiple machine learning models are trained to predict strokes. The models used include:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
XGBoost

## Model Evaluation
Confusion Matrix: Used to evaluate the performance of the classification models.
Accuracy, Precision, Recall, and F1-Score: Computed for each model to compare their performance.
ROC Curve and AUC Score: Analyzed to understand the models' ability to distinguish between classes.
