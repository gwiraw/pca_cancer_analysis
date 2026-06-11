# Milestone Assignment 2: Principal Component Analysis

## Overview

This project demonstrates the use of Principal Component Analysis (PCA) on the Breast Cancer dataset from Scikit-Learn.

The objective is to reduce the dimensionality of the dataset while preserving as much information as possible and identify the most important variables.

## Dataset

The dataset used is the Breast Cancer Wisconsin Diagnostic Dataset available in sklearn.datasets.

Original Features: 30

Samples: 569

Classes:

* Malignant
* Benign

## Tasks Completed

1. Loaded the Breast Cancer dataset.
2. Standardized the data using StandardScaler.
3. Applied PCA.
4. Reduced the dataset to two principal components.
5. Visualized the transformed data.
6. Implemented Logistic Regression for prediction (Bonus Task).

## Installation

Install dependencies:

pip install pandas matplotlib scikit-learn

## Running the Program

python pca_cancer_analysis.py

## Output

The program produces:

* PCA-transformed dataset
* Explained variance ratio
* Scatter plot of two principal components
* Logistic Regression accuracy score
* Classification report

## Author

Your Name
