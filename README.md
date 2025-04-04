  Fraud Detection with Decision Trees
This project implements a banknote fraud detection system. Developed in Python using Google Colab.

  Project Description  :  The dataset contains 4 features of banknotes (variance, skewness, kurtosis, entropy) 
and class information (0: Authentic, 1: Fraudulent). It's an imbalanced dataset (85% authentic, 15% fraudulent transactions).

  Main Tasks:
1. Dataset loading and analysis
2. Decision tree model training
3. Hyperparameter optimization (criterion and max depth)
4. Model performance evaluation
5. Visualization of tree structure and feature importance

   Technology Stack
- Python 3.x
- Google Colab
- Libraries:
  - pandas, numpy
  - matplotlib, seaborn
  - scikit-learn

 Dataset
Source: [Google Drive](https://drive.google.com/uc?export=download&id=1xoUonHBmS-iFarordEh3RwGy8k tiBUZ)

Features: 
-Variance
-Skewness
-Kurtosis
-Entropy
-Class (target variable: 0=Authentic, 1=Fraudulent)

        Methodology

Data Analysis:
-Missing value check
-Class distribution analysis
-Pairwise feature distribution visualization

Modeling:
-80% training, 20% test split
-Decision tree classification
-Hyperparameter combinations tested:
-Criterion: 'gini' and 'entropy'
-Max depth: 3, 5, 7

Evaluation:
-Classification report (precision, recall, f1-score)
-Confusion matrix
-Decision tree visualization
-Feature importance analysis
