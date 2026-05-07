# Credit Card Fraud Detection with Continuous Monitoring

This project implements a machine learning pipeline for credit card fraud detection using Logistic Regression, Random Forest, and XGBoost. It also includes a continuous monitoring approach using sliding windows, Population Stability Index (PSI), and adaptive retraining strategies.

## Project Overview

Credit card fraud detection is challenging because fraudulent transactions are rare and fraud patterns can change over time. This project compares static machine learning models with adaptive monitoring strategies.

## Models Used

- Logistic Regression
- Random Forest
- XGBoost

## Key Techniques

- Class imbalance handling
- Threshold optimisation
- PR-AUC and F1-score evaluation
- Sliding window monitoring
- PSI-based drift detection
- Rolling retraining
- Event-triggered retraining

## Dataset

This project uses the European Credit Card Fraud Dataset from Kaggle.  
The dataset is not included in this repository. Please download it from Kaggle and place it in the expected data folder.

## Repository Structure

```text
notebooks/   Original Colab notebook
src/         Python scripts
data/        data 
outputs/    Example charts and results
