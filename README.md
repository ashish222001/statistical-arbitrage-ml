# statistical-arbitrage-ml
Statistical arbitrage strategy using machine learning-based signal generation on financial time-series data.
# Statistical Arbitrage using Machine Learning

## Overview
This project implements a statistical arbitrage framework using machine learning-based signal generation on financial time-series data. The objective is to identify trading opportunities using predictive models and evaluate performance using classification metrics.

## Problem
Financial markets are noisy and difficult to predict. This project explores whether machine learning models can generate reliable trading signals using historical price and market microstructure data.

## Approach
- Data preprocessing and feature engineering on financial time-series data
- Machine learning model training and evaluation
- Ensemble learning using Random Forest and XGBoost
- Signal generation based on model predictions
- Evaluation using classification metrics

## Models Used
- Random Forest
- XGBoost
- Ensemble (RF + XGBoost)

## Results
- Accuracy: 96.67%
- F1-score: 0.94
- ROC-AUC: up to 0.99

These results demonstrate strong predictive performance for signal generation under experimental conditions.

## Tech Stack
Python, NumPy, Pandas, Scikit-learn, XGBoost

## How to Run
1. Install dependencies
2. pip install -r requirements.txt
2. Run the model
python stat_arb_model.py

