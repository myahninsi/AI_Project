# Stock Market Prediction and Trend Analysis

## Overview
This repository contains a project focused on predicting stock market trends and classifying price movements using advanced machine learning techniques. The project leverages financial indicators, feature engineering, and machine learning models to forecast stock prices and classify market behavior. An interactive Streamlit-based demo provides real-time insights.

## Problem Statement
Stock market predictions are complex due to market volatility and the influence of multiple variables. This project addresses these challenges by:
- Developing models to predict stock price movements.
- Classifying market trends using technical indicators.
- Providing an interactive platform for analyzing and visualizing predictions.

## Features
- **Stock Price Prediction**: Predicts price movements with high accuracy using LSTM and regression models.
- **Market Trend Classification**: Classifies trends with models like RandomForest and XGBoost.
- **Interactive Demo**: Streamlit-based interface for real-time model predictions.
- **Financial Indicators**: Uses EMA, MACD, RSI, and Bollinger Bands for feature engineering.

## Tools and Technologies
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy: Data manipulation and computation.
  - Matplotlib, Seaborn: Data visualization.
  - Scikit-learn: Machine learning and preprocessing.
  - XGBoost: Gradient boosting models.
  - Imbalanced-learn (SMOTE): Addressing class imbalances.
  - TA (Technical Analysis): Financial indicators like EMA, MACD, and Bollinger Bands.
- **Modeling Techniques**:
  - LSTM
  - RandomForestClassifier
  - XGBoostClassifier
  - GradientBoostingClassifier
  - Logistic Regression
  - SVM
- **Visualization Tools**: Matplotlib, Seaborn, Streamlit.
- **Data Sources**: Yahoo Finance (yfinance).

## Workflow
1. **Data Collection and Preprocessing**:
   - Fetch data using `yfinance`.
   - Clean data and handle missing values.
2. **Feature Engineering**:
   - Create technical indicators (EMA, MACD, RSI).
   - Add custom features like Bollinger Bands and Stochastic Oscillator.
3. **Exploratory Data Analysis (EDA)**:
   - Visualize relationships and trends using correlation matrices, line charts, and scatterplots.
4. **Model Development**:
   - Implement classifiers for trend prediction.
   - Optimize models using TimeSeriesSplit and SMOTE for resampling.
5. **Interactive Dashboard**:
   - Build a user-friendly interface using Streamlit for real-time model predictions and visualizations.

## Results
- Achieved **97.5% R² accuracy** for stock price predictions using LSTM.
- Developed a trend classification model with **78.7% accuracy** using RandomForest and XGBoost.
- Delivered an intuitive interface for analyzing predictions and exploring insights.