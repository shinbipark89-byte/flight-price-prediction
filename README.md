# Flight Price Prediction Using Machine Learning

This project explores how different machine learning models can be used to predict airline ticket prices using real-world flight data.
The primary goal is to compare models of increasing complexity and understand the trade-offs between accuracy, stability, and interpretability in airfare prediction.


# Dataset

The dataset consists of approximately 11,000 domestic flight records from India’s airline market.
Each observation includes information such as:

- Airline
- Source and destination
- Number of stops
- Flight duration
- Departure and arrival times
- Date of journey
T- icket price

For consistency and interpretability, all prices were converted from INR to USD.
Several variables originally stored as strings were cleaned and transformed into numeric values.


# Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_linear.ipynb
│   ├── 04_model_tree.ipynb
│   ├── 05_model_bagging.ipynb
│   ├── 06_model_xgboost.ipynb
│   └── 07_model_comparison.ipynb
├── src/
│   ├── preprocessing.py
│   └── metrics.py
├── reports/
│   └── figures/
├── requirements.txt
└── README.md


## Modeling Approaches

Models were built and evaluated in increasing order of complexity:

1. Linear & Multiple Regression
- Baseline model for interpretability
- Identifies key pricing factors
2. Decision Tree Regressor
- Captures nonlinear relationships
- Includes cost-complexity pruning to control overfitting
3. Bagging Regressor
- Reduces variance by averaging multiple decision trees
- Demonstrates improved stability and generalization
4. XGBoost Regressor
- Gradient boosting framework for high predictive accuracy
- Tuned using cross-validation and early stopping

All models share a consistent preprocessing pipeline and are evaluated using RMSE, MAE, and R².


## Key Findings

- Flight duration, number of stops, airline, and journey timing consistently emerged as the most important predictors.
- Linear regression provided strong interpretability but limited predictive power.
- Tree-based models significantly improved performance by capturing nonlinear pricing patterns.
- Bagging reduced variance and improved robustness.
- XGBoost achieved the best overall predictive performance, balancing flexibility and generalization.


## Notebook Guide

- 01_eda.ipynb
Exploratory analysis to understand price distributions and feature behavior.
- 02_preprocessing.ipynb
Data cleaning, feature engineering, and dataset construction.
- 03_model_linear.ipynb
Linear and multiple regression models for baseline analysis.
- 04_model_tree.ipynb
Decision tree modeling with pruning and interpretability analysis.
- 05_model_bagging.ipynb
Ensemble learning using bagging to reduce variance.
- 06_model_xgboost.ipynb
Gradient boosting model with hyperparameter tuning.
- 07_model_comparison.ipynb
High-level comparison of all models and discussion of trade-offs.


## Takeaways

This project shows how progressively more flexible models improve performance in airfare prediction while introducing trade-offs in interpretability and complexity.
The final comparison suggests that model choice should depend on the intended use case—whether prioritizing explanation or predictive accuracy.