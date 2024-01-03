GDP Forecasting Using Support Vector Regression

Overview
This Python script forecasts GDP using Support Vector Regression (SVR).
It encapsulates data preprocessing, model optimization, training, and forecasting functionalities into a class GDPForecast.

How It Works
Data Initialization: The class GDPForecast is initialized with historical GDP data.
Model Optimization: Uses GridSearchCV to find the best hyperparameters for the SVR model.
GDP Forecasting: Forecasts GDP for a specified number of years ahead using the optimized SVR model.

Features
Optimize Model: Finds the best hyperparameters for the SVR model to ensure accurate forecasting.
Forecast: Provides future GDP predictions based on historical data and the optimized model.

Output
The script will output the optimized parameters, the model score, and forecasted GDP values for the specified future years.
