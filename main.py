import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

class GDPForecast:
    """
    A class that forecasts GDP using Support Vector Regression (SVR).
    Includes data preprocessing, model optimization, training, and forecasting functionalities.
    """

    def __init__(self, gdp_data):
        """
        Initializes the GDPForecast with historical GDP data.
        :param gdp_data: A list of GDP values.
        """
        self.gdp_data = np.array(gdp_data).reshape(-1, 1)
        self.years = np.arange(len(gdp_data)).reshape(-1, 1)

    def optimize_model(self):
        """
        Optimizes the SVR model using GridSearchCV to find the best hyperparameters.
        """
        parameters = {'kernel': ['rbf'], 'C': [1e3, 1e4, 1e5], 'gamma': [0.01, 0.1, 1]}
        svr = SVR()
        self.model = GridSearchCV(svr, parameters, cv=5)
        self.model.fit(self.years, self.gdp_data.ravel())
        print(f"Optimized parameters: {self.model.best_params_}")
        print(f"Model score: {self.model.best_score_}")

    def forecast(self, years_ahead=3):
        """
        Forecasts GDP for the specified number of years ahead.
        :param years_ahead: Number of years to forecast.
        :return: A list of forecasted GDP values.
        """
        future_years = np.arange(len(self.gdp_data), len(self.gdp_data) + years_ahead).reshape(-1, 1)
        return self.model.predict(future_years)

# Example usage (with placeholder GDP data)
if __name__ == "__main__":
    # Placeholder GDP data - replace with your actual GDP data
    gdp_data = [500, 550, 600, 650, 700, 750, 800, 850, 900, 950] # Replace with actual data
    forecaster = GDPForecast(gdp_data)
    forecaster.optimize_model()
    future_gdp = forecaster.forecast(years_ahead=3)
    print(f"Forecasted GDP for the next {3} years: {future_gdp}")
