import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

def main():
    """Some comments of the data 
    * Data is not stationary, neither the 1-diff serie According adfuller test. 
    * 1-diff data are normal if you take all data. But the standart deviation changes over time so the serie shows **heteroscedasticity**.
    * Data shows stationality (using a stationality test over data)

    Using an autoarime, the best model is  SARIMAX(data, order=(0,1,2), seasonal_order=(0,0,1,12)).
    Since data has Heteroskedasticity, we model the residuals of the model using a Garch(2,2) model
    """
    #Read the data
    data = pd.read_csv("train.csv", index_col= 0).dropna()
    data.index = pd.to_datetime(data.index, format='%d.%m.%y')


    # SARIMAX model
    model = SARIMAX(data, order=(0,1,2), seasonal_order=(0,0,1,12))
    result = model.fit()
    residuals = result.resid

    # GARCH model for residuals 
    model_garch = arch_model(residuals, vol='Garch',p=2, q=2)
    fitted_garch = model_garch.fit(disp='off')

    # Predict variance
    variance_prediction = fitted_garch.forecast(horizon=12)

    forecast = result.forecast(12)
    #some postprocessing
    variance = variance_prediction.variance.tail(1).T
    variance.columns = ['y']
    variance = variance['y']
    variance.index = forecast.index
    #Calculate prediction confidence intervals at 95%
    upper_bound = forecast + variance*1.96
    lower_bound = forecast - variance*1.96

    #Save the data
    data.append(forecast.to_frame('y')).assign(upper_bound = upper_bound, lower_bound = lower_bound).fillna(0).to_csv("test.csv", index=True)
    
    #finally, we plot the results
    plt.figure(figsize=(10, 6))

    plt.plot(data, label='Train data', color='blue')

    # Plot the forecast
    plt.plot(forecast, label='Forecast', color='red')

    # Plot confidence intervals
    plt.fill_between(forecast.index.tolist(), lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Interval confidence')

    # some conf
    plt.title('Train data and prediction with confidence intervals')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    # Show plot
    plt.show()
    print("Predictions are written in results.csv")

if __name__ == "__main__":
    main()
    
