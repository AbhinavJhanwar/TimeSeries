# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 11:43:57 2018

@author: abhinav.jhanwar
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# read data file
# method 1
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
# method 2
#data = pd.read_csv('AirPassengers.csv')
# convert 'Month' to datetime object
#data.Month = pd.to_datetime(data.Month)
# set 'Month' as index of data
#data.set_index('Month', inplace=True)
data.info()

# printing specific data rows
ts = data['#Passengers']

data.plot()

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    # Plot rolling statistics:
    plt.figure(figsize=(15,7))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
ts_log = np.log(ts)
ts_log.plot()

# remove trend
moving_avg = ts_log.rolling(12).mean()
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
ts_log_moving_avg_diff.plot()
test_stationarity(ts_log_moving_avg_diff)

# remove seasonality
ts_log_diff = ts_log_moving_avg_diff - ts_log_moving_avg_diff.shift()
ts_log_diff.dropna(inplace=True)
plt.plot(ts_log_diff)
test_stationarity(ts_log_diff)


# FORECASTING
# DEFINE p,d,q values using appropriate ranges
def evaluate_arima_model(data, forecast_num, arima_order): 
    # forecast_num: difenes the number of samples to be done forecasting for
    train = data[:-forecast_num-1]
    test = data[-forecast_num:]
    # convert test and train data into list
    history = [x for x in train]    
    test = [x for x in test]
    # forecast values in for test data and append the forecast test data to
    # train data so that next value is forcasted
    predictions = list()    
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)        
        model_fit = model.fit(disp=-1)
        yhat = model_fit.forecast()[0]        
        predictions.append(yhat)        
        history.append(test[t])
    # calculate out of sample error
    mse = mean_squared_error(test, predictions)
    rmse = np.sqrt(mse)
    return rmse

# initially set best configuration as None
best_cfg = None

# function to evaluate combinations of p, d, and q values for an ARIMA model
def evaluate_model(dataset, p_values, d_values, q_values):
    best_score = float("inf") 
    global best_cfg  
    for p in p_values:        
        for d in d_values:            
            for q in q_values:                
                order = (p,d,q)                
                try:
                    mse = evaluate_arima_model(data=dataset, forecast_num=15, arima_order=order)                    
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s, RMSE=%.3f' %(order,mse))
                except:
                    continue
    # print best configuration or values of p, d & q
    print('Best ARIMA%s RMSE=%.3f' %(best_cfg, best_score))            

# set range for p, d & q values to be tested for best configuration
p_values = range(0,5)
d_values = range(0,3)
q_values = range(0,5)

# print best values for p,d,q
evaluate_model(ts_log_diff, p_values, d_values, q_values)

# FINAL PREDICTIONS AND PLOTTING
test = ts_log_diff[-15:]
train = ts_log_diff[:-16]
history = [x for x in train]    
test = [x for x in test]
prediction = list()
for i in range(len(test)):
    model = ARIMA(history, order=best_cfg)
    model_fit = model.fit(disp=-1)
    output = model_fit.forecast()
    yhat = output[0]
    prediction.append(yhat)
    history.append(test[i])
    
plt.plot(test)
plt.plot(prediction[:], color='red')