# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:11:43 2018

@author: abhinav.jhanwar
"""

################ ARIMA - Autoregressive Integrated Moving Average Model##############

# p: The number of lag observations included in the model, also called the lag order.
# d: The number of times that the raw observations are differenced, also called the degree of differencing.
# q: The size of the moving average window, also called the order of moving average.

# AR: Autoregression. A model that uses the dependent relationship between an observation and some number of lagged observations.
# I: Integrated. The use of differencing of raw observations (e.g. subtracting an observation from an observation at the previous time step) in order to make the time series stationary.
# MA: Moving Average. A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations.

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

def parser(x):
	return pd.datetime.strptime('190'+x[3:], '%Y-%m')
 
series = pd.read_csv('sales-of-shampoo-over-a-three-ye.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# check series for seasonality and trend
series.plot(figsize=(15,10), linewidth=3, fontsize=18)
# series has clear trend
# hence diff of atleast 1 degree will be required

# check series for autocorrelation
pd.plotting.autocorrelation_plot(series)
# looking at plot gives possibility that AR could be set 5 for starting

'''
First, we fit an ARIMA(5,1,0) model. This sets the lag value to 5 for 
autoregression, uses a difference order of 1 to make the time series 
stationary, and uses a moving average model of 0.
'''

train = series[:-15]
test = series[-15:]
# fit model
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# get residuals
# residuals: get the difference of actual and predicted value
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
residuals.plot(kind='kde')
print(residuals.describe())

# predictions
prediction = list()
for i in range(len(test)):
    # train on initial values and append the next test value as soon as it is forecasted
    train = series[:-13+i]
    model = ARIMA(train, order=(5,1,0))
    model_fit = model.fit(disp=0)
    # forecast next value in series
    # forecast method works as it will always predict the next value in 
    # index of time based on training data
    output = model_fit.forecast()
    yhat = output[0]
    prediction.append(yhat)
    
# plot actual vs predicted values
prediction = pd.DataFrame(prediction, index=test.index)
pyplot.plot(test)
pyplot.plot(prediction, color='red')

# DEFINE p,d,q values using appropriate ranges
def evaluate_arima_model(data, forecast_num, arima_order): 
    # forecast_num: difenes the number of samples to be done forecasting for
    train = data[:-forecast_num]
    test = data[-forecast_num:]
    # convert test and train data into list
    history = [x for x in train]    
    test = [x for x in test]
    # forecast values in for test data and append the forecast test data to
    # train data so that next value is forcasted
    predictions = list()    
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)        
        model_fit = model.fit(trend='c', disp=0)
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
evaluate_model(series, p_values, d_values, q_values)
# best values = (4,2,1); RMSE = 72.885

# FINAL PREDICTIONS AND PLOTTING
test = series[-15:]
train= series[:-15]
history = [x for x in train]    
# test = [x for x in test]
prediction = list()
for i in range(len(test)):
    model = ARIMA(history, order=(4,2,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    prediction.append(yhat)
    history.append(test[i])

pyplot.figure(figsize=(12,8))
pyplot.plot(pd.DataFrame(test, index=test.index))
pyplot.plot(pd.DataFrame(prediction, index=test.index),color='red')