# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 09:28:34 2018

@author: abhinav.jhanwar
"""

################ SARIMAX MODEL, PACF, ACF ####################

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf, acf
import itertools

data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')
series = data['#Passengers']

plt.figure(figsize=(12,5))
plt.xlabel("Years")
plt.ylabel("No. of Passengers")
plt.title("Passenger Data")
plt.plot(series)
# increasing trend is present with seasonality

# divide training and test data
percent_training = 0.70
split_point = round(len(series)*percent_training)
training, testing = series[0:split_point], series[split_point:]

# remove trend
training_log = np.log(training)
# first order differencing
# d=1
training_log_diff = training_log.diff(periods=1)[1:]

plt.figure(figsize=(12,5))
plt.xlabel("Years")
plt.ylabel("No. of Passengers")
plt.title("Passenger Data")
plt.plot(training_log_diff)
# unstable seasonality is present hence will use SARIMA and hence D=0

# determine model parameters
#########################################################
#       ACF                          PACF
# AR    geometric(gradual decrease)  significant till p lags
# MA    significant till p lags      geometric
# ARMA  geometric                    geometric
###########################################################
# nlags: number of samples for autocorrelation to be returned for
# autucorrelation
lag_acf = acf(training_log_diff, nlags=30)
# partial autocorrelation
lag_pacf = pacf(training_log_diff, nlags=30, method='ols')

plt.figure(figsize=(12,5))
plt.xlabel("no. of lag")
plt.ylabel("lag")
plt.title("ACF PLOT")
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(training)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training)), linestyle='--', color='gray')
#plt.plot(lag_acf)
plt.stem(lag_acf)
# suggests seasonality period=12, hence S=12
# at S=12, lag is positive so P=1 and Q=0
# significat lag at 1 so q=1

plt.figure(figsize=(12,5))
plt.xlabel("no. of lag")
plt.ylabel("lag")
plt.title("PACF PLOT")
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(training)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training)), linestyle='--', color='gray')
#plt.plot(lag_acf)
plt.stem(lag_pacf)
# significant lag at 1 so p=1

# train and fit model
# order = p,d,q- 1,1,1
# seasonal order = P,D,Q,S- 1,0,0,12
model = SARIMAX(training_log, order=(1,1,1), seasonal_order=(1,0,0,12), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

# define no. of forecasts to be done
K = len(testing)

# make forecasts
forecast = model_fit.forecast(K)

# transform forecast back to original scale by exponential
forecast = np.exp(forecast)

plt.figure(figsize=(12,5))
plt.xlabel("Years")
plt.ylabel("No. of Passengers")
plt.title("Passengers PLot- RMSE: %.2f"%np.sqrt(sum((forecast-testing)**2)/len(testing)))
plt.axvline(x=series.index[split_point], color='green')
plt.plot(forecast, 'r')
plt.plot(series, 'b')

best_pdq, best_seasonal_pdq = None, None
def getBestOrder(pdq, seasonal_pdq, y, testing):
    # one with lowest aic and bic should be the choice
    global best_pdq, best_seasonal_pdq
    best_rmse=float("inf")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(y,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                forecast = results.forecast(len(testing))
                rmse = np.sqrt(sum((forecast-testing)**2)/len(testing))
                if best_rmse>rmse:
                    best_pdq=param
                    best_seasonal_pdq = param_seasonal
                    best_rmse=rmse
            except:
                continue
    print("RMSE- {0}\nBest pdq- {1}\nBest seasonal pdq- {2}".format(best_rmse, best_pdq, best_seasonal_pdq))

# get order for seasonal arima model
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
getBestOrder(pdq, seasonal_pdq, training_log, np.log(testing))
# best order gives rmse of 22 approx
