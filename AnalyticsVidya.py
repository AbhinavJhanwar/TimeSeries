# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:34:16 2018

@author: abhinav.jhanwar
"""

######### removing trend - log, cuberoot, sqrt and differencing, decomposition #############
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

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
ts['1949']
ts[:'10-01-1949']
ts['1952-02-01']

# plotting to check seasonality and trend
data.plot()
# rolling mean for every year 
data.rolling(12).mean().plot()
# increasing trend detected
# group data monthly to see monthly trend
data['month']=data.Month.dt.month
data.groupby('month')['#Passengers'].mean().plot.bar()
# july and august has max no. of passengers




# overall increasing trend and some seasonality is also present
# hence we can say timseries is not stationary

# another way to check seasonality and trend is Dickey-Fuller test
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
    
# compare Test Statistic and Critical Values: should be in similar ranges
test_stationarity(ts)

# ESTIMATING AND ELIMINATING TREND
# could be done by taking log, square root, cube root
# here we should notice that smaller values has to be reduced lesser than larger values
ts_log = np.log(ts)
ts_log.plot()

ts_cuberoot = np.cbrt(ts)
ts_cuberoot.plot()

ts_squareroot = np.sqrt(ts)
ts_squareroot.plot()

# sometime there is noise in the data and we are not able to see the trend
# clearly. some techniques to remove this trend are:
# 1) Aggregation-taking average for a specific time period like weekly/monthly
# 2) Smoothing-taking rolling averages
# 3) Polynomical fitting-fit a regression model
    
# 2) SMOOTHING
# a) using a specific time period
moving_avg = ts_log.rolling(12).mean()
# since we are taking average over last 12 months and hence initial 11 values
# will not be defined here
plt.figure(figsize=(15,7))
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
# removing inital 11 values
ts_log_moving_avg_diff.dropna(inplace=True)
# test stationarity again
test_stationarity(ts_log_moving_avg_diff)
# here test statistic value is smaller than 10% and 5% and hence we can say
# that there is 95% confidance that series is stationary series

# b) taking weighted average when a period is hard to define like stock prices
# weighted moving average- recent values are given higher weight
# exponentially weighted moving average
# http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-moment-functions
expweighted_avg = ts_log.ewm(halflife=12).mean()
plt.figure(figsize=(15,7))
plt.plot(ts_log)
plt.plot(expweighted_avg, color='red')
ts_log_ewm_diff= ts_log - expweighted_avg
test_stationarity(ts_log_ewm_diff)
# this is with 99% confidance as test statistics has values lower than 1% critical value

# trend and seasonlity doesn't always eliminated this way, particularly data
# with high seasonality and hence we need to try other methods like-
# 1) Differencing- taking difference with particular time lag
# 2) Decomposition- modeling both trends and seasonality and removing them

# 1) Differencing
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
plt.plot(ts_log_diff)
test_stationarity(ts_log_diff)

ts_log_diff2 = ts_log_diff- ts_log_diff.shift()
ts_log_diff2.dropna(inplace=True)
plt.plot(ts_log_diff2)
test_stationarity(ts_log_diff2)

# 2) Decomposing
decomposition = seasonal_decompose(ts_log, model='additive')
# extract trend
trend = decomposition.trend
# extract seasonality
seasonal = decomposition.seasonal
# extract residual
residual = decomposition.resid

# method 1
result = decomposition.plot()

# method 2
plt.figure(figsize=(12,10))
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# lets check stationarity of residual
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)


# FORECASTING
#ACF and PACF plots:
lag_acf = acf(ts_log_diff, nlags=10)
lag_pacf = pacf(ts_log_diff, nlags=10, method='ols')

#Plot ACF: 
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
# p – The lag value where the PACF chart crosses the upper confidence interval 
# for the first time. If you notice closely, in this case p=2.
# can also be considered as when it goes negative
# q – The lag value where the ACF chart crosses the upper confidence interval 
# for the first time. If you notice closely, in this case q=2.
# can also be considered as when it goes negative

# AR MODEL
# order = (p,d,q)
model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.figure(figsize=(15,10))
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

# MA MODEL
model = ARIMA(ts_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.figure(figsize=(15,10))
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

# COMBINED MODEL
model = ARIMA(ts_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.figure(figsize=(15,10))
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues[:], color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

# TAKING BACK TO ORIGINAL SCALE
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

# taking cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

# taking log
predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

# taking exponential
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.figure(figsize=(15,8))
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))