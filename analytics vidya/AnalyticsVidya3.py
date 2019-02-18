# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:04:48 2018

@author: abhinav.jhanwar
"""

######### for ExponentialSmoothing, SimpleExpSmoothing, Holt #########

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm

# read datasets
df = pd.read_csv("Train_SU63ISt.csv")
test = pd.read_csv("Test_0qrQsBZ.csv")

# check dataset columns
df.columns
# check dataset column types
df.dtypes
# check dataset shape
df.shape

# convert Datetime column from object type to Datetime
df['Datetime'] = pd.to_datetime(df.Datetime)
test['Datetime'] = pd.to_datetime(test.Datetime)

# set index as Datetime column
df.index = df['Datetime']
df.sort_index(inplace=True)

test.index = test['Datetime']
test.sort_index(inplace=True)

# convert data to daily basis
df = df.resample('D').mean()
test = test.resample('D').mean()

# divide df into train and validation set
# 14 months for training and 2 months for testing
train = df['2012-9':'2013-10'].fillna(0)
valid = df['2013-11':'2013-12']

# plot data
train.Count.plot(figsize=(15,7), title='Daily Ridership', fontsize=14)
valid.Count.plot(figsize=(15,7), title='Daily Ridership', fontsize=14)

#################### simple exponential smoothing #############
# recent observaions are assigned higher weights
# it requires a list of weights (which should add up to 1). 
''' 
 For example if we pick [0.40, 0.25, 0.20, 0.15] as weights, 
 we would be giving 40%, 25%, 20% and 15% to the last 4 points respectively

 for exponential smoothing we will have to choose a smoothing parameter alpha
 i.e. the weight for the most recent value (0 to 1) and then it exponentialy
 decreases the value for other observations using formula-
 y(t+1) = alpha(y(t)) + alpha(1-alpha)(y(t-1)) + alpha(1-alpha)^2(y(t-2)) and so on..

 mostly only last two observations are in effect hence y(t+1) is usually the
 weighted some of only last two observations
 
 smoothing level will always be >0.5 otherwise second last observation will
 be assigned higher weightage
'''

y_hat_avg = valid.copy()
fit1 = ExponentialSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit.forecast(len(valid))
plt.figure(figsize=(15,8))
plt.plot(train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SES))
print(rms)

################ Holt's Linear Trend model ###################
# extension of simple exponential smoothing
# this method takes in account of trend as well
# forecast = combination of weighted average using alpha and trend using beta
# multiplicative modeling- it multiplies the weighted average and trend for forecasting
# this is used when trend is exponential
# additive modeling- it adds the weighted avearge and trend
# this is used when trend is linear
# for details see tutorial file

sm.tsa.seasonal_decompose(train.Count).plot()
result = sm.tsa.stattools.adfuller(train.Count)
plt.show()

y_hat_avg = valid.copy()

fit2 = Holt(np.asarray(train['Count'])).fit(smoothing_level=0.6, smoothing_slope=0.01)
y_hat_avg['Holt_linear'] = fit2.forecast(len(valid))

plt.figure(figsize=(15,8))
plt.plot(train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_linear))
print(rms)


################ Holt's Winter model ###################
# this takes into account seasonality along with trend
# alpha- for level: weighted average
# beta- for trend
# gamma- for seasonality
# st- indicates seasonal period
# seasonality- it will take the weightage average as: gamma(y(t)) + (1-gamma)(y(t-st))
# additive and mulitplications modelling same as Holt's trend model

# seasonal_period = 7 as data repeats itself weekly
y_hat_avg = valid.copy()
fit3 = ExponentialSmoothing(np.asarray(train['Count']) ,
                            seasonal_periods=7 ,
                            trend='add', 
                            seasonal='add',
                            ).fit(smoothing_level=0.6, 
                            smoothing_slope=0.05, 
                            smoothing_seasonal=0.28)
y_hat_avg['Holt_Winter'] = fit3.forecast(len(valid))
plt.figure(figsize=(15,8))
plt.plot(train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter))
print(rms)


###################### SARIMAX Model ####################
y_hat_avg = valid.copy()
fit4 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit4.predict(start="2013-11-1", end="2013-12-31", dynamic=True)
plt.figure(figsize=(15,8))
plt.plot(train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SARIMA))
print(rms)
