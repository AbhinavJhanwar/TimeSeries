# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:23:14 2018

@author: abhinav.jhanwar
"""
############### for plotting and group functions ################3

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
train = pd.read_csv("Train_SU63ISt.csv")
test = pd.read_csv("Test_0qrQsBZ.csv")

# keep an original copy of datasets
train_original = train.copy()
test_original = test.copy()

# check dataset columns
train.columns
# check dataset column types
train.dtypes
# check dataset shape
train.shape

# convert Datetime column from object type to Datetime
train['Datetime'] = pd.to_datetime(train.Datetime)
test['Datetime'] = pd.to_datetime(test.Datetime)
train_original['Datetime'] = pd.to_datetime(train_original.Datetime)
test_original['Datetime'] = pd.to_datetime(test_original.Datetime)

# extract year, month, day and hour of data
for i in (train, test, train_original, test_original):
    i['year'] = i.Datetime.dt.year
    i['month'] = i.Datetime.dt.month
    i['day'] = i.Datetime.dt.day
    i['Hour'] = i.Datetime.dt.hour

# extract weekend
train['day of week'] = train['Datetime'].dt.dayofweek
temp = train['Datetime']

def applyer(row):
    if row.dayofweek == 5 or row.dayofweek==6:
        return 1
    else:
        return 0

temp2 = train['Datetime'].apply(applyer)
train['weekend'] = temp2

# set index as Datetime column
train.index = train['Datetime']
train.sort_index(inplace=True)

test.index = test['Datetime']
test.sort_index(inplace=True)

# drop ID column
df = train.drop('ID',1)

# get timeseries for count
ts = df['Count']

# plot data
plt.figure(figsize=(16,8))
plt.plot(ts, label='Passenger Count')
plt.title('Time Series')
plt.xlabel('Time(year-month)')
plt.ylabel('Passenger Count')
plt.legend(loc='best')

# plot data as per year
train.groupby('year')['Count'].mean().plot.bar()
# increasing with year

# plot data as per month
train['2013'].groupby('month')['Count'].mean().plot.bar()
# increasing as the end of year approaches
# taken only for 2013 as 2012 & 2014 data is not properly collected

# plot data as per day
train.groupby('day')['Count'].mean().plot.bar()

# plot data as per hours
train.groupby('Hour')['Count'].mean().plot.bar()
# peak traffic is at 7pm then a decreasing trend till 5am

# plot data as per weekdays
train.groupby('weekend')['Count'].mean().plot.bar()
# traffic is more on week days than weekends

# plot week day wise
train.groupby('day of week')['Count'].mean().plot.bar()
# lesser traffic on 5/6 than other days

# drop ID column from train data
train = train.drop('ID',1)

# Hourly time series
hourly = train.resample('H').mean()
hourly = hourly.fillna(0)

# Converting to daily mean
daily = train.resample('D').mean()
daily = daily.fillna(0)

# Converting to weekly mean
weekly = train.resample('W').mean()
weekly = weekly.fillna(0)

# Converting to monthly mean
monthly = train.resample('M').mean()
monthly = monthly.fillna(0)

# plotting complete data
fig, axs = plt.subplots(4,1)
hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0])
daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1])
weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2])
monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3])

# resample data daily wise
train = train.resample('D').mean()
test = test.resample('D').mean()

# divide train into train and validation
Train = train.ix['2012-08-25':'2014-06-24'].fillna(100)
valid = train.ix['2014-06-25':'2014-09-25'].fillna(100)   

Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train')
valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc='best')


############ predictions using moving average ###############
y_hat_avg = valid.copy()
y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(10).mean().iloc[-1] # average of last 10 observations.
plt.figure(figsize=(15,5)) 
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 10 observations')
plt.legend(loc='best')
plt.show()
y_hat_avg = valid.copy()
y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(20).mean().iloc[-1] # average of last 20 observations.
plt.figure(figsize=(15,5))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 20 observations')
plt.legend(loc='best')
plt.show()
y_hat_avg = valid.copy()
y_hat_avg['moving_avg_forecast'] = Train['Count'].rolling(50).mean().iloc[-1] # average of last 50 observations.
plt.figure(figsize=(15,5))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast using 50 observations')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.moving_avg_forecast))
print(rms)


#################### predictions using exponential smoothing #############
# newest observaions are assigned highest weights
y_hat_avg = valid.copy()
fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6,optimized=False)
y_hat_avg['SES'] = fit2.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['SES'], label='SES')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.SES))
print(rms)

################ Holt's Linear Trend model ###################
# extension of simple exponential smoothing
# this method takes in account of trend

sm.tsa.seasonal_decompose(Train.Count).plot()
result = sm.tsa.stattools.adfuller(train.fillna(100).Count)
plt.show()

y_hat_avg = valid.copy()

fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(valid))

plt.figure(figsize=(16,8))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_linear))
print(rms)

y_hat_avg = valid.copy()
fit1 = ExponentialSmoothing(np.asarray(Train['Count']) ,seasonal_periods=12 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(valid))
plt.figure(figsize=(16,8))
plt.plot(Train['Count'], label='Train')
plt.plot(valid['Count'], label='Valid')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(valid.Count, y_hat_avg.Holt_Winter))
print(rms)

