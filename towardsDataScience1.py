# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:53:16 2018

@author: abhinav.jhanwar
"""

############ EXPLORING SUPERSTORE DATA- NON PERIODIC IS PRESENT ##############3

import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

df = pd.read_excel("Superstore.xls")

# fetch only furniture data
furniture = df.loc[df['Category'] == 'Furniture']

# print minimum date and maximum date of furniture data
furniture['Order Date'].min(), furniture['Order Date'].max()

# drop all the columns which are not required to predict sales
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 
        'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 
        'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 
        'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)

# sort based on order date
furniture = furniture.sort_values('Order Date')

# check null values
furniture.isnull().sum()

# sum sales across a day
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

# set index to order date
furniture = furniture.set_index('Order Date')

# take monthly average
# 'MS' for month start
y = furniture['Sales'].resample('MS').mean()

# plot data
y.plot(figsize=(15, 6))

# plot data for trend and seasonality
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()


best_pdq, best_seasonal_pdq = None, None
def getBestOrder(pdq, seasonal_pdq, y):
    global best_pdq, best_seasonal_pdq
    best_aic=float("inf")
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                if best_aic>results.aic:
                    best_pdq=param
                    best_seasonal_pdq = param_seasonal
                    best_aic=results.aic
            except:
                continue

# get order for seasonal and arima models
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
'''print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))'''

getBestOrder(pdq, seasonal_pdq, y)

# perform predictions with best configuration
mod = sm.tsa.statespace.SARIMAX(y,
                                order=best_pdq,
                                seasonal_order=best_seasonal_pdq,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()      
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

#################################################
############ method 2 ############################
##################################################

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf, acf

series = furniture['Sales'].resample('MS').mean()
plt.figure(figsize=(12,5))
plt.xlabel("Years")
plt.ylabel("Sales")
plt.title("Furniture Data")
plt.plot(series)

percent_training = 0.75
split_point = round(len(series)*percent_training)
training, testing = series[0:split_point], series[split_point:]

training_log = np.sqrt(training)

lag_acf = acf(training_log, nlags=30)
lag_pacf = pacf(training_log, nlags=30, method='ols')

plt.figure(figsize=(12,5))
plt.xlabel("acf")
plt.ylabel("lag")
plt.title("ACF PLOT")
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(training_log)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training_log)), linestyle='--', color='gray')
#plt.plot(lag_acf)
plt.stem(lag_acf)
# suggests seasonality period=12, hence S=12
# at S=12, lag is positive so P=1 and Q=0
# cut-off crossed at 1 so q=1

plt.figure(figsize=(12,5))
plt.xlabel("pacf")
plt.ylabel("lag")
plt.title("PACF PLOT")
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(training_log)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training_log)), linestyle='--', color='gray')
#plt.plot(lag_acf)
plt.stem(lag_pacf)
# cut off at lag 1 so p=1

model = SARIMAX(training_log, order=(1, 0, 1), seasonal_order=(0, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)

K = len(testing)
forecast = model_fit.forecast(K)
forecast = np.square(forecast)

plt.figure(figsize=(12,5))
plt.xlabel("Years")
plt.ylabel("Sales")
plt.title("Furniture Data- RMSE: %.2f"%np.sqrt(sum((forecast-testing)**2)/len(testing)))
plt.axvline(x=series.index[split_point], color='green')
plt.plot(forecast, 'r')
plt.plot(series, 'b')

best_pdq, best_seasonal_pdq = None, None
def getBestOrder(pdq, seasonal_pdq, y, testing):
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
getBestOrder(pdq, seasonal_pdq, training_log, np.sqrt(testing))


######################################################
##################Offic Supplies Sales#######################
######################################################
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import pacf, acf

series = df[df.Category=='Office Supplies'][['Order Date', 'Sales']]
series.set_index('Order Date', inplace=True)
series = series['Sales']
series = series.resample('MS').mean()

plt.figure(figsize=(12,5))
plt.xlabel("Years")
plt.ylabel("Sales")
plt.title("Office Supplies")
plt.plot(series)

percent_training = 0.75
split_point = round(len(series)*percent_training)
training, testing = series[0:split_point], series[split_point:]

training_diff = training.diff(1)[1:]

plt.figure(figsize=(12,5))
plt.xlabel("Years")
plt.ylabel("Sales")
plt.title("Office Supplies")
plt.plot(training_diff)


lag_acf = acf(training, nlags=30)
lag_pacf = pacf(training, nlags=30, method='ols')

plt.figure(figsize=(12,5))
plt.xlabel("acf")
plt.ylabel("lag")
plt.title("ACF PLOT")
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(training)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training)), linestyle='--', color='gray')
#plt.plot(lag_acf)
plt.stem(lag_acf)

plt.figure(figsize=(12,5))
plt.xlabel("pacf")
plt.ylabel("lag")
plt.title("PACF PLOT")
plt.axhline(y=0, linestyle='-', color='black')
plt.axhline(y=1.96/np.sqrt(len(training)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(training)), linestyle='--', color='gray')
#plt.plot(lag_acf)
plt.stem(lag_pacf)

# seasonality is present but period is decreasing
# increasing trend is present