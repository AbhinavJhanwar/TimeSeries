# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:06:29 2018

@author: abhinav.jhanwar
"""


######### autoarima #############
# pip install pyramid-arima
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima

# read data file
# method 1
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)

ts = data['#Passengers']

# plotting to check seasonality and trend
data.plot()

ts_log = np.log(ts)
ts_log.plot()

decomposition = seasonal_decompose(ts_log, model='additive')
result = decomposition.plot()

stepwise_model = auto_arima(ts_log, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())