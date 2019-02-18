# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:27:37 2018

@author: abhinav.jhanwar
"""

############ ALL ROUNDER- ANOMALIES, SARIMAX, HOLT WINTER, DICKEY FULLER ###############

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from sklearn.model_selection import TimeSeriesSplit

from itertools import product
from tqdm import tqdm_notebook

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

# r2_score- interpreted as percentage of variance explained by model
# absolute error- has same unit as initial series and hence interpretable metric
# squared error- used as gives higher penalties to big mistakes
# logarithmic error- practically same as abolute error, usually used with exponential trend
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

import warnings
warnings.filterwarnings('ignore')

# load data
#data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

data = pd.read_csv("multiTimeline.csv", skiprows=2,  parse_dates=['Month'], index_col='Month')
data.columns = ['diet', 'gym', 'finance']

gym = data[['gym']]
finance = data[['finance']]
diet = data[['diet']]
# plot data
plt.figure(figsize=(15,7))
plt.plot(finance)
plt.grid(True)

plt.figure(figsize=(15,7))
plt.plot(gym)
plt.grid(True)


# moving average for last observer year 12 months
gym.rolling(12).mean()['gym'][-1]

def plotMovingAverage(series, window, plot_intervals=False,  plot_anomalies=False, upper_scale=1.96, lower_scale=1.96):
    """
        series - dataframe with timeseries
        window - rolling window size 
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    
    rolling_mean = series.rolling(window=window).mean()
    
    plt.figure(figsize=(15,5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")
    
    # plot confidence intervals for smoothed values
    if plot_intervals:
        # from window as rolling mean is NaN for initial values
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + lower_scale * deviation)
        upper_bond = rolling_mean + (mae + upper_scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        
        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series<lower_bond] = series[series<lower_bond]
            anomalies[series>upper_bond] = series[series>upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
            
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)

# smoothing for last 12 months
plotMovingAverage(finance,12, True, True)
# anonmalies are finely caught here

# create a random anomaly
finance_anomaly = finance.copy()
finance_anomaly.iloc[-20] = finance_anomaly.iloc[-20] * 0.2
# say we have 80% drop of ads 
plotMovingAverage(finance_anomaly,12, True, True)
# it catches this anomaly as well

plotMovingAverage(gym,12, True, True)
# it is showing too many anomalies for this data

# using complex model: weighted average- more recent observation will have greater weight
def exponential_smoothing(series, alpha):
    
    """
        series - dataset with timestamps
        alpha - float [0.0, 1.0], smoothing parameter
    """
    # result[n] = alpha * series[n] + (1 - alpha) * result[n-1]
    # The less α is the more influence previous model values have, 
    # and the smoother the series will get
    
    result = SimpleExpSmoothing(series).fit(smoothing_level=alpha)
    return result.fittedvalues
    
def plotExponentialSmoothing(series, alphas):
    """
        Plots exponential smoothing with different alphas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters
        
    """
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(15, 7))
        for alpha in alphas:
            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series, "c", label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True);
        
plotExponentialSmoothing(finance, [0.3, 0.05])        
plotExponentialSmoothing(gym, [0.3, 0.05])     

def double_exponential_smoothing(series, alpha, beta):
    """
        series - dataset with timeseries
        alpha - float [0.0, 1.0], smoothing parameter for level
        beta - float [0.0, 1.0], smoothing parameter for trend
    """
    # first value is same as series
    # last_level, level = level, alpha*value + (1-alpha)*(level+trend)
    # trend = beta*(level-last_level) + (1-beta)*trend
    # result = level+trend
    result = Holt(series).fit(smoothing_level=alpha, smoothing_slope=beta).fittedvalues
    return result

def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
        Plots double exponential smoothing with different alphas and betas
        
        series - dataset with timestamps
        alphas - list of floats, smoothing parameters for level
        betas - list of floats, smoothing parameters for trend
    """
    
    with plt.style.context('seaborn-white'):    
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series, label = "Actual")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        
plotDoubleExponentialSmoothing(finance, alphas=[0.9, 0.02], betas=[0.9, 0.02])
plotDoubleExponentialSmoothing(gym, alphas=[0.9, 0.02], betas=[0.9, 0.02])   


def plotHoltWinters(model, series, test, gamma, plot_intervals=False, plot_anomalies=False, scaling_factor=1.96):
    """
        series - dataset with timeseries
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies 
    """
    
    # forecast for test data and future data
    forecastedValues = model.forecast(len(test)+36)
    # combine model fitted values and forecasted values
    modelValues = pd.concat([model.fittedvalues, forecastedValues])
    plt.figure(figsize=(20, 10))
    plt.plot(modelValues, "g", label = "Model")
    plt.plot(series, "b", label = "Actual")
    # get error of actual series and (fitted values + test data)
    error = mean_absolute_percentage_error(series.values, modelValues[:len(series)].values)
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))    

    # create series with index of timeseries as actual series + forecasted series
    predictedDeviation = pd.Series(index=modelValues.index)
    # set first value to be 0
    predictedDeviation[0] = 0
    
    # bond = fitted values +/- scalingfactor*deviation
    UpperBond = pd.Series(index=modelValues.index)
    UpperBond[0] = (modelValues.values[0] + scaling_factor * predictedDeviation[0])
    LowerBond = pd.Series(index=modelValues.index)
    LowerBond[0] = (modelValues.values[0] - scaling_factor * predictedDeviation[0])
    
    # for training dataset
    for i in range(1, len(series)):
        # Deviation is calculated according to Brutlag algorithm.
        predictedDeviation[i] = (gamma * np.abs(series.values[i] - modelValues.values[i]) + (1-gamma)*predictedDeviation[i-1])
        UpperBond[i] = (modelValues.values[i] + scaling_factor * predictedDeviation[i])
        LowerBond[i] = (modelValues.values[i] - scaling_factor * predictedDeviation[i])
    
    # for forecasted dataset
    for i in range(len(series), len(modelValues)):
        # increase uncertainity on each next step by multiplying deviation to 1.01
        predictedDeviation[i] = (predictedDeviation[i-1] * 1.01)
        UpperBond[i] = (modelValues.values[i] + scaling_factor * predictedDeviation[i])
        LowerBond[i] = (modelValues.values[i] - scaling_factor * predictedDeviation[i])
    
    if plot_anomalies:
        anomalies = pd.Series(index=series.index)
        anomalies[series.values<LowerBond[:len(series)].values] = \
            series[series.values<LowerBond[:len(series)].values]
        anomalies[series.values>UpperBond[:len(series)]] = \
            series.values[series.values>UpperBond[:len(series)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    if plot_intervals:
        plt.plot(UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=modelValues.index, y1=UpperBond, 
                         y2=LowerBond, alpha=0.2, color = "grey")    
        
    plt.vlines(series[-1:].index, ymin=min(LowerBond), ymax=max(UpperBond), linestyles='dashed')
    plt.axvspan(series[-1:].index, modelValues[-1:].index, alpha=0.4, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13);

def timeseriesCVscore(params, series, loss_function=mean_squared_error, n_splits=3, slen=12):
    """
        Returns error on CV  
        
        params - vector of parameters for optimization
        series - dataset with timeseries
        slen - season length for Holt-Winters model
    """
    # errors array
    errors = []
    
    values = series.values
    alpha, beta, gamma = params
    
    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits) 
    
    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        model = ExponentialSmoothing(values[train] ,
                            seasonal_periods=slen ,
                            trend='add', 
                            seasonal='add',
                            ).fit(smoothing_level=alpha, 
                            smoothing_slope=beta, 
                            smoothing_seasonal=gamma)
        predictions = model.forecast(len(test))
        
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)
        
    return np.mean(np.array(errors))

# initializing model parameters alpha, beta and gamma
x = [0, 0, 0] 

# Minimizing the loss function 
# TNC - Truncated Newton conjugate gradient
opt = minimize(timeseriesCVscore, x0=x, 
               args=(gym[:-12], mean_squared_log_error, 11), 
               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
              )

# Take optimal values...
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)

# ...and train the model with them, forecasting for the next 50 hours
model = ExponentialSmoothing(gym[:-12] ,
                            seasonal_periods=12 ,
                            trend='add', 
                            seasonal='add',
                            ).fit(smoothing_level=alpha_final, 
                            smoothing_slope=beta_final, 
                            smoothing_seasonal=gamma_final)
# model.aic
# model.bic
# model.resid
# model.season
# model.fittedvalues
# model.level
# model.sse
# model.slope
# model.aicc
plotHoltWinters(model, gym.gym, gym.gym[-12:], plot_intervals=True, plot_anomalies=True, gamma=0.05)

################ for diet ##########
opt = minimize(timeseriesCVscore, x0=x, 
               args=(data.diet[:-12], mean_squared_log_error, 11), 
               method="TNC", bounds = ((0, 1), (0, 1), (0, 1))
              )
# Take optimal values...
alpha_final, beta_final, gamma_final = opt.x
print(alpha_final, beta_final, gamma_final)
model = ExponentialSmoothing(data.diet[:-12] ,
                            seasonal_periods=12 ,
                            trend='add', 
                            seasonal='add',
                            ).fit(smoothing_level=alpha_final, 
                            smoothing_slope=beta_final, 
                            smoothing_seasonal=gamma_final)
plotHoltWinters(model, data.diet, data.diet[-12:], plot_intervals=True, plot_anomalies=True, gamma=gamma_final)


def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        ###################################
        # DICKEY FULLER
        # x(t)=rho*x(t−1)+e(t)
        # 
        #### as we increase the rho value the deviation of time series increases
        #### around its mean value
        #
        #### for rho=1 there will be nothing that will bring x(t) to its mean value
        #### back once it reaches the critical value and hence time series become
        #### non stationary. for this first difference will make series stationary 
        #
        # for details look medium tutorial file
        ###################################
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
############## NON STATIONARY - mean/variance/seasonality ##########

# check variance and trend through first plot
tsplot(data.diet, lags=30)

# remove seasonality for 12 months
diet_diff = data.diet.diff(12)
tsplot(diet_diff[12:], lags=30)
# autocorrelation function still has too many significant lags
# lets take difference
diet_diff = diet_diff.diff(1)
tsplot(diet_diff[13:], lags=30)
'''
 Our series now look like something undescribable, oscillating around zero, 
 Dickey-Fuller indicates that it’s stationary and the number of significant 
 peaks in ACF has dropped
'''

##############################################################################
########################## SARIMA(p,d,q)(P,D,Q,s) ############################
################ Seasonal Autoregression Moving Average model ################
##############################################################################

# AR(p) - autoregression model
''' 
 regression of the time series onto itself.
 current series values depend on its previous values with some lag. The maximum 
 lag in the model is referred to as p. To determine the initial p you need to 
 have a look at PACF plot — find the biggest significant lag, after which most 
 other lags are becoming not significant.
'''
# MA(q) — moving average model
'''
 it models the error of the time series.
 current error depends on the previous with some lag, which is referred to as q. 
 Initial value can be found on ACF plot with the same logic.
'''
# I(d) - order of integration
'''
 It is simply the number of nonseasonal differences needed for making the 
 series stationary.
'''
# S(s)- responsible for seasonality and hence equals to season of a series
# P- order of autoregression for seasonal component
'''
 can be derived from PACF, but this time you need to look at the number of 
 significant lags, which are the multiples of the season period length, for 
 example, if the period equals 24 and looking at PACF we see 24-th and 48-th 
 lags are significant, that means initial P should be 2.
 '''
# Q — same logic, but for the moving average model of the seasonal component use ACF plot
# D - order of seasonal integration
'''
 Can be equal to 1 or 0, depending on whether seasonal differences were 
 applied or not
'''

tsplot(diet_diff[13:], lags=36)
# p - 1 (since it’s the last significant lag on PACF after which most others are becoming not significant.)
# q - 1 (similar reason but on ACF)
# d - 1 (as we had first differences)
# P - 1 (since 12-th lag is somewhat significant on PACF, while 24-th is not)
# Q - 1 (same reason but on ACF)
# D- 1 (as seasonal difference is required)
# S- 12 (12 months seasonality)

########### automatic selection of parameters ###########
ps = range(1, 5)
d=1 
qs = range(1, 5)
Ps = range(0, 3)
D=1 
Qs = range(0, 2)
s = 12 # season length is still 1

# creating list with all the possible combinations of parameters
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

def optimizeSARIMA(series, parameters_list, d, D, s):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order in ARIMA model
        D - seasonal integration order 
        s - length of season
    """
    
    results = []
    best_aic = float("inf")

    for param in tqdm_notebook(parameters_list):
        # we need try-except because on some combinations model fails to converge
        try:
            model=sm.tsa.statespace.SARIMAX(series, order=(param[0], d, param[1]), 
                                            seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        # saving best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # sorting in ascending order, the lower AIC is - the better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table
    

result_table = optimizeSARIMA(data.gym, parameters_list, d, D, s)

# set the parameters that give the lowest AIC
p, q, P, Q = result_table.parameters[0]
best_model = sm.tsa.statespace.SARIMAX(data.diet, order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(best_model.summary())

# inspect the residuals of the model
tsplot(best_model.resid[12+1:], lags=30)
# clearly residuals are stationary


def plotSARIMA(series, model, n_steps, s, d, gamma, scaling_factor=1.96, plot_anomalies=False, plot_intervals=False):
    """
        Plots model vs predicted values
        
        series - dataset with timeseries
        model - fitted SARIMA model
        n_steps - number of steps to predict in the future
        
    """
    # adding model values
    data = series.copy()
    data.columns = ['actual']
    data['arima_model'] = model.fittedvalues
    # making a shift on s+d steps, because these values were unobserved by the model
    # due to the differentiating
    data['arima_model'][:s+d] = np.NaN
    
    # forecasting on n_steps forward 
    forecast = model.predict(start = data.shape[0], end = data.shape[0]+n_steps)
    forecast = data.arima_model.append(forecast)
    # calculate error, again having shifted on s+d steps from the beginning
    error = mean_absolute_percentage_error(data['actual'][s+d:], data['arima_model'][s+d:])

    plt.figure(figsize=(20, 10))
    plt.title("Mean Absolute Percentage Error: {0:.2f}%".format(error))
    plt.plot(forecast, color='g', label="model")
    plt.plot(data.actual, "b", label="actual")
    
    # create series with index of timeseries as actual series + forecasted series
    predictedDeviation = pd.Series(index=forecast.index)
    # set first value to be 0
    predictedDeviation[s+d+0] = 0
    
    # bond = fitted values +/- scalingfactor*deviation
    UpperBond = pd.Series(index=forecast.index)
    UpperBond[s+d+0] = (forecast.values[s+d+0] + scaling_factor * predictedDeviation[s+d+0])
    LowerBond = pd.Series(index=forecast.index)
    LowerBond[s+d+0] = (forecast.values[s+d+0] - scaling_factor * predictedDeviation[s+d+0])
    
    # for training dataset
    for i in range(s+d+1, len(series)):
        # Deviation is calculated according to Brutlag algorithm.
        predictedDeviation[i] = (gamma * np.abs(series.values[i] - forecast.values[i]) + (1-gamma)*predictedDeviation[i-1])
        UpperBond[i] = (forecast.values[i] + scaling_factor * predictedDeviation[i])
        LowerBond[i] = (forecast.values[i] - scaling_factor * predictedDeviation[i])
    
    # for forecasted dataset
    for i in range(len(series), len(forecast)):
        # increase uncertainity on each next step by multiplying deviation to 1.01
        predictedDeviation[i] = (predictedDeviation[i-1] * 1.01)
        UpperBond[i] = (forecast.values[i] + scaling_factor * predictedDeviation[i])
        LowerBond[i] = (forecast.values[i] - scaling_factor * predictedDeviation[i])
    
    if plot_anomalies:
        anomalies = pd.Series(index=series.index)
        anomalies[data.actual.values<LowerBond[:len(data)].values] = \
            data.actual[data.actual.values<LowerBond[:len(data)].values]
        anomalies[data.actual.values>UpperBond[:len(data)]] = \
            data.actual.values[data.actual.values>UpperBond[:len(data)]]
        plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    if plot_intervals:
        plt.plot(UpperBond, "r--", alpha=0.5, label = "Up/Low confidence")
        plt.plot(LowerBond, "r--", alpha=0.5)
        plt.fill_between(x=forecast.index, y1=UpperBond, 
                         y2=LowerBond, alpha=0.2, color = "grey")    
        
    plt.vlines(series.index[-1], ymin=min(LowerBond[s+d:]), ymax=max(UpperBond[s+d:]), linestyles='dashed')
    plt.axvspan(series.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc="best", fontsize=13);
    
plotSARIMA(data[['diet']], best_model, 25, s, d, gamma=0.02, plot_anomalies=True, plot_intervals=True)

#######################################################################
################### non timeseries methods ##########################
#########################################################################
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Creating a copy of the initial dataframe to make various transformations 
data = diet.copy()
data.columns = ["y"]

# Adding the lag of the target variable from 6 steps back up to 12
for i in range(6, 24):
    data["lag_{}".format(i)] = data.y.shift(i)
    
# for time-series cross-validation set 5 folds 
tscv = TimeSeriesSplit(n_splits=5)

def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test

def plotModelResults(model, X_train=X_train, X_test=X_test, plot_intervals=False, plot_anomalies=False):
    """
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_absolute_error")
        mae = cv.mean() * (-1)
        deviation = cv.std()
        
        scale = 1.96
        lower = prediction - (mae + scale * deviation)
        upper = prediction + (mae + scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(prediction, y_test)
    plt.title("Mean absolute percentage error {0:.2f}%".format(error))
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    
    
y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

# reserve 30% of data for testing
X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

# machine learning in two lines
lr = LinearRegression()
lr.fit(X_train, y_train)

plotModelResults(lr, X_train=X_train, X_test=X_test, plot_intervals=True)
plotCoefficients(lr)

# create few more features
#data["hour"] = data.index.hour
data["weekday"] = data.index.weekday
data['is_weekend'] = data.weekday.isin([5,6])*1

scaler = StandardScaler()

y = data.dropna().y
X = data.dropna().drop(['y'], axis=1)

X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

plotModelResults(lr, X_train=X_train_scaled, X_test=X_test_scaled, plot_intervals=True)
plotCoefficients(lr)



