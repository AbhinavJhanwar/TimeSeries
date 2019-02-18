'''
Created on Dec 26, 2017

@author: abhinav.jhanwar
'''

################## TIMSERIES PANDAS ################

import pandas as pd
import numpy as np

# creating time series index day wise
myTseries = pd.date_range('2017-08-01', '2017-10-31')
#print(myTseries)
# print size of time series
#print(myTseries.size)

# creating time series index hours wise
myTseriesHrs = pd.date_range('2017-08-01', '2017-10-31', freq = 'H')
#print(myTseriesHrs)

# creating pandas series with timeseries index
myTseriesSeq = pd.Series(np.random.normal(150, 10, len(myTseries)), index = myTseries)
#print(myTseriesSeq)

myTseriesSeqHrs = pd.Series(np.random.normal(150, 10, len(myTseriesHrs)), index = myTseriesHrs)
#print(myTseriesSeqHrs.head())
#print(myTseriesSeqHrs.tail())
#print(myTseriesSeqHrs.size)

# modify time series to have frequency minute wise
myTseriesSeqHrs = myTseriesSeqHrs.resample('T').sum()
# using mean value instead of sum
myTseriesSeqHrs = myTseriesSeqHrs.resample('T').mean()
# every 2 minute
myTseriesSeqHrs = myTseriesSeqHrs.resample('2T').mean()

# hour wise
myTseriesSeqHrs = myTseriesSeqHrs.resample('H').mean()

# day wise
myTseriesSeqHrs = myTseriesSeqHrs.resample('D').mean()

# day wise
myTseriesSeqHrs = myTseriesSeqHrs.resample('D').mean()

# week wise
myTseriesSeqHrs = myTseriesSeqHrs.resample('W').mean()

# modify time series to have frequency month wise
myTseriesSeqHrs = myTseriesSeqHrs.resample('M').mean()
#print(myTseriesSeqHrs)


# creating time deltas
a = pd.Timedelta('1 days 4 hours 15 min 2 s 8 ms')
#print(a)
# another way of creating time deltas
a = pd.Timedelta(days=1, hours=4, minutes=15, seconds=8, milliseconds=8)
#print(a)
# another way of creating time deltas
a = pd.to_timedelta(np.arange(4), unit='s')
#print(a)

# adding timedeltas
ts1 = pd.Series(pd.date_range('2017-10-01', periods=5, freq='D'))
#print(ts1)
ts2 = pd.Timedelta(hours=4, minutes=15)
#print(ts2)
#print(ts1 + ts2)



