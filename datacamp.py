import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# read data
df = pd.read_csv("multiTimeline.csv", skiprows=2)

# check data info
print(df.info())

# rename columns to remove white spaces
df.columns = ['month', 'diet', 'gym', 'finance']

# convert month data column to datetime type instead of object type
df.month = pd.to_datetime(df.month)

# set index of data to month
df.set_index('month', inplace=True)

# plot data
df.plot(figsize=(15,10), linewidth=3, fontsize=18)
plt.xlabel('Year', fontsize=20)
'''
 Numbers represent search interest relative to the highest point on the chart
 for the given region and time. A value of 100 is the peak popularity for the 
 term. A value of 50 means that the term is half as popular. Likewise a score 
 of 0 means the term was less than 1% as popular as the peak.
'''

# plot diet
df[['diet']].plot(figsize=(15,10), linewidth=3, fontsize=18)
plt.xlabel('Year', fontsize=20)
'''
the first thing to notice is that there is seasonality: each January, 
there's a big jump. Also, there seems to be a trend: it seems to go slightly 
up, then down, back up and then back down. In other words, it looks like 
there are trends and seasonal components to these time series.
'''

# CHECKING TRENDS
''' 
this could be done by removing seasonality or we say taking an average
every 12 months i.e. n-12 to n-1
'''
diet = df[['diet']]
gym = df[['gym']]
finance = df[['finance']]
# take rolling average
diet.rolling(12).mean().plot(figsize=(15,10), linewidth=3, fontsize=18)
plt.xlabel('Year', fontsize=20)
'''
here 12 in rolling signifies that we are checking trends every 12 months
i.e. it will take the average of values based on 12 months before and 12 
months after the current value. we can also say we have removed seasonality
which occures within a year.
'''
# comparing two columns for trend
# concatenating diet and gym along column and plotting
df_rm = pd.concat([diet.rolling(12).mean(), gym.rolling(12).mean()], axis=1)
df_rm.plot(figsize=(15,10), linewidth=3, fontsize=18)
plt.xlabel('Year', fontsize=20)
'''
here it is clear that gym has a trend of continuous increase
'''

# CHECKING SEASONALITY
# this could be explored by removing trend
''' 
using first order differencing i.e. difference between 1 data point before 
and 1 after
'''
finance.diff().plot(figsize=(15,10), linewidth=3, fontsize=18)
plt.xlabel('Year', fontsize=20)

# checking correlation with trend and seasonality if present
df.corr()

# plot correlation after removing trend
# or correlation in seasonal component if present
df.diff().plot(figsize=(15,10), linewidth=3, fontsize=20)
plt.xlabel('Year', fontsize=20);
df.diff().corr()

''' 
checking autocorrelation i.e. whether the series is repeating itself after
a particular duration or not.
'''
# for this data autocorrelation should be there in every 12 months for diet
pd.plotting.autocorrelation_plot(diet);
# here lag is the duration before and after the particular time
