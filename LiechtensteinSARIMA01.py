''''estimating Liechtenstein Covid Dataset'''
# report with R is found here file:///C:/Users/c8451269/Desktop/Rezgar%20Project/report.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#import statsmodels.api as sm
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#import warnings
#warnings.filterwarnings('ignore')
#%matplotlib inline

# Reading and transforming the file
print("hello1")
epd = pd.read_csv('C:/Users/c8451269/PycharmProjects/CovidDataset/LiechtensteinSARIMA01/EPD.csv')

print(epd.head(5))
print("hello")

epd['date'] = pd.to_datetime(epd['Unnamed: 0'],infer_datetime_format=True) #convert from string to datetime
print(epd.head(5))
epd['activeCases'] = epd['confirmed'] - epd['deaths'] - epd['recovered']
print(epd.head(5))
print(epd['date'].head(5))

epd1 = epd['date'].max()-epd['date'].min()
print(epd1)


filt=(epd['date']>=pd.to_datetime('2021-01-01'))
print(epd.loc[filt])

epd.set_index('date', inplace=True) #consider date as index, not the numbers anymore (for the rows)
print(epd['2021-01':'2021-02']) #now that the dates are considered index we can do this and write it in this easy way
