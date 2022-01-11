''''estimating Liechtenstein Covid Dataset'''
# report with R is found here file:///C:/Users/c8451269/Desktop/Rezgar%20Project/report.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Reading and transforming the file
epd = pd.read_csv('C:/Users/c8451269/PycharmProjects/CovidDataset/LiechtensteinSARIMA01/EPD.csv')

activeCases = epd['confirmed'] - epd['deaths'] - epd['recovered']


rio.columns = ['Date','Temp']
rio['Date'] = pd.to_datetime(rio['Date'])
rio.reset_index(drop=True, inplace=True)
rio.set_index('Date', inplace=True)

#I'm going to consider the temperature just from 1900 until the end of 2012
rio = rio.loc['1900':'2013-01-01']
rio = rio.asfreq('M', method='bfill')
rio.head()