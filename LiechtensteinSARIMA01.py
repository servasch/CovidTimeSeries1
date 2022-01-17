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

plt.figure(figsize=(22,6))
#sns.lineplot(x=epd.index, y=epd['activeCases'])
plt.plot(epd['activeCases'])
plt.title('active cases among the time')
plt.savefig("C:/Users/c8451269/Desktop/SARIMA01.png")
#plt.show()


# making a pivot table (which shows the results monthly or yearly) might be needed, look at kaggle SARIMA tutorial
# here we just make a test
epd['day'] = epd.index.day
epd['month'] = epd.index.month
pivot = pd.pivot_table(epd, values='activeCases', index='day', columns='month', aggfunc='mean')
pivot.plot(figsize=(20,6))
plt.title('monthly active cases')
plt.xlabel('days')
plt.ylabel('active cases')
#plt.xticks([x for x in range(1,30)])
plt.legend()
plt.savefig("C:/Users/c8451269/Desktop/SARIMA02.png")

plt.figure(figsize=(22,6))
#sns.lineplot(x=epd.index, y=epd['activeCases'])
plt.plot(epd['activeCases'])
plt.title('active cases among the time')
plt.savefig("C:/Users/c8451269/Desktop/SARIMA01.png")
#plt.show()
plt.clf()
# Plot the data using bar() method

X = list(epd.iloc[:, 0])
Y = list(epd.iloc[:, 1])

#plt.bar(epd['confirmed'], height=5, color='g')
plt.bar(X, Y, color='g')
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
plt.title("active Cases 02")
plt.xlabel("time")
plt.ylabel("Number")
plt.savefig("C:/Users/c8451269/Desktop/SARIMA02.png")

# making a pivot table (which shows the results monthly or yearly) might be needed, look at kaggle SARIMA tutorial
# here we skip it (look at https://www.kaggle.com/leandrovrabelo/climate-change-forecast-sarima-model)
