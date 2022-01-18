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


# Plot multiple graphs in one fig
fig, ax1=plt.subplots()

X = list(epd.iloc[:, 0])
Y1 = epd['activeCases']
Y2 = epd['people_vaccinated']
Y3 = epd['people_fully_vaccinated']
Y4 = list(sar.iloc[:, 3])

ax1.bar(X, Y1, color='silver')

ax1.set_xlabel('time (d)')
ax1.set_ylabel('active cases', color='dimgray')
ax1.tick_params(axis='y', labelcolor='dimgray')
ax1.tick_params(axis='x',rotation = 45) # Rotates X-Axis Ticks by 45-degrees



ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(X, Y4, color='red')
ax2.set_ylabel('SarsCov2 Titer (cpies/ml)', color='red')  # we already handled the x-label with ax1
ax2.tick_params(axis='y', labelcolor='red')

ax2.plot(X, Y2, color='green')
ax2.plot(X, Y3, color='darkgreen')


tick_spacing = 5
ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("C:/Users/c8451269/Desktop/SARIMA02.png")

# making a pivot table (which shows the results monthly or yearly) might be needed, look at kaggle SARIMA tutorial
# here we skip it (look at https://www.kaggle.com/leandrovrabelo/climate-change-forecast-sarima-model)

# making a pivot table (which shows the results monthly or yearly) might be needed, look at kaggle SARIMA tutorial
# here we skip it (look at https://www.kaggle.com/leandrovrabelo/climate-change-forecast-sarima-model)
