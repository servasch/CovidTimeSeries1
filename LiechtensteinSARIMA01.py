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

# making a pivot table (which shows the results monthly or yearly) might be needed, look at kaggle SARIMA tutorial
# here we skip it (look at https://www.kaggle.com/leandrovrabelo/climate-change-forecast-sarima-model)

plt.clf()
#plot mapping

gmaps.configure(api_key='AI...')

marker_locations = [
    (9.501531, 47.214083)
]

figMap = gmaps.figure()
markers = gmaps.marker_layer(marker_locations)
figMap.add_layer(markers)

#plot restriction policy (skiped)
#plot Spearman correlation (skiped)

# Mann-Kendall Trend Test

print(mk.original_test(epd['activeCases']))
pValue=mk.original_test(epd['activeCases'])
#Mann_Kendall_Test(trend='no trend', h=False, p=0.422586268671707,                   z=0.80194241623, Tau=0.147058823529, s=20.0,                   var_s=561.33333333, slope=0.0384615384615, intercept=27.692307692)
print(pValue[2])
zt=epd['activeCases']
print(zt[3], zt[4], zt[5])

if pValue[2]>0.05:
    zt=np.diff(zt)
    print(zt[4])
    print('hello5')
else:
    zt=np.diff(zt,n=12)

# normality tests and plots
fig, ax3=plt.subplots()

ax3.hist(zt, bins='sturges', density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='black')
# Density Plot and Histogram of all arrival delays
#sns.displot(zt, hist=False, kde=True)
ax3.set_xlabel('Prevelance')
ax3.set_ylabel('density', color='black')
plt.savefig("C:/Users/c8451269/Desktop/SARIMA03.png")


# qq plot
#fig, ax4=plt.subplots()
sm.qqplot(zt, line ='r')
ax3.set_xlabel('theoretical quantiles')
ax3.set_ylabel('sample quantiles', color='dimgray')
#perform Shapiro-Wilk test
pValue=shapiro(zt)
print('p.Value from shapira test is', pValue[1])
plt.savefig("C:/Users/c8451269/Desktop/SARIMA04.png")

#if(p.value<0.05)
#boxcox parameters optimization
