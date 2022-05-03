
''''estimating Liechtenstein Covid Dataset'''
# report with R is found here file:///C:/Users/c8451269/Desktop/Rezgar%20Project/report.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from math import sqrt
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
import matplotlib.ticker as ticker
import gmaps
import pymannkendall as mk
from scipy.stats import shapiro
from scipy.stats import boxcox
#from sklearn.preprocessing import PowerTransformer #for boxcox another way

# create a differenced series
def difference(dataset, interval=12):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff


# SARIMA Function
def best_sarima_model(train_data,p,q,P,Q,d=1,D=1,s=12):
    best_model_aic = np.Inf
    best_model_bic = np.Inf
    best_model_hqic = np.Inf
    best_model_order = (0,0,0)
    models = []
    for p_ in p:
        for q_ in q:
            for P_ in P:
                for Q_ in Q:
                    try:
                        no_of_lower_metrics = 0
                        model = SARIMAX(endog=train_data,order=(p_,d,q_), seasonal_order=(P_,D,Q_,s),
                                        enforce_invertibility=False).fit()
                        models.append(model)
                        if model.aic <= best_model_aic: no_of_lower_metrics+=1
                        if model.bic <= best_model_bic: no_of_lower_metrics+=1
                        if model.hqic <= best_model_hqic:no_of_lower_metrics+=1
                        if no_of_lower_metrics >= 2:
                            best_model_aic = np.round(model.aic,0)
                            best_model_bic = np.round(model.bic,0)
                            best_model_hqic = np.round(model.hqic,0)
                            best_model_order = (p_,d,q_,P_,D,Q_,s)
                            current_best_model = model
                            models.append(model)
                            print("Best model so far: SARIMA" +  str(best_model_order) +
                                  " AIC:{} BIC:{} HQIC:{}".format(best_model_aic,best_model_bic,best_model_hqic)+
                                  " resid:{}".format(np.round(np.exp(current_best_model.resid).mean(),3)))

                    except:
                        pass

    print('\n')
    print(current_best_model.summary())
    return current_best_model, models

#Shapiro test
def shapiro_normality_test(data):
    p_value = shapiro(data)[1]
    if p_value >= 0.05:
        print("Data follows normal distribution: X~N"+ str((np.round(np.mean(data),3), np.round(np.std(data),3))))
        print("Shapiro test p_value={}".format(np.round(p_value,3)))
    else:
        print("Data failed shapiro normality test with p_value={}".format(np.round(p_value,3)))


# MAPE Error
def mean_abs_pct_error(actual_values, forecast_values):
    err=0
    for i in range(len(forecast_values)):
        err += np.abs(actual_values[i] - forecast_values[i])/actual_values[i]
    return err * 100/len(forecast_values)



def main() -> None:
    # Reading and transforming the file
    print("hello1")
    epd = pd.read_csv('C:/Users/c8451269/PycharmProjects/CovidDataset/LiechtensteinSARIMA01/EPD.csv') #epidemiological data
    gmd = pd.read_csv('C:/Users/c8451269/PycharmProjects/CovidDataset/LiechtensteinSARIMA01/GMD.csv') #google mobility data
    sar = pd.read_csv('C:/Users/c8451269/PycharmProjects/CovidDataset/LiechtensteinSARIMA01/SAR.csv') #wastewater data


    print(epd.head(5))
    print('This is GMD')
    print(gmd.head(5))
    print('This is SAR')
    print(sar.head(5))

    epd['date'] = pd.to_datetime(epd['Unnamed: 0'],infer_datetime_format=True) #convert from string to datetime
    gmd['date'] = pd.to_datetime(gmd['Unnamed: 0'],infer_datetime_format=True) #convert from string to datetime
    sar['date'] = pd.to_datetime(sar['Unnamed: 0'],infer_datetime_format=True) #convert from string to datetime
    print(epd.head(5))
    epd['activeCases'] = epd['confirmed'] - epd['deaths'] - epd['recovered']
    print(epd.head(5))
    print(epd['date'].head(5))

    epd1 = epd['date'].max()-epd['date'].min()
    print(epd1)

    epd.set_index('date', inplace=True) #consider date as index, not the numbers anymore (for the rows)
    gmd.set_index('date', inplace=True)
    sar.set_index('date', inplace=True)
    print(epd['2021-01':'2021-02']) #now that the dates are considered index we can do this and write it in this easy way

    plt.figure(figsize=(22,10))
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
    ax2.plot(X, Y4, color='red', label='wastewater data')
    ax2.set_ylabel('SarsCov2 Titer (cpies/ml)', color='red')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor='red')

    ax2.plot(X, Y2, color='green', label='vaccinated people')
    ax2.plot(X, Y3, color='darkgreen', label='fully-vaccinated people')

    legend = ax2.legend(loc='upper center', shadow=True, fontsize='x-small')
    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('silver')

    tick_spacing = 5
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA02.png")

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

    # create a differenced series

    zt=difference(zt, interval=12)
    zt = np.array(zt) #convert list to array
    print('zt is', zt)

    # normality tests and plots
    fig, ax3=plt.subplots()

    ax3.hist(zt, bins='sturges', density=True, alpha=0.5, histtype='stepfilled', color='steelblue', edgecolor='black')
    sns.distplot(zt, hist=False, kde=True)
    # Density Plot and Histogram of all arrival delays
    #sns.displot(zt, hist=False, kde=True)
    ax3.set_xlabel('Prevelance')
    ax3.set_ylabel('density', color='black')
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA03.png")


    # qq plot
    fig, ax4=plt.subplots()
    sm.qqplot(zt, line ='r')
    ax4.set_xlabel('theoretical quantiles')
    ax4.set_ylabel('sample quantiles', color='dimgray')
    #perform Shapiro-Wilk test
    pValue=shapiro(zt)
    print('p.Value from shapira test is', pValue[1])
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA04.png")

    #if(p.value<0.05)
    #boxcox parameters optimization

    plt.clf()
    print('the minimum of the data is', zt.min())
    transedZt1=zt-zt.min()+1
    bxc=boxcox(transedZt1)
    bestLambda=bxc[1]
    print('the lambda is', bestLambda)
    bxc=bxc[0]

    fig, ax5=plt.subplots()
    ax5.hist(bxc, bins='sturges', density=True, alpha=0.5, histtype='stepfilled')
    sns.distplot(bxc, hist=False, kde=True)
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA05.png")

    plt.clf()
    print(bxc)
    sm.qqplot(bxc, line ='r')
    #plot the distribution of the transformed data values
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA06.png")

    plt.clf()
    acf_plot=plot_acf(bxc, lags=12)
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA07.png")

    plt.clf()
    pacf_plot=plot_pacf(bxc, lags=12)
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA08.png")



    # some renewing the data is left here

#   first strategy
    zt1=bxc

#    modelChoices = np.array([(p, q, P, Q) for p in range(3) for q in range(3) for P in range(3) for Q in range(3)])

    calibrationPeriod=zt1[0:int(0.8 * zt1.shape[0])]
    validationPeriod=zt1[int(0.8 * zt1.shape[0]):]
    print('the lengths of zt1, calibrationPeriod and validationPeriod are', len(zt1), ',', len(calibrationPeriod), 'and', len(validationPeriod), 'respectively')

    best_model, models = best_sarima_model(train_data=calibrationPeriod, p=range(3), q=range(3), P=range(3), Q=range(3))
#    best_model.plot_diagnostics(figsize=(15,12));
    plt.clf()
    plt.scatter(x=calibrationPeriod[1:], y=best_model.resid[1:]);
    plt.title('Residuals plot')
    plt.xlabel('Log transformed data')
    plt.ylabel('Residuals')
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA09.png")

#    shapiro_normality_test(best_model.resid[1:].drop(index=pd.datetime(2020, 1, 24)))

    plt.clf()
    plot_acf(best_model.resid[1:], lags=12);
    plt.savefig("C:/Users/c8451269/Desktop/SARIMA10.png")

    print('shape of calibrationPeriod is', calibrationPeriod.shape[0])
    forecast = best_model.predict(start=calibrationPeriod.shape[0], end=calibrationPeriod.shape[0]+len(validationPeriod)-1, dynamic=True, typ='levels')
    print('the forecast is', forecast)
    print("MAPE:{}%".format(np.round(mean_abs_pct_error(validationPeriod, forecast), 2)))
    print("MAE:{}".format(np.round(mean_absolute_error(validationPeriod, forecast), 2)))


if __name__ == "__main__":
    main()
