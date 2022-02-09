######################################################################
# Python codes for Time Series Analysis Lecture 
#
# Data Mining Course, Fall 2021
# School Of Information Technology
# Halmstad University
#
# Hadi Fanaee, Ph.D., Assistant Professor
# hadi.fanaee@hh.se
# www.fanaee.com

######################################################################


#**********************************************************************
#Slide-28: Seasonal-Trend Decomposition
#**********************************************************************
from pandas import read_csv
from statsmodels.tsa.seasonal import seasonal_decompose
df  = read_csv('airline-passengers.csv',  header=0)
df.Month = pd.to_datetime(df.Month)
df = df.set_index('Month')
result = seasonal_decompose(df, model='multiplicative')
result.plot()

#**********************************************************************
#Slide-30: Augmented Dickey-Fuller Test for Non-stationary/stationary check
#**********************************************************************
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
df  = read_csv('airline-passengers.csv', header=0, index_col=0)
result = adfuller(df.values)
print('p-value: %f' % result[1])

#**********************************************************************
#Slide-32: Logarithmic/Square root/Box-Cox transform for Fixing Non-constant Variance
#**********************************************************************
from pandas import read_csv
from numpy import log
from numpy import sqrt
from scipy.stats import boxcox
import matplotlib.pyplot as plt
series = read_csv('airline-passengers.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
series_log = log(series)
series_sqr = sqrt(series)
series_boxcox, lam= boxcox(series)
print('Lambda: %f' % lam)
plt.plot(series_log)
plt.plot(series_sqr)
plt.plot(series_boxcox)
plt.show()

#**********************************************************************
#Slide-33: Trend Removal with differencing
#**********************************************************************
from pandas import datetime
from matplotlib import pyplot
series = read_csv('airline-passengers.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
diff = series.diff()
pyplot.plot(series.values)
pyplot.plot(diff.values)

#**********************************************************************
#Slide-34: Fixing Non-constant Variance + Trend Removal
#**********************************************************************
from pandas import read_csv
from pandas import Series
import pandas as pd
from scipy.stats import boxcox
series = read_csv('airline-passengers.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
series_boxcox, lam= boxcox(series)
series_boxcox=pd.DataFrame(series_boxcox)
series_boxcox_diff = series_boxcox.diff()
plt.plot(series)
plt.plot(series_boxcox_diff)


#**********************************************************************
#Slide-43: Exponential Smoothing 
# - Simple Exponential Smoothing (Brown,1956)
# - Double Exponential Smoothing (Holt, 1957)
# - Triple Exponential Smoothing (Holt-Winter, 1960)
#**********************************************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
df = pd.read_csv('airline-passengers.csv', parse_dates=['Month'],index_col='Month')
df.index.freq = 'MS'
ts=df.iloc[:, 0]
es1 = SimpleExpSmoothing(ts).fit(smoothing_level=0.2)
ts_es1 = es1.predict(start=ts.index[0], end=ts.index[-1])
es2 = ExponentialSmoothing(ts, trend='add').fit(smoothing_level=0.2,smoothing_trend=0.2)
ts_es2 = es2.predict(start=ts.index[0], end=ts.index[-1])
es3 = ExponentialSmoothing(ts, trend='add',seasonal='mul', seasonal_periods=12).fit(smoothing_level=0.2,smoothing_trend=0.2,smoothing_seasonal=0.2)
ts_es3 = es3.predict(start=ts.index[0], end=ts.index[-1])

#**********************************************************************
#Slide-46: Simulation of Autoregressive Time Series
#**********************************************************************

import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
np.random.seed(12345)
arparams = np.array([.75, -.25])
ar = np.r_[1, -arparams]
arma_process = sm.tsa.ArmaProcess(ar,ma=None)
x = arma_process.generate_sample(100)
plt.plot(x)


#**********************************************************************
#Slide-52: Fitting an Autoregressive Model
#**********************************************************************

import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
train, test = x[1:len(x)-7], x[len(x)-7:]
model = AutoReg(train, lags=2)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
x_ar = model_fit.predict(start=0, end=len(x), dynamic=False)
plt.plot(x[2:len(x)],label='Actual')
plt.plot(x_ar, color='red',label='AR(2) Model')
plt.legend(loc='best')


#**********************************************************************
#Slide-62: Fitting a Moving Average Model
#**********************************************************************

from pandas import read_csv
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df = read_csv('HalmstadTempM.csv', header=0, parse_dates=[0], index_col=0)
dftrain, dftest = df[1:len(df)-7], df[len(df)-7:]
model = ARMA(dftrain, order=(0, 5)) #MA(5)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
predictions = model_fit.predict(start=len(dftrain), end=len(dftrain)+len(dftest)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], dftest.values[i]))
rmse = sqrt(mean_squared_error(dftest.values, predictions))
print('Test RMSE: %.3f' % rmse)
plt.plot(dftest,label='Actual')
plt.plot(predictions, color='red',label='MA(1) Forecast')
plt.legend(loc='best')
plt.show()


#**********************************************************************
#Slide-66: Auto-SARIMA
# - Seasonal Auto Regressive Integrated Moving Average (SARIMA)
# - Auto Regressive Integrated Moving Average (ARIMA)
# - Auto Regressive Moving Average (ARMA)
#**********************************************************************

from pandas import read_csv
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import pandas as pd
df = read_csv('airline-passengers.csv', header=0, parse_dates=[0], index_col=0)
train, test = df[1:len(df)-7], df[len(df)-7:]
# SARIMA with automatic hyperparameter setting
model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True, d=1, D=1, 
                           trace=True,  error_action='ignore’,  
                            suppress_warnings=True, stepwise=True)
model.fit(train)
forecast = model.predict(n_periods=len(test))
plt.plot(test.values,label='Actual')
plt.plot(forecast, color='red',label='SARIMA'+str(model.order)+str(model.seasonal_order))
plt.legend(loc='best')



#**********************************************************************
#Slide-73: ACF/PACF Plots
#**********************************************************************
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)
arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag
arma_process = sm.tsa.ArmaProcess(ar, ma)
y = arma_process.generate_sample(250)
sm.graphics.tsa.plot_acf(y, lags=20)
sm.graphics.tsa.plot_pacf(y, lags=20)
plt.show()

#**********************************************************************
#Slide-103: Hidden Markov Model
#**********************************************************************

import numpy as np
from hmmlearn import hmm
states = ["45", "50", "55"] 
observations = ["200", "225","250","275","300"]
start_probability = np.array([0.1, 0.8, 0.1])
transition_probability = np.array([
  [0.6, 0.3, 0.1],
  [0.1, 0.7, 0.2],
  [0.1, 0.3, 0.6]
])
emission_probability = np.array([
  [0.2, 0.5, 0.2, 0.05, 0.05],
  [0.05, 0.15, 0.6, 0.15, 0.05],
  [0.05, 0.15, 0.2, 0.5, 0.1]
])

n_states = len(states)
n_observations = len(observations)
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob=start_probability
model.transmat=transition_probability
model.emissionprob=emission_probability
observed_dx = np.array([[225, 225, 225, 250, 275, 300]]).T
model = model.fit(observed_dx)
hidden_states = model.predict(observed_dx) 
print("Possible Velocity:", ", ".join(map(lambda x: states[x], hidden_states)))

#**********************************************************************
#Slide-106: Feed-forward Neural Networks  for Time Series Forecasting
#**********************************************************************

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps_in, n_steps_out = 3, 2
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=n_steps_in))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=2000, verbose=0)
model.summary()
weights = model.get_weights() 
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)
print(yhat)


#**********************************************************************
#Slide-116: 1D Convolutional Neural Networks for Time Series Forecasting
#**********************************************************************

from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps_in, n_steps_out = 3, 2
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps_in,n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.summary()
weights = model.get_weights() 
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in))
yhat = model.predict(x_input, verbose=0)
print(yhat)


#**********************************************************************
#Slide-123: Long Short-Term Memory (LSTM) for Time Series Forecasting
#**********************************************************************

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps_in, n_steps_out = 3, 2
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in,
n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)
weights = model.get_weights() 
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#**********************************************************************
#Slide-153: Time Series Mining with Matrix Profile (Motifs)
#**********************************************************************


from matplotlib import pyplot as plt
import numpy as np
import matrixprofile as mp
ecg = mp.datasets.load('ecg-heartbeat-av')
ts = ecg['data']
window_size = 150
profile = mp.compute(ts, windows=window_size)
mprofile = mp.discover.motifs(profile, k=1)
figures = mp.visualize(mprofile)

#**********************************************************************
#Slide-153: Time Series Mining with Matrix Profile (Discords)
#**********************************************************************

from matplotlib import pyplot as plt
import numpy as np
import matrixprofile as mp
import math 
ecg = mp.datasets.load('ecg-heartbeat-av')
ts = ecg['data']
window_size = 150
profile = mp.compute(ts, windows=window_size)
dprofile = mp.discover.discords(profile, k=1, exclusion_zone=window_size*7)
mp_adjusted = np.append(dprofile['mp'], np.zeros(dprofile['w'] - 1) + np.nan)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=(20,7))
ax.plot(np.arange(len(dprofile['data']['ts'])), dprofile['data']['ts'])
for discord in profile['discords']:
    x = np.arange(discord, discord + profile['w'])
    y = profile['data']['ts'][discord:discord + profile['w']]
    ax.plot(x, y, c='r')
plt.show()

#**********************************************************************
#Slide-158: Fourier Transform 
#**********************************************************************

import numpy as np
import matplotlib.pyplot as plotter
samplingFrequency=100
time = np.arange(0, 1, 1/samplingFrequency);
sine1 = np.sin(2*np.pi*2*time)
sine2 = np.sin(2*np.pi*7*time)
ts = sine1 + sine2
plt.plot(time,ts, label='Time Series' )
plt.plot(time,amplitude1, label='Sine 2 Hz')
plt.plot(time,amplitude2, label='Sine 7 Hz')
plt.ylabel('Amplitude')
plt.xlabel('Time')
plt.legend(loc='best')
plt.show()
fourierTransform = np.fft.fft(ts)/len(ts)
fourierTransform = fourierTransform[range(int(len(ts)/2))] 
tpCount = len(ts)
values  = np.arange(int(tpCount/2))
timePeriod  = tpCount/samplingFrequency
frequencies = values/timePeriod
plt.plot(frequencies, abs(fourierTransform))
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.show()

#**********************************************************************
#Slide-163: Wavelets
#**********************************************************************

import numpy as np
import pywt
import matplotlib.pyplot as plt
df= read_csv('airline-passengers.csv', header=0, index_col=0)
x=df.iloc[:,0]
ts = x.values
thresh = 0.30*np.nanmax(ts)
coeff = pywt.wavedec(ts, "db4", level=1)
coeff[1:] = (pywt.threshold(i, value=thresh) for i in coeff[1:])
ts_reconstructed = pywt.waverec(coeff, "db4" )
plt.plot(ts, label='Raw Time Series')
plt.plot(ts_reconstructed, label='Wavelets Smoothing', linestyle='--')
plt.legend(loc='best')
plt.show()

#**********************************************************************
#Slide-163: Piecewise Aggregate Approximation (PAA)
#**********************************************************************
import numpy
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
df= read_csv('airline-passengers.csv', header=0, index_col=0)
df_Series=df.iloc[:,0]
dataset = df_Series.values
dataset=dataset.reshape(1, 144)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
dataset = scaler.fit_transform(dataset)
paa = PiecewiseAggregateApproximation(n_segments=12)
paa_dataset_inv = paa.inverse_transform(paa.fit_transform(dataset))
plt.plot(dataset[0].ravel(),label='Raw Time Series')
plt.plot(paa_dataset_inv[0].ravel(), label='PAA')
plt.legend(loc='best')
plt.show()

#**********************************************************************
#Slide-167: Symbolic Aggregate Approximation (SAX)
#**********************************************************************
import numpy
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import SymbolicAggregateApproximation
df= read_csv('airline-passengers.csv', header=0, index_col=0)
df_Series=df.iloc[:,0]
dataset = df_Series.values
dataset=dataset.reshape(1, 144)
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.) 
dataset = scaler.fit_transform(dataset)
sax = SymbolicAggregateApproximation(n_segments= 12,  alphabet_size_avg=3)
sax_dataset_inv = sax.inverse_transform(sax.fit_transform(dataset))
plt.plot(dataset[0].ravel(),label='Raw Time Series')
plt.plot(sax_dataset_inv[0].ravel(), label='SAX')
plt.legend(loc='best')
plt.show()


#**********************************************************************
#Slide-179: Missing Value Estimation (Imputation
#**********************************************************************

import pandas as pd
import numpy as np
df  = read_csv('airline-passengers.csv', header=0, index_col=0)
x=df.iloc[:,0]
x=x.astype('float64')
xt=x[4]
x[4]=np.nan
y=x.interpolate()
print ([xt, y[4]])


#**********************************************************************
#Slide-182: Range-based Normalization of Time Series
#**********************************************************************

from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
df= read_csv('airline-passengers.csv', header=0, index_col=0)
ts=df.iloc[:10, :]
values = ts.values
values = values.reshape((len(values), 1))
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
normalized = scaler.transform(values)
print (normalized)
inversed = scaler.inverse_transform(normalized)
print (inversed)


#**********************************************************************
#Slide-182: Standardization of Time Series
#**********************************************************************

from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from math import sqrt
df= read_csv('airline-passengers.csv', header=0, index_col=0)
ts=df.iloc[:10, :]
values = ts.values
values = values.reshape((len(values), 1))
scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, Std: %f' % (scaler.mean_, sqrt(scaler.var_)))
normalized = scaler.transform(values)
print (normalized)
inversed = scaler.inverse_transform(normalized)
print (inversed)

#**********************************************************************
#Slide-182: Resampling (Upsampling/Downsampling) of Time Series
#**********************************************************************


from pandas import read_csv
df = read_csv('airline-passengers.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)
df.resample("D").interpolate() #Upsampling
df.resample(“Y").mean() #Downsampling

#**********************************************************************
# References
#**********************************************************************
Nielsen, Aileen. Practical time series analysis: Prediction with statistics and machine learning. O'Reilly Media, 2019. [Applied Time Series Analysis with Python/R codes]
Brownlee, Jason. Deep learning for time series forecasting: predict the future with MLPs, CNNs and LSTMs in Python. Machine Learning Mastery, 2018. [Deep Learning for Time Series Forecasting]

