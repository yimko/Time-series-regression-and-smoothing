# Summary
A collection of scripts to perform time series analysis via the Box-Jenkins method for non-seasonal ARIMA models. The scripts are divided into a regression module and time series analysis module. This project was mainly an implementation exercise to understand the theories and methods outlined in "Time series analysis and its applications" by Robert H. Shumway and David S. Stofer. Testing and comparisons were done against statsmodel.

Current functionality includes:
  - Regression via ordinary least squares
  - Smoothing via normal kernel
  - Smoothing via Savitzky–Golay filter
  - Simulating arbitrary ARIMA(p,d,q) models
  - Estimating sample ACFs, PACFs, autocovariances
  - Fitting an ARIMA(p,d,q) model on data
  - A script to query and obtain data from Yahoo Finance

## Dependencies
numpy, pandas, matplotlib, and scipy.

# Examples
**Import dependencies**
```python
#Analysis of stock data from Yahoo Finance
from src import DataQuery as DQ
from src import tsa
from src import Regression as reg

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import scipy.stats as stats
```
**Import data through data query module**
```python
#1 day interval since past year
dq = DQ.DataQuery("1d",datetime.now()-timedelta(days=365),datetime.now())
df = dq.fetch("msft") #microsoft
close = df["close"]
close = close - np.mean(close)
```
**Setup ARIMA estimation**
```python
#Estimate on close data with an ARIMA(1,0,0) model (e.g AR(1))
est = tsa.estimation(close,1,0,0)

# Fit the AR(1) model
fit, err, serr, var, l = est.fit()
model = tsa.arima(fit,0,[],var)
```
**Plot ACF AND PACFS**
```python
lag = 20
acfs = est.estacf(lag)
pacfs = est.estpacf(lag)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('ACF and PACF of close')
ax1.plot(np.arange(lag), acfs)
ax2.plot(np.arange(lag), pacfs)
plt.xticks(np.arange(0, lag, step=1))
```
**Forecast and plot m=10 points and the corresponding confidence interval**
```python
m = 10
train = close[:-m]
forecast, p = model.forecast(train,m)
ci = 1.96 * np.array(p)/np.sqrt(len(train))

fig, ax = plt.subplots()
x = np.arange(len(train)+1, len(train)+m)
ax.plot(x,forecast)
ax.fill_between(x, (forecast-ci), (forecast+ci), color='b', alpha=.1)
```
**Plot ACF of residuals, standardised and q statistic**
```python
lag = 20
acfs = est.estacf(lag,err)
qstat = est.get_qstat(err,H=lag)

fig, (ax1, ax2,ax3,ax4) = plt.subplots(4)
ax1.plot(serr)
ax2.plot(acfs)
ax3.plot(qstat)
stats.probplot(serr, dist="norm", plot=ax4)
```
**Get the AIC,AICc and BIC**
```python
print("AIC: ", est.get_AIC(l))
print("BIC: ", est.get_BIC(l))
print("AICc: ", est.get_AICc(l))
```
## Regression/smoothing models
```python
#Normal kernel smoothing
nk = reg.normal_kernel(np.arange(len(close)),close,0.5)
#SG filter
sg = reg.sgfilt(close, 50,1)

#Ordinary least squares linear regression
ols = reg.lm([np.arange(len(close))] , close)
fitls = ols.fit()

fig, ax1 = plt.subplots(1)
ax1.plot(close, label='original')
ax1.plot(nk, label='normal kernel')
ax1.plot(sg, label= 'savitzky golay')
ax1.plot(fitls[0]+fitls[1]*np.arange(len(close)), label="ols")
ax1.legend()
```