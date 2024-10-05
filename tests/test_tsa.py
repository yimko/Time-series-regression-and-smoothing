from src import tsa

import pytest
import numpy as np
import statsmodels.tsa.arima_process as arimap
import statsmodels.tsa.arima.model as arimam


class TestClassSim():
    model = tsa.arima([0.1],0,[0.1],1)

    def test_sim(self):
        sim = self.model.sim(100)
        assert len(sim)==100

    def test_acf(self):
        acfs = self.model.acf(20)
        assert len(acfs)==20

    def test_acfma(self):
        model = tsa.arima([],0,[0.1],1)
        acfs = model.acf(20)
        assert acfs[2]==0

    def test_pacf(self):
        pacfs = self.model.pacf(20)
        assert len(pacfs)==20

    def test_pacfar(self):
        model = tsa.arima([0.1],0,[],1)
        pacfs = model.pacf(20)
        assert round(pacfs[2],3)==0

    def test_roots(self):
        arparams = np.array([0.1])
        maparams = np.array([0.1])
        ar = np.r_[1, -arparams]
        ma = np.r_[1, maparams]
        modelsm = arimap.ArmaProcess(ar,ma)
        rootsmar = np.array(modelsm.arroots)
        rootsmma = np.array(modelsm.maroots)

        roots = self.model.roots()

        assert rootsmar==roots["ar"] and rootsmma==roots["ma"]

    def test_forecast(self):
        data = self.model.sim(100)
        forecast, p = self.model.forecast(data,10)

        modelsm = arimam.ARIMA(data, order=(1,0,1))
        fit = modelsm.fit()
        forecastsm = fit.get_forecast(10)
        fcast = forecastsm.summary_frame(alpha=0.1)
        val = fcast["mean"]
        cil = fcast['mean_ci_lower']
        cih = fcast['mean_ci_upper']

        bools = [True if forecast[i] >= val[i]+cil[i] and forecast[i] <= val[i]+cih[i] else False for i in range(len(forecast))]

        assert np.any(bools)

class TestClassFit():
    m = tsa.arima([0.1],1,[0.1],1)
    data = m.sim(100)
    est = tsa.estimation(data,1,1,1)
    _,_,_,_,l = est.fit()

    msm = arimam.ARIMA(data, order=(1,1,1)).fit()

    def test_ar1(self):
        m = tsa.arima([0.1],0,[],1)
        est  = tsa.estimation(m.sim(100),1,0,0)
        fit, _,_,_,_ = est.fit()

        assert len(fit)==1

    def test_ma1(self):
        m = tsa.arima([],0,[0.1],1)
        est = tsa.estimation(m.sim(100),0,0,1)

        fit, _,_,_,_ = est.fit()

        assert len(fit)==1

    def test_arma11(self):
        m = tsa.arima([0.1],0,[0.1],1)
        est= tsa.estimation(m.sim(100),1,0,1)
        fit, _,_,_,_= est.fit()

        assert len(fit)==2

    def test_arima111(self):
        m = tsa.arima([0.1],1,[0.1],1)
        est = tsa.estimation(m.sim(100),1,1,1)
        fit, _,_,_,_ = est.fit()

        assert len(fit)==2

    def test_highar(self):
        m = tsa.arima([0.1],0,[],1)
        est = tsa.estimation(m.sim(100),10,0,0)
        fit, _,_,_,_ = est.fit()

        assert len(fit)==10

    def test_highma(self):
        m = tsa.arima([],0,[0.1],1)
        est = tsa.estimation(m.sim(100),0,0,10)
        fit, _,_,_,_= est.fit()

        assert len(fit)==10

    def test_qstat(self):
        est = tsa.estimation(self.data,1,1,1)
        fit, err,_,_,_ = est.fit()

        qstat = est.get_qstat(err)

        assert len(qstat)==20