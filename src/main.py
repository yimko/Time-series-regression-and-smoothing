#CLI program
import DataQuery as DQ
import tsa
import Regression as reg

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import scipy.stats as stats

from argparse import *
import sys

def data_helper(**kwargs):
    f = kwargs["f"]
    t = kwargs["t"]
    typ = kwargs["type"]
    i = kwargs["int"]
    data = None

    if not (f or t):
        print("No valid data given!")
        return
    
    if f:
        data = np.loadtxt(data)
    else:
        dq = DQ.DataQuery(i,kwargs["start"],kwargs["end"])
        data = dq.fetch(t)[typ]
        mean = np.mean(data)
    
    return data,mean

#smoothing commands
def smooth_sg(args):
    data,mean = data_helper(f=args.f,t=args.t,type=args.type,int=args.int,start=args.start,end=args.end)
    data -= mean

    ws = args.winsize
    d = args.degree

    sg = reg.sgfilt(data, ws,d)+mean
    fig, ax1 = plt.subplots(1)
    ax1.plot(data+mean, label='original')
    ax1.plot(sg, label= 'savitzky golay')
    ax1.legend()
    plt.show()

def smooth_norm(args):
    data,mean = data_helper(f=args.f,t=args.t,type=args.type,int=args.int,start=args.start,end=args.end)
    data -= mean

    nk = reg.normal_kernel(np.arange(len(data)),data,args.bandwidth)+mean
    fig, ax1 = plt.subplots(1)
    ax1.plot(data+mean, label='original')
    ax1.plot(nk, label= 'normal kernel')
    ax1.legend()
    plt.show()

def smooth_ols(args):
    data,mean = data_helper(f=args.f,t=args.t,type=args.type,int=args.int,start=args.start,end=args.end)

    ols = reg.lm([np.arange(len(data))] , data)
    fitls = ols.fit()

    fig, ax1 = plt.subplots(1)
    ax1.plot(data, label='original')
    ax1.plot(fitls[0]+fitls[1]*np.arange(len(data)), label="ols")
    ax1.legend()
    plt.show()

#Sim commands
def sim_helper(**kwargs):
    ar = kwargs["ar"]
    ma = kwargs["ma"]
    d = kwargs["d"]
    var = kwargs["var"]

    model = tsa.arima(ar,d,ma,var)
    return model

def sim_sample(args):
    model = sim_helper(
        ar = args.ar,
        ma = args.ma,
        d = args.d,
        var = args.var
    )
    plt.plot(model.sim(args.n))
    plt.show()

def sim_acf(args):
    model = sim_helper(
        ar = args.ar,
        ma = args.ma,
        d = args.d,
        var = args.var
    )

    plt.plot(model.acf(args.lag))
    plt.show()

def sim_pacf(args):
    model = sim_helper(
        ar = args.ar,
        ma = args.ma,
        d = args.d,
        var = args.var
    )

    plt.plot(model.pacf(args.lag))
    plt.show()

#estimation commands
def tsa_helper(data,**kwargs):
    ar = kwargs["ar"]
    ma = kwargs["ma"]
    d = kwargs["d"]

    est = tsa.estimation(data,ar,d,ma)

    return est

def tsa_corr(args):
    data,_ = data_helper(f=args.f,t=args.t,type=args.type,int=args.int,start=args.start,end=args.end)

    est = tsa_helper(data,ar=args.ar,d=args.d,ma=args.ma)

    lag = args.lag
    acfs = est.estacf(lag)
    pacfs = est.estpacf(lag)

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(np.arange(lag), acfs)
    ax1.title.set_text('ACF')
    ax2.plot(np.arange(lag), pacfs)
    ax2.title.set_text('PACF')
    plt.xticks(np.arange(0, lag, step=1))
    plt.show()

def tsa_summary(args):
    data,_ = data_helper(f=args.f,t=args.t,type=args.type,int=args.int,start=args.start,end=args.end)

    est = tsa_helper(data,ar=args.ar,d=args.d,ma=args.ma)

    fit, err, serr, var, l = est.fit()
    arc = fit[:args.ar]
    mac = fit[-args.ma:]
    model = tsa.arima(arc,args.d,mac,var)

    #Plot acf of residuals, standardised and q stat
    lag = args.lag
    acfs = est.estacf(lag,err)
    qstat = est.get_qstat(err,H=lag)

    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.plot(serr)
    ax1.title.set_text('Standardised error')
    ax2.plot(acfs)
    ax2.title.set_text('ACF')
    ax3.plot(qstat)
    ax3.title.set_text('Q-Statistic')
    stats.probplot(serr, dist="norm", plot=ax4)
    ax4.title.set_text('Q-Q plot')

    #AIC,BIC and AICc
    print("AIC: ", est.get_AIC(l))
    print("BIC: ", est.get_BIC(l))
    print("AICc: ", est.get_AICc(l))

    plt.show()

def tsa_forecast(args):
    data,mean = data_helper(f=args.f,t=args.t,type=args.type,int=args.int,start=args.start,end=args.end)
    data-=mean

    est = tsa_helper(data,ar=args.ar,d=args.d,ma=args.ma)
    fit, err, serr, var, l = est.fit()
    arc = fit[:args.ar]
    mac = fit[-args.ma:]
    model = tsa.arima(arc,args.d,mac,var)

    m = args.points
    train = data[-m:]
    forecast, p = model.forecast(train,m)
    forecast+=mean
    ci = 1.96 * np.array(p)/np.sqrt(len(train))

    fig, ax = plt.subplots()
    x = np.arange(len(data)-1, len(data)+m-2)
    ax.plot(x,forecast)
    ax.fill_between(x, (forecast-ci), (forecast+ci), color='b', alpha=.1)

    plt.plot(data+mean)
    plt.show()

def main():
    #Commands
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(
        title="Commands"
    )

    tsa_parser = subparsers.add_parser("tsa")
    tsa_subparser = tsa_parser.add_subparsers()
    tsa_parser.add_argument("-f",help="path to file")
    tsa_parser.add_argument("-t",help="symbol in Yahoo finance if not using file")
    tsa_parser.add_argument("--type",help="path to file",choices=["open","high","low","close","adjclose","volume"])
    tsa_parser.add_argument("--int",help="interval between data points",choices=["1d","5d","1m","1y","5y","6m"],default="1d")
    tsa_parser.add_argument("--start",help="start date (in format YYYY-MM-DD)",type=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    tsa_parser.add_argument("--end",help="end date (in format YYYY-MM-DD)",type=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    tsa_parser.add_argument("--ar", help="AR order",type=int,default=1)
    tsa_parser.add_argument("--ma", help="MA order",type=int,default=1)
    tsa_parser.add_argument("--d", help="Backshift order",default=0,type=int)

    tsa_forecast_parser = tsa_subparser.add_parser("forecast")
    tsa_forecast_parser.add_argument("points",help="number of points to forecast",type=int)
    tsa_forecast_parser.set_defaults(func=tsa_forecast)

    tsa_sum = tsa_subparser.add_parser("summary", help="provides ACF, standardised residuals and Q-statistic")
    tsa_sum.add_argument("lag",help="lag",type=int)
    tsa_sum.set_defaults(func=tsa_summary)

    tsa_cf = tsa_subparser.add_parser("corr",help="plots ACF and PACF")
    tsa_cf.add_argument("lag",help="lag",type=int)
    tsa_cf.set_defaults(func=tsa_corr)

    sim_parser = subparsers.add_parser("sim")
    sim_subparser = sim_parser.add_subparsers()
    sim_parser.add_argument("-ar",action='append',help="AR coefficients",type=float)
    sim_parser.add_argument("-ma",action='append',help="MA coefficients",type=float)
    sim_parser.add_argument("-d",type=int,help="difference order",default = 0)
    sim_parser.add_argument("-var",type=float,help="variance",default = 1)

    sim_sample_parser = sim_subparser.add_parser("sample")
    sim_sample_parser.add_argument("n",type=int,help="number of samples")
    sim_sample_parser.set_defaults(func=sim_sample)

    sim_acf_parser = sim_subparser.add_parser("acf")
    sim_acf_parser.add_argument("lag",type=int,help="lag")
    sim_acf_parser.set_defaults(func=sim_acf)

    sim_pacf_parser = sim_subparser.add_parser("pacf")
    sim_pacf_parser.add_argument("lag",type=int,help="lag")
    sim_pacf_parser.set_defaults(func=sim_pacf)

    smooth_parser = subparsers.add_parser("sm")
    smooth_subparser = smooth_parser.add_subparsers()
    smooth_parser.add_argument("-f",help="path to file")
    smooth_parser.add_argument("-t",help="symbol in Yahoo finance if not using file")
    smooth_parser.add_argument("--type",help="path to file",choices=["open","high","low","close","adjclose","volume"])
    smooth_parser.add_argument("--int",help="interval between data points",choices=["1d","5d","1m","1y","5y","6m"],default="1d")
    smooth_parser.add_argument("--start",help="start date (in format YYYY-MM-DD)",type=lambda d: datetime.strptime(d, '%Y-%m-%d'))
    smooth_parser.add_argument("--end",help="end date (in format YYYY-MM-DD)",type=lambda d: datetime.strptime(d, '%Y-%m-%d'))

    sg_smooth_parser = smooth_subparser.add_parser("sg")
    sg_smooth_parser.add_argument("winsize",help="window size",type=int)
    sg_smooth_parser.add_argument("degree",help="degree (must be less than winsize)",type=int)
    sg_smooth_parser.set_defaults(func=smooth_sg)

    norm_smooth_parser = smooth_subparser.add_parser("norm")
    norm_smooth_parser.add_argument("bandwidth",help="bandwidth",type=float)
    norm_smooth_parser.set_defaults(func=smooth_norm)

    ols_smooth_parser = smooth_subparser.add_parser("ols")
    ols_smooth_parser.set_defaults(func=smooth_ols)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()