import numpy as np
from numpy.polynomial import polynomial as P
from scipy.linalg import solve_toeplitz
from scipy.optimize import minimize
from scipy.stats import chi2

class arima:
    """
    Create a zero-mean ARMA(p,q) model given coefficients for AR and MA based on standard Gaussian white noise variates.

    Parameters
        ar, AR coefficients, t-i corresponds to ith element 
        
        ma, MA coefficients, t-i corresponds to ith element 

        d, an integer representing the difference order for ARIMA
    """
    def __init__(self, ar, d,ma, var):
        self.ar = ar[::-1]
        self.ma = ma[::-1]
        self.p = len(ar)
        self.q = len(ma)
        self.d = d
        self.var = var

        self.arpoly = np.append([1],-1*np.array(ar))
        self.mapoly = [1]+ma
        self.dpoly = -1*(P.polymul(self.arpoly, P.polypow([1,-1], d)))[1:]

    def sim(self, n):
        """
        Simulate n samples from the given model. 
        If the coefficient vectors are both empty, n samples will be drawn from the standard Gaussian.
        
        Parameters:
            n, an integer of the number of samples.

            var, the variance
        """
        wt = np.random.normal(0, self.var, n+max(len(self.dpoly),self.q))
        xt = np.zeros(n+max(len(self.dpoly),self.q))
        upper = max(len(self.dpoly),self.q)

        if self.d != 0:
            xt = np.zeros(n+upper)
            wt = np.random.standard_normal(n+upper)
        
        for k in range(upper,n+upper):
            #Apply coeffs over ARMA window
            xt[k] = np.dot(self.dpoly[::-1], xt[k-len(self.dpoly):k]) + np.dot(self.ma, wt[k-self.q:k]) + wt[k]
        
        return xt[upper:].tolist()
    
    def roots(self):
        """
        Return the roots of the AR and MA polynomials
        """
        return {"ar":np.roots(self.arpoly[::-1]),"ma":np.roots(self.mapoly[::-1])}
    
    def __getpsi(self,lag):
        """
        Helper method to obtain the psi weights for a specified lag.
        """
        psi = [1]
        
        #For 0 < k <= max(p,q+1)
        for k in range(max(self.p,self.q+1)):
            theta = 0
            if k < self.q:
                theta = self.ma[k]

            insum = 0
            if k < self.p:
                insum = np.dot(psi,self.ar[::-1][:k+1])
            
            psi.append(theta + insum)
        
        #For k > max(p,q+1)
        if lag > max(self.p,self.q+1):
            for k in range( max(self.p,self.q+1),lag):
                psi.append(np.dot(psi[::-1][:self.p],self.ar[::-1]))
        
        return psi
    
    def acov(self, lag):
        """
        Calculate the autocovariance at a specified lag

        Parameter:
            lag, an integer of which the autocovariance is to be calculated at
        """
        psi = self.__getpsi(lag)

        return self.var*sum([psi[i]*psi[i-lag] for i in range(lag,len(psi))])

    def acf(self,lag):
        """
        Returns a vector of autocorrelated values for a casual ARMA process

        Parameter:
            lag, an integer indicating the number of lags to calculate

        References:
            Based on Peter J. Brockwell, Richard A. Davis, 3.3.3 and 3.3.4
        """
        aroots = np.abs(self.roots()["ar"])
        acfs = []

        #Model must be casual
        if np.any(aroots < 1): 
            return []

        for h in range(lag):
            acfs+=[self.acov(h)/self.acov(0)]
        
        return acfs

    def pacf(self,lag):
        """
        Returns a vector of partial autocorrelated values

        Parameter:
            lag, an integer indicating the number of lags to calculate

        References:
            Levinson recursion via toeplitz solver
        """
        pacfs = [1]
        acov0 = self.acov(0)

        for n in range(1,lag):
            #first row and col of acf matrix
            r = [1]+[self.acov(i)/acov0 for i in range(1,n)]
            #vector of acfs
            b = [self.acov(i)/acov0 for i in range(1,n+1)]

            pacfs.append(solve_toeplitz(r,b)[-1])
        
        return pacfs

    
    def forecast(self, data, m):
        """
        Forecast m data points on given data based on truncated ARIMA prediction.

        Returns the forecasted data points and corresponding mean-square prediction errors
        """
        pe = []
        transformed = np.array(data[:])
        wpred = [0 for i in range(self.q)]
        ipoly = np.append([1],-1*self.dpoly)

        for t in range(1,m):
            #Get the past noise variates
            wt =  np.dot(transformed[-len(ipoly):][::-1], ipoly)

            if self.q > 0:
                wt -= np.dot(self.ma, wpred[-self.q:])

            xt = np.dot(transformed[-len(self.dpoly):][::-1], self.dpoly)+wt
            wpred.append(wt)
            transformed = np.append(transformed, xt)

            #Get the errors using psi weights
            psi = self.__getpsi(t)
            pe.append(sum([psi[j]**2 for j in range(t)]))
        
        return transformed[-m+1:], pe


class estimation:
    def __init__(self, data, p, d, q):
        """
        A class for fitting ARIMA(p,d,q) models

        Parameters:
            data, the data to be fitted (non-empty)

            p, guess of AR order

            d, backshift order

            q, guess of MA order
        """
        self.data = data
        self.p = p
        self.d = d
        self.q = q

    def estacov(self,lag ,data=[]):
        """
        Estimates the autocovariance at a specified lag for the given data (default is the fitted data)

        Reference:
            Based on Peter J. Brockwell, Richard A. Davis 7.2.1
        """
        data = self.data if len(data)==0 else data
        estmean = np.mean(data)
        n = len(data)

        return (1/n)*sum( [(data[j] - estmean)*(data[j+lag] - estmean) for j in range(1,n-lag)] )

    
    def estacf(self, lag ,data=[]):
        """
        Returns the estimated autocorrelated function up to the specified lag

        Reference:
            Based on Peter J. Brockwell, Richard A. Davis 7.2.2
        """
        data = self.data if len(data)==0 else data
        return np.array([self.estacov(i,data) for i in range(lag)])/self.estacov(0)
    
    def estpacf(self,lag,data=[]):
        """
        Returns a vector of estimated partial autocorrelated values

        Parameter:
            lag, an integer indicating the number of lags to calculate

        References:
            Levinson recursion via toeplitz solver
        """
        data = self.data if len(data)==0 else data
        pacfs = [1]
        acov0 = self.estacov(0,data)

        for n in range(1,lag):
            #first row and col of acf matrix
            r = np.append(1,np.array([self.estacov(i,data) for i in range(1,n)])/acov0)
            #vector of acfs
            b = np.array([self.estacov(i,data) for i in range(1,n+1)])/acov0

            pacfs.append(solve_toeplitz(r,b)[-1])
        
        return pacfs
    
    def yulewalker(self):
        """
        Returns the fitted coefficients for an AR(p) model and variance

        Returns
            - estimated AR coefficients

            - variance
        """
        p = self.estacf(self.p+1)[1:]
        #row of p matrix
        r = self.estacf(self.p)
        coeffs = solve_toeplitz(r,p)

        var = self.estacov(0)*(1 - np.matmul(p.T , coeffs))

        return coeffs, var
    
    def innoalg(self,delta=0.001):
        """
        Estimate MA parameters using innovations

        Parameters:
            delta, an optional parameter used for convergence
        
        Returns:
            - estimated MA coefficients

            - corresponding errors

            - one-step ahead predictors
        """
        acov0 = self.estacov(0,self.data)
        theta = [[self.estacov(1,self.data)/acov0]]
        p = [acov0, acov0 - acov0*theta[0][0]**2]
        xs = [0]

        for m in range(2,len(self.data)-1):
            #Generate the thetas for row m
            thetam = [self.estacov(m)/acov0]
            for k in range(1,m):
                thetam.append((self.estacov(m-k) - sum([thetam[j]*theta[k-1][j]*p[j] for j in range(k)]))/p[k])

            p.append(acov0 - sum([p[j]*(thetam[j]**2) for j in range(m)]))
            theta.append(thetam)

            #one-step ahead predictors
            xs.append(sum([thetam[-i]*(self.data[m+1-i]-xs[-i]) for i in range(m)]))

            #Convergent if l2 norm of coefficients are <= delta
            if np.abs(np.linalg.norm(theta[-1][-self.q:][::-1]) - np.linalg.norm(theta[-2][-self.q:][::-1])) <= delta:
                break

        return theta[-1][-self.q:][::-1], p, xs
    
    def dlalgo(self):
        """
        Calculate the errors up to the nth AR coefficient

        Returns
            - estimated AR coefficients

            - corresponding errors
        """
        acov0 = self.estacov(0)
        p = [acov0]
        phi = self.estpacf(len(self.data))

        for i in range(1,len(self.data)):
            p.append(p[i-1]*(1-phi[i]**2))

        return phi, p

    def fit(self, method="BFGS"):
        """
        Fit a zero-mean ARMA(p,q) model using MLE; minimisation of the log-likelihood is done by the specified method (default is BFGS)

        Returns:
            - the estimated coefficients

            - the residuals

            - the standardised residuals

            - the variance

            - the log likelihood

        References:
            Shumway,Stoffer 3.5
        """
        theta = []
        phi = []
        data = np.diff(self.data,self.d)

        if self.p > 0:
            phi, _ = self.yulewalker()
            
        if self.q > 0:
            theta, _, _ = self.innoalg()

        initial = np.append(phi, theta)
        r = lambda ps: [1-ps[i]**2 for i in range(self.p)]
        ls = lambda ps, qs: np.sum( [( data[i] - np.dot(data[i-self.p:i], ps[::-1]) - np.dot(data[i-self.q:i], qs[::-1]) )**2 / np.prod(r(ps)) for i in range(max(self.p,self.q),len(data))] )
        
        #MLE estimation of ARMA(p,q)
        def conlikelihood(x):
            qs = x[-self.q:] if self.q > 0 else []
            ps = x[:self.p] if self.p > 0 else []
            
            rp = r(ps)
            ss = ls(ps,qs)
            rs = np.sum(np.log(rp))
            
            return np.log(ss/ len(data) ) + rs/len(data)

        res =  minimize(conlikelihood, x0=initial, method=method)
        coeffs = res["x"]
        l = res["fun"]
        theta = coeffs[-self.q:] if self.q > 0 else []
        phi =  coeffs[:self.p] if self.p > 0 else []
        var = ls(phi,theta)/len(data)
        errors, serrors = self.get_res(phi,theta,var,data)

        return coeffs, errors, serrors, var, l
    
    def get_res(self, phi,theta, var, data):
        """
        Returns the residuals and standardised residuals

        Parameters:
            phi, the AR coefficients
            
            theta, the MA coefficients

            var, the variance

            data, the data that was fitted
        """
        errors = []

        for i in range(max(self.p,self.q)+1,len(data)):
            errors.append(data[i] - np.dot(data[i-self.p:i], phi[::-1]) - np.dot(data[i-self.q:i], theta[::-1]))

        return errors, np.array(errors)/np.sqrt(var)

    def get_qstat(self, errors, H=20):
        """
        Return a vector of Ljung–Box–Pierce Q-statistics up to the given lag 

        Parameters:
            errors, the residuals of the fit

            H, the maximum lag
        """
        pvals = []
        n = len(self.data)
        acov0 = self.estacov(0,data=errors)
        
        for i in range(self.p+self.q+1,H+self.p+self.q+1):
            qs = n*(n+2)*sum( [ (self.estacov(j,data=errors)/acov0)**2 / (n-j)  for j in range(1,i)] )
            pvals.append(1-chi2.cdf(qs, i-self.p-self.q))

        return pvals

    def get_AIC(self, l, k=0)->float:
        """
        Returns the Akaike Information Criterion assuming normal regression.

        Parameters:
            l, the log likelihood

            k, 0 for zero-mean fit
        """
        return -2*l + 2*(self.p+self.q+1+k)

    def get_BIC(self, l, k=0)->float:
        """
        Returns the Bayesian Information Criterion assuming normal regression 

        Parameters:
            l, the log likelihood

            k, 0 for zero-mean fit
        """
        return self.get_AIC(l) + (np.log(len(self.data)) - 2)*(self.p+self.q+1+k)

    def get_AICc(self, l, k=0)->float:
        """
        Returns the AIC adjusted for bias

        Parameters:
            l, the log likelihood

            k, 0 for zero-mean fit
        """
        return self.get_AIC(l) + (2*(self.p+self.q+1+k)*(self.p+self.q+2+k))/(len(self.data) - self.p - self.q - k - 2)