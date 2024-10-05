import numpy as np
from math import floor,ceil


class regression_model():
    """
    Base class for regression models, defined by given data points

    Parameters:
        x, vector with each point's corresponding abscissa
        
        y, vector with each point's corresponding ordinate
    """
    def __init__(self,xs,y):
        self.xs = xs
        self.y = y
        self.d = len(self.xs)
        self.n = len(self.xs[0])

        self.sse = 0
        self.mse = 0
        self.r2 = 0
        self.mean_x = [np.mean(x) for x in xs]
        self.mean_y = np.mean(y)
    
class lm(regression_model):
    """
        Class for ordinary least squares regression
    """
    def __init__(self, xs, y):
        super().__init__(xs, y)
        #Coefficient matrix for least squares
        self.coeffs = np.append([np.ones(self.n)],self.xs,axis=0)

    def fit(self)->float:
        """
        Fits a linear model via OLS

        Returns the vector of estimated coefficients
        """
        return np.matmul( np.linalg.inv((np.matmul(self.coeffs,self.coeffs.T))), np.matmul(self.coeffs,self.y))
    
    def get_SSE(self, estco)->float:
        """
        Returns the sum of square residuals given estimated coefficients.

        Parameters:
            estco, vector of coefficients
        """
        estimates = sum(estco[0] + [estco[k] * self.xs[k-1] for k in range(1,estco.size)])
        residuals = self.y - estimates

        self.sse = np.matmul(residuals.T,residuals)

        return self.sse
    
    def get_MSE(self)->float:
        """
        Returns the MSE of a simple linear regression model
        """
        self.mse = self.sse/(self.n-(self.d+1))

        return self.mse
    
    def get_R2(self)->float:
        """
        Returns the coefficient of determination or R^2
        """
        tss = sum([(self.y[i] - self.mean_y)**2 for i in range(self.y.size)])

        return 1-self.mse/tss
    
    def get_AIC(self)->float:
        """
        Returns the Akaike Information Criterion assuming normal regression.
        """
        mlevar = self.sse/self.n

        return np.log(mlevar) + (self.n+2*self.d)/self.n

    def get_BIC(self)->float:
        """
        Returns the Bayesian Information Criterion assuming normal regression 
        """
        mlevar = self.sse/self.n

        return np.log(mlevar) + (np.log(self.n)*self.d)/self.n

    def get_AICc(self)->float:
        """
        Returns the AIC adjusted for bias
        """
        mlevar = self.sse/self.n

        return np.log(mlevar) + (self.n+self.d)/(self.n-self.d-2)


def normal_kernel(x,y,bandwidth=1):
    """
    Returns the smoothed estimates on y via a normal kernal

    Parameters:
        x, vector with each point's corresponding abscissa

        y, vector with each point's corresponding ordinate

        bandwidth, default value of 1 that determines the 'smoothness' of the kernel.
    """

    est = []

    #Return the smoothed average for each point based on normal distribution
    for xval in x:
        kernel = 1/(bandwidth*np.sqrt(2*np.pi))*np.exp(-(xval-x)**2/2*bandwidth**2)
        estx = sum(y*kernel)/sum(kernel)

        est.append(estx)

    return est

def sgfilt(y,winsize:int,degree:int):
    """
    Returns the smoothed data using a Savitzky–Golay filter

    Parameters:
        y, vector with each point's corresponding ordinate

        winsize, the desired window size

        degree, the degree of each local polynomial (MUST BE LESS THAN winsize)
    
    Reference:
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5888646
    """
    #Power series computation on a given z
    getco = lambda z: [z**k for k in range(degree+1)]
    coi = []

    #Get the Vandermonde matrix
    for j in range(-floor(winsize/2), ceil(winsize/2)):
        coi.append(getco(j))

    coi = np.array(coi)
    ata = np.matmul(coi.T,coi)
    inv0 = np.linalg.solve(ata,np.identity(degree+1)[0])

    #Perform convolution over the sample data using 0-centered coefficients as kernel
    return np.convolve(np.matmul(inv0,coi.T),y,mode="same")