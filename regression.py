"""
    SIMPLE LINEAR REGRESSION IMPLEMENTED FROM SCRATCH
    AS PART OF THE COURSE RTSM (MA31020)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t, f

# GENERATE DATA
# samples
N = 100
# regression coefficients
b0 = 2
b1 = 3
X = np.random.uniform(low=2, high=5, size=N)
epsilon = np.random.normal(loc=0, scale=1, size=N)
Y = b0 + b1*X + epsilon

# WRITE DATA TO FRAME
df = pd.DataFrame()
df['Y'] = Y
df['X'] = X

# READ DATA FROM FRAME
X = np.array(df['X'])
y = np.array(df['Y'])
N = len(X)
X = X.reshape((N, 1))


# LIBRARY CODE FOR OLS
model = sm.OLS(y,sm.add_constant(X))
results = model.fit()
print(results.summary())


class SLR:
    def __init__(self, endog, exog):
        self.n, self.p = exog.shape
        self.p += 1
        self.df_res = self.n - self.p
        self.df_model = self.p - 1
        self.X = exog.reshape(self.n)
        self.y = endog.reshape(self.n)
    
    def fit(self):
        X_mean = np.mean(self.X)
        X_dev = (self.X - X_mean)
        S_xy = np.dot(X_dev, self.y)
        S_xx = np.dot(X_dev, X_dev)
        self.B1 = S_xy / S_xx
        self.B0 = np.mean(self.y) - self.B1 * X_mean
        self.coef = {"intercept": self.B0, "slope": self.B1}
        
    def predict(self, X):
        return self.B0 + self.B1 * X
    
    def var_coeff(self):
        y_pred = self.predict(self.X)
        SS_res = np.sum((y_pred - self.y) ** 2)
        SS_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        return 1 - SS_res/SS_tot
    
    def adj_var_coeff(self, var_coeff=None):
        if var_coeff == None:
            var_coeff = self.var_coeff()
            
        return 1 - ((1 - var_coeff) * (self.n-1) / (self.n - self.p - 1))
    
    def var_estimator(self):
        X_mean = np.mean(self.X)
        X_dev = (self.X - X_mean)
        y_dev = self.y - np.mean(self.y)
        S_xx = np.dot(X_dev, X_dev)
        S_xy = np.dot(X_dev, self.y)
        S_yy = np.dot(y_dev, y_dev)
        
        # estimate the population variance
        self.pop_var = (S_yy - (S_xy ** 2) / S_xx) / (self.n - 2)
        
        B0_var = self.pop_var * (1 / self.n + (X_mean ** 2) / S_xx)
        B1_var = self.pop_var / S_xx
        
        self.se_B0 = np.sqrt(B0_var)
        self.se_B1 = np.sqrt(B1_var)
        
        return self.se_B0, self.se_B1
    
    def t_test(self):
        # Performs a t test on the parameter
        # estimates with null hypothesis
        # H0: Bi = 0 
        # and alternative hypothesis
        # H1: Bi != 0
        
        X_mean = np.mean(self.X)
        X_dev = (self.X - X_mean)
        S_xx = np.dot(X_dev, X_dev)
        
        T_B1 = self.B1 / np.sqrt(self.pop_var / S_xx)
        T_B0 = self.B0 / np.sqrt(self.pop_var * (1 / self.n + X_mean * X_mean / S_xx))
        
        p_B1 = t.sf(T_B1, self.df_res) * 2
        p_B0 = t.sf(T_B0, self.df_res) * 2
        
        return T_B0, T_B1, p_B0, p_B1
        
    def F_test(self):
        y_pred = self.predict(self.X)
        SS_res = np.sum((y_pred - self.y) ** 2)
        SS_model = np.sum((y_pred - np.mean(self.y)) ** 2)
        
        F = (SS_model / (self.df_model)) / (SS_res / (self.df_res))
        
        # non-centrality parameter calculation
        # ncp = (np.dot(y_pred, y_pred) - (np.sum(y_pred) ** 2) / self.n) / self.pop_var
        
        p_F = f.sf(F, self.df_model, self.df_res)
        return F, p_F
    
    def log_likelihood(self):
        y_dev = (self.y - self.predict(self.X))
        return -self.n * np.log(2 * np.pi * self.pop_var) / 2 + np.dot(y_dev, y_dev) / (2 * self.pop_var)
    
    def summary(self):
        print(self.coef)
        print("R-squared", self.var_coeff())
        print("Adj. R-squared", self.adj_var_coeff())
        print("variance of estimator: ", self.var_estimator())
        print("t statistic: ", self.t_test())
        print("F statistic", self.F_test())
        print("log likelihood: ", self.log_likelihood())

        
model = SLR(y, X)
model.fit()
model.summary()
