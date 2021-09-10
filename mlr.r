'
  MULTIPLE LINEAR REGRESSION IMPLEMENTED FROM SCRATCH
  AS PART OF THE COURSE RTSM (MA31020)
  
  Y is a stochastic variable.
  X is a stochastic vector.
  
  X has been sampled from an uniform distribution.
  
  The regression model is:
    Y_i = B0 + dot(B_R, X_i) + epsilon_i
  
  where epsilon_i is gaussian noise.
  Homoskedastic noise has been assumed.
'
library(matlib)

## SAMPLE GENERATION
# sample size
N <- 100
# number of regressors (not including intercept)
p <- 6
# regression coefficients
B = c(2, 3, 1, -3, -2, -4, 5)
# generate sample
x <- matrix(runif(N * p, min=3, max=5), N)
X <- cbind(matrix(1, N, 1), x)
# Gaussian noise
epsilon <- rnorm(N, 0, 1)
y <- X %*% B + epsilon


# REGRESSION
fit <- function(X, y){
  df_res <<- N - p - 1
  df_model <<- p
  
  # estimate the regression coefficients
  b <<- inv(t(X) %*% X) %*% t(X) %*% y
  
  # estimate the population variance
  y_pred <<- predict(X)
  e <<- y - y_pred
  pop_var <<- (t(e) %*% e)[1][1] / (N - p - 1)
}

predict <- function(X){
  y = X %*% b
  return (y)
}

r_squared <- function(){
  y_mean <<- mean(y)
  SS_res <<- sum((y - y_pred) ** 2)
  SS_tot <<- sum((y - y_mean) ** 2)
  SS_model <<- SS_tot - SS_res
  cod <<- 1 - SS_res/SS_tot
  return (cod)
}

adj_r_squared <- function(){
  adj_cod <<- 1 - ((1 - cod) * (N - 1) / (N - p - 1))
  return (adj_cod)
}

coeff_var <- function(){
  # Estimates the variance
  # of the estimates of the regression
  # coefficients and population variance
  # calculates the covariance matrix of the regression coefficient vector b
  
  coeff_dispersion <<- (inv(t(X) %*% X)) * pop_var
  coeff_std_error <<- sqrt(diag(coeff_dispersion))
  return (coeff_std_error)
}

t_test <- function(){
  # Performs a t test on the parameter
  # estimates with null hypothesis
  # H0: Bi = 0 
  # and alternative hypothesis
  # H1: Bi != 0
  
  T_b <<- b / coeff_std_error
  p_b <<- c()
  for(t in T_b){
    p_b <<- append(p_b, 2 * pt(t, df=df_res, lower.tail = t < 0))
  }
}

F_test <- function(){
  # Performs a t test on the parameter
  # estimates with null hypothesis
  # H0: Bi = 0 for all i
  # and alternative hypothesis
  # H1: Bi != 0 for all i
  # tests whether the model performs better than
  # a model with vanishing coefficients
  # non centrality parameter of the F statistic is 0 under H0
  
  F_statistic <<- (SS_model / df_model) / (SS_res / df_res)
  p_F <<- pf(F, df_model, df_res, lower.tail = TRUE)
}

calc_skewness <- function(){
  e <- y - y_pred
  m3 = sum(e ** 3)
  m2 = sum(e ** 2)
  skewness <<- m3 / (m2 ** 1.5)
}

calc_kurtosis <- function(){
  e <- y - y_pred
  m4 = sum(e ** 4)
  m2 = sum(e ** 2)
  kurtosis <<- m4 / (m2 ** 2)
}

durbin_watson <- function(){
  # test for homoskedasticity
  
  e <- y - y_pred 
  dw <<- sum(diff(e) ** 2) / sum(e ** 2)
}

fit(X, y)
y_pred <- predict(X)
r_squared()
adj_r_squared()
coeff_var()
t_test()
F_test()
