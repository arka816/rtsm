'
  SIMPLE LINEAR REGRESSION IMPLEMENTED FROM SCRATCH
  AS PART OF THE COURSE RTSM (MA31020)
  
  Both X and Y are stochastic variables.
  X has been sampled from an uniform distribution.
  The regression model is:
    Y_i = B0 + B1 * X_i + E_i
  where E_i is gaussian noise.
  Homoskedastic noise has been assumed.
'

## SAMPLE GENERATION
# sample size
N <- 100
p <- 2
# regression coefficients
b0 <- 2
b1 <- 3
X <- runif(N, 2, 5)
# Gaussian noise
epsilon <- rnorm(N, 0, 1)
y <- b0 + b1 * X + epsilon


fit <- function(X, y){
  df_res <<- N - p
  df_model <<- p - 1
  X_mean <<- mean(X)
  y_mean <<- mean(y)
  X_dev <- X-X_mean
  y_dev <- y - y_mean
  S_xy <<- X_dev %*% y
  S_xx <<- X_dev %*% X_dev
  S_yy <<- y_dev %*% y_dev
  
  # estimate the regression coefficients
  B1 <<- S_xy / S_xx
  B0 <<- y_mean - B1 * X_mean
  
  # estimate the population variance
  pop_var <<- (S_yy - (S_xy ** 2) / S_xx) / (N - 2)
  
  return (c(B0, B1, pop_var))
}

predict <- function(X){
  return (as.vector(B0) + as.vector(B1) * X)
}

r_squared <- function(){
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

estimate_var <- function(){
  # Estimates the distribution
  # of the estimates of the regression
  # coefficients and population variance
  
  B0_var <- pop_var * (1 / N + (X_mean ** 2) / S_xx)
  B1_var <- pop_var / S_xx
  
  se_B0 <<- sqrt(B0_var)
  se_B1 <<- sqrt(B1_var)
  se_var <<- sqrt(2 * pop_var / (N-2))
  
  return (c(se_B0, se_B1, se_var))
}

t_test <- function(){
  # Performs a t test on the parameter
  # estimates with null hypothesis
  # H0: Bi = 0 
  # and alternative hypothesis
  # H1: Bi != 0
  
  T_B1 <<- B1 / sqrt(pop_var / S_xx)
  T_B0 <<- B0 / sqrt(pop_var * (1 / N + (X_mean ** 2) / S_xx))
  
  p_B1 <<- pt(T_B1, df=df_res, lower.tail = FALSE) * 2
  p_B0 <<- pt(T_B0, df=df_res, lower.tail = FALSE) * 2
}

F_test <- function(){
  # Performs a t test on the parameter
  # estimates with null hypothesis
  # H0: Bi = 0 for all i
  # and alternative hypothesis
  # H1: Bi != 0 for all i
  # tests whether the model performs better than
  # a model with vanishing coefficients
  
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
y_pred <<- predict(X)
r_squared()
adj_r_squared()
estimate_var()
t_test()
F_test()
calc_skewness()
calc_kurtosis()
durbin_watson()
