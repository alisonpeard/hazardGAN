library(reticulate)
library(threshr)
library(knitr)
library(forecast)

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

np <- import("numpy")
var <- 'wind_data'
block.size <- 'daily'

if(block.size == 'daily'){
  freq <- 365
}else if(block.size == 'weekly'){
  freq <- 52
}else if(block.size == 'monthly'){
  freq <- 12
}

X <- np$load(paste0("/Users/alison/Documents/DPhil/multivariate/", var, "/", block.size, "/train/images.npy"))
X = X[,,,1]
M <- dim(X)[2]
N <- dim(X)[3]
heatmap(X[1,,])
dim(X)

i = 15; j = 20;
x <- X[,i,j]
plot(x, type='l')
x.ts <- ts(x, frequency=freq)
x.decomp <- data.frame(stl(x.ts, s.window='periodic')$time.series)
par(mfrow=c(4,1));plot(x.ts, type='l');plot(x.decomp$seasonal, type='l');plot(x.decomp$trend, type='l');plot(x.decomp$remainder, type='l')
par(mfrow=c(1,1));plot(x.ts, type='l')
x.deseasoned <- x.ts - x.decomp$seasonal
par(mfrow=c(1,1));plot(x.deseasoned, type='l')
par(mfrow=c(1,2));acf(x.deseasoned, lag.max=(1 * freq));pacf(x.deseasoned, lag.max=(1 * freq))
par(mfrow=c(1,2));acf(diff(x.deseasoned), lag.max=(1 * freq));pacf(diff(x.deseasoned), lag.max=(1 * freq))

par(mfrow=c(1,1))
x.d <- diff(x.ts)
plot(x.d, type='l')
par(mfrow=c(1,2));acf(x.d, lag.max=freq);pacf(x.d, lag.max=freq)

x.D <- diff(x.ts, freq)
plot(x.D, type='l')
par(mfrow=c(1,2));acf(x.D, lag.max=freq);pacf(x.D, lag.max=freq)

var <- 'remainder'
par(mfrow=c(1,2))
acf(x.decomp[[var]], lag.max=53)
pacf(x.decomp[[var]], lag.max=53)

acf(diff(x.decomp[[var]]), lag.max=53)
pacf(diff(x.decomp[[var]]), lag.max=53)

n <- length(x.ts)
obsv <- 1:n
sine.wave <- sin((2 * pi * obsv) / freq)
cos.wave <- cos((2 * pi * obsv) / freq)
plot(sine.wave, type='l')
lines(cos.wave)
data.frame(sin=sine.wave, cos=cos.wave)
#model <- auto.arima(x.ts, xreg=sine.wave)

# very simplest AR(1) model
model <- arima(x.ts, seasonal=c(2,1,0))
res <- model$residuals
par(mfrow=c(1,2));acf(res, lag.max=(4*freq));pacf(res, lag.max=(4*freq))
par(mfrow=c(1,2));acf(res, lag.max=freq);pacf(res, lag.max=freq)
summary(model)

