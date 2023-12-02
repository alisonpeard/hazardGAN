rm(list=ls())
library(reticulate)
library(threshr)
library(knitr)

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

np <- import("numpy")
var <- 'wave_data'
block.size <- 'weekly'

if(var=='precip_data'){
  min_quantile <- .9  # very skewed because of near-zero observations
}else{
  min_quantile <- .6
}

nexcesses <- 20

X <- np$load(paste0("/Users/alison/Documents/DPhil/multivariate/", var, "/", block.size, "/train/images.npy"))
M <- dim(X)[2]
N <- dim(X)[3]

u_mat <- matrix(nrow=M, ncol=N)  # matrix of threshold values
n.excesses <- matrix(nrow=M, ncol=N)
verbose <- TRUE
for(i in 1:M){
  for(j in 1:N){
    x <- X[, i, j, 1]
    if(var(x) > 0){
      attr(x, 'npy') <- npy
      
      # restrict search so that number of excesses always >= 10
      max.allowed <- sort(x)[(length(x) - nexcesses)]
      max.quantile <- ecdf(x)(max.allowed)
      
      q_vec <- seq(min_quantile, max.quantile, by=0.005)
      u_vec <- quantile(x, p=q_vec)
      
      #var_cv <- ithresh(x, u_vec=u_vec, trans='BC')
      best_u <- NULL
      for(u in u_vec){
        excesses <- x[x>u]
        pval <- Box.test(excesses)$p.value
        if(pval > 0.01){ # i.e., if fail to reject H0: all independent at 99% confidence level
          best_u <- u
          break
        }
      }
      if(!is.null(best_u)){
        u_vec <- u_vec[u_vec >= best_u]
        var_cv <- ithresh(x, u_vec=u_vec, trans='BC')
        best_u <- getmode(summary(var_cv)[, "best u"])
        
        best_u <- max(x[x <= best_u]) # use an actual observation as threshold, important for interpolating
        
        if(FALSE){
          # for looking at results in dev
          hist(x);abline(v=best_u, col='red', lwd=2, lty=2)
          excesses <- x[x>best_u]
          excesss.ind <- x[x>best_u]
          print(paste0(length(excesses), " excesses selected."));print(paste0("F(u) = ", ecdf(x)(best_u)))
          par(mfrow=c(1,1));plot(x, type='l');points(which(x>best_u), x[x>best_u], pch=20, col='red');abline(h=best_u, col='red')
          Box.test(excesses)
        }
        if(verbose){
          excesses <- x[x>best_u]
          print(paste0(length(excesses), " excesses selected for (", i, ', ', j, ')'))
          print(paste0("Ljung-Box p-value: ", round(Box.test(excesses)$p.value, 4)))
        }
        
        print(paste0('Best u for (', i, ", ", j, "): ", round(best_u, 4)))
        u_mat[i, j] <- best_u
        n.excesses[i, j] <- length(x[x > best_u])
      }else{
        print(paste0("No threshold found for (", i, ", ", j, ")"))
        u_mat[i, j] <- NA
      }
    }else{
      u_mat[i, j] <- NA
    }
  }
}

heatmap(u_mat)
excesses <- x[x>best_u]
inds <- which(x > best_u)
plot(x, type='l');points(inds, excesses, col='red', pch=20);abline(h=best_u, col='red', lwd=2, lty=2)
plot(excesses, type='l')
hist(excesses)

np$save(paste0("/Users/alison/Documents/DPhil/multivariate/", var, "/", block.size, "/train/pot/thresholds.npy"), u_mat)

if(FALSE){
  summary(var_cv)
  pred <- predict(var_cv, which_u="best", n_years=1/12, type="d") # 1-month RP event distribution
  par(mfrow=c(2, 1))
  hist(x, breaks=50, probability=TRUE)
  abline(v=best_u, col='red', lty='dashed', lwd=2)
  hist(x[x>best_u], probability=TRUE, breaks=50)
  lines(pred$x, pred$y)
  abline(v=best_u, col='red', lty='dashed', lwd=2)
  mtext("Fitted Generalised Pareto for 10(?) years ", side = 3, line = 2.5)
  
}
