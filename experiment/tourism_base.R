library(dplyr)
library(hts)
library(foreach)
library(doParallel)



#--------------------
# Config
train_length <- 96
TOTAL_LENGTH <- 240
h <- 12
cluster_cores <- 7
data_path <- 'experiment/data/TourismData_v4.csv'
output_file <- "test.csv"


#-----------------

tourism <- read.csv(data_path)
tourism <- hts(tourism, characters = c(1,1,1))
s <- smatrix(tourism)
y <- s %*% t(tourism$bts)

cl <- parallel::makeCluster(cluster_cores)
doParallel::registerDoParallel(cl)


sf_autoarima <- function (series, h) {
  library(dplyr)
  library(forecast)
  fcasts <- series %>%
    auto.arima() %>%
    forecast(h=h)
  fcasts <- c(fcasts$fitted, fcasts$mean)
  return(fcasts)
}

all_forecast <-
  foreach(index=train_length:(TOTAL_LENGTH-1), .combine = rbind) %do%{
    foreach(i=1:dim(y)[1], .combine = rbind) %dopar%{
      series <- ts(y[i, (index-train_length+1): index], frequency = 12)
      c(i, index, sf_autoarima(series, h))
    }
  }

write.csv(data.frame(all_forecast), output_file)





