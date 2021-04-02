# Title     : TODO
# Objective : TODO
# Created on: 2021/3/14


library(hts)
tourism <- read.csv("/Users/songxinyang/Documents/projects/hierarchical_forecasting/hts/experiment/data/TourismData_v4.csv")

tourism <- ts(tourism, frequency = 12)

tourism_train <- hts(window(tourism, time(tourism)[1], time(tourism)[228]), characters = c(1,1,1))
tourism_test <- hts(window(tourism, time(tourism)[229], time(tourism)[240]), characters = c(1,1,1))


fs_ols <- forecast.gts(tourism_train, h=12, method = "comb",
             weights = "ols", fmethod = "arima")



# auto.arima精度
tourism_train %>% aggts(levels = 0) %>%
  auto.arima()  %>%
  forecast(h=12) %>%
  accuracy(aggts(tourism_test, levels = 0))

# MinT精度
accuracy.gts(fs, tourism_test, levels = 3)["MASE",] %>% mean()


# OLS 精度
accuracy.gts(fs_ols, tourism_test)["MASE",]


# WLS 精度
fs_wls <- forecast.gts(tourism_train, h=12, method = "comb",
                       weights = "wls", fmethod = "arima")
accuracy.gts(fs_wls, tourism_test, levels = 1)["RMSE",] %>% mean()


# nseries
fs_s <- forecast.gts(tourism_train, h=12, method = "comb",
                       weights = "nseries", fmethod = "arima")
accuracy.gts(fs_s, tourism_test, levels = 0)["MASE",]

fs_sam <- forecast.gts(tourism_train, h=12, method = "comb",
                     weights = "mint", fmethod = "arima",
                     covariance = "sam")
accuracy.gts(fs_s, tourism_test, levels = 0)["MASE",]

