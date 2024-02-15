library(dplyr)
library(foreach)

# load dataset
dt <- read.csv("data/tourism.csv")
struc <- dt[,1:4]
dt <- unname(as.matrix(t(dt[,5:232])))
S <- read.csv("data/tourism_S.csv") %>% 
  select(-X) %>%
  as.matrix() %>%
  unname()

allts <- dt %*% t(S)

# generate base forecasts
bf_cs <- foreach(series=iterators::iter(allts, by="column"), 
              .packages = "forecast") %do% {
  train_length <- length(series) - 12
  train_ts <- ts(dt[1:train_length], frequency = 12, start = c(1998, 1))
  print(train_length)
  mdl <- auto.arima(train_ts)
  list(
    fcasts = as.numeric(forecast(mdl, h=12)$mean),
    residuals = as.numeric(residuals(mdl, type = "response"))
  )
}

cs_bf <- lapply(bf_cs, function(x){x$fcasts}) %>%
  do.call(cbind, .)
cs_resids <- lapply(bf_cs, function(x){x$residuals}) %>%
  do.call(cbind, .)

write.csv(cs_bf, "data/tourism_baseforecast_1.csv", row.names = FALSE)
write.csv(cs_resids, "data/tourism_residuals_1.csv", row.names = FALSE)



for (i in c(2, 3, 4, 6, 12)) {
  print(i)
  bf <- foreach(series=iterators::iter(allts, by="column"),
          .packages = "forecast") %do% {
    train_length <- length(series) - 12
    print(train_length)
    train_ts <- matrix(series[1:train_length], ncol=i, byrow=TRUE)
    train_ts <- ts(apply(train_ts, 1, sum), frequency = 12 / i)
    mdl <- auto.arima(train_ts)
    list(
      fcasts = as.numeric(forecast(mdl, h=12/i)$mean),
      residuals = as.numeric(residuals(mdl, type="response"))
    )
  }
  cs_bf <- lapply(bf, function(x){x$fcasts}) %>%
    do.call(cbind, .)
  cs_resids <- lapply(bf, function(x){x$residuals}) %>%
    do.call(cbind, .)
  write.csv(cs_bf, sprintf("data/tourism_baseforecast_%s.csv", i), row.names = FALSE)
  write.csv(cs_resids, sprintf("data/tourism_residuals_%s.csv", i), row.names = FALSE)
}

# Temporal hierarchy



