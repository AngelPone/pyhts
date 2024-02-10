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
bf <- lapply(c(1,2,3,4,6,12), function(i){
  dt <- read.csv(sprintf("data/tourism_baseforecast_%s.csv", i)) %>%
    as.matrix() %>%
    unname()
  dt[1:(12/i),,drop=FALSE]
})
resids <- lapply(c(1,2,3,4,6,12), function(i){
  read.csv(sprintf("data/tourism_residuals_%s.csv", i)) %>%
    as.matrix() %>%
    unname()
})

library(FoReco)
ols <- sapply(1:555, function(i) {
  bf <- do.call(c, lapply(bf[6:1], function(x) { x[,i] }))
  resids <- do.call(c, lapply(resids[6:1], function(x) { x[,i] }))
  recf <- unname(thfrec(bf, 12, "ols", res, keep = "recf")[17:28])
  sqrt(mean((recf - allts[217:228, i])^2))
})

wlss <- sapply(1:555, function(i) {
  bf <- do.call(c, lapply(bf[6:1], function(x) { x[,i] }))
  resids <- do.call(c, lapply(resids[6:1], function(x) { x[,i] }))
  recf <- unname(thfrec(bf, 12, "struc", res, keep = "recf")[17:28])
  sqrt(mean((recf - allts[217:228, i])^2))
})

recf_rmse <- list()
for (method in c("ols", "struc", "wlsv", "shr")) {
  recf_rmse[[method]] <- 
    sapply(1:555, function(i) {
      bf <- do.call(c, lapply(bf[6:1], function(x) { x[,i] }))
      resids <- do.call(c, lapply(resids[6:1], function(x) { x[,i] }))
      recf <- unname(thfrec(bf, 12, method, resids, keep = "recf")[17:28])
      sqrt(mean((recf - allts[217:228, i])^2))
    })
}

recf_rmse <- do.call(cbind, recf_rmse)

write.csv(recf_rmse, "data/tourism_temporal.csv")



