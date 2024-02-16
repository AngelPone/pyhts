library(dplyr)
setwd("examples")

dt <- read.csv("data/tourism.csv")
struc <- dt[,1:4]
dt <- unname(as.matrix(t(dt[,5:232])))
S <- read.csv("data/tourism_S.csv") %>% 
  select(-X) %>%
  as.matrix() %>%
  unname()

allts <- dt %*% t(S)

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
for (method in c("ols", "struc", "wlsv", "wlsh", "shr")) {
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