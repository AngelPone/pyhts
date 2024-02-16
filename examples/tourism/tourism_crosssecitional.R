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

output <- list()

for (method in c("ols", "struc", "wls", "shr")) {
  output[[method]] <- sapply(1:6, function(i) {
    bf <- bf[[i]]
    resids <- resids[[i]]
    recf <- htsrec(bf, method, S[1:(NROW(S) - NCOL(S)),], res = resids)$recf
    alldt <- apply(dt[217:228,], 2, function(x){
      k <- c(1,2,3,4,6,12)[i]
      apply(matrix(x, ncol=k, byrow=TRUE), 1, sum)
    })
    sqrt(mean((recf[,252:555] - alldt)^2))
})
}

recf_rmse = data.frame(output)

write.csv(recf_rmse, "data/tourism_crosssectional.csv")
