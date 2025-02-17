library(dplyr)
library(FoReco)
setwd("examples/tourism")

dt <- read.csv("data/tourism.csv")
struc <- dt[, 1:4]
dt <- unname(as.matrix(t(dt[, 5:232])))
S <- read.csv("data/tourism_S.csv") %>%
  select(-X) %>%
  as.matrix() %>%
  unname()

allts <- dt %*% t(S)
tts <- allts[217:228, c(1, 252:555)]

# For speed, only total and bottom level are used in this example
bf <- lapply(c(1, 2, 3, 4, 6, 12), function(i) {
  dt <- read.csv(sprintf("data/tourism_baseforecast_%s.csv", i)) %>%
    as.matrix() %>%
    unname()
  dt[1:(12 / i), c(1, 252:555), drop = FALSE]
})
resids <- lapply(c(1, 2, 3, 4, 6, 12), function(i) {
  (read.csv(sprintf("data/tourism_residuals_%s.csv", i)) %>%
    as.matrix() %>%
    unname())[, c(1, 252:555)]
})

# prepare base forecasts and residuals into FoReco::octrec required format
bf <- do.call(rbind, bf[6:1]) %>% t()
resids <- do.call(rbind, resids[6:1]) %>% t()
m <- 12
C <- S[1, , drop = FALSE]


output <- list()
method <- "ols"
for (method in c("ols", "struc", "wlsh", "wlsv")) {
  recf <- octrec(bf, m, C, method, resids, keep = "recf")[2:305,17:28] %>% t()
  output[[method]] <- sqrt(mean((recf - tts[,2:305])^2))
}


data.frame(output) %>%
  write.csv("data/tourism_crosstemporal.csv", row.names = FALSE)
