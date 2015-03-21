library(parallel)
source("groupquiz.R")
c2 <- makePSOCKcluster(rep("localhost",2))
f <- function(x) {sin(x*3*pi)}
findroots(c2,f,3,100)
