library(parallel)

c2 <- makePSOCKcluster(rep("localhost",2))

print("c2")
c2

a <- matrix(sample(1:50,16,replace=T),ncol=2)

print("a")
a

b <- c(5,-2)

print("b")
b

print("a %*%b (this is what we should when we call mmul)")
a %*% b

clusterExport(c2,c("a","b"))

print("check that c2 has a")
clusterEvalQ(c2,a)

source('mmul.R')

print("call to mmul should produce same matrix as above")
mmul(c2,a,b)

