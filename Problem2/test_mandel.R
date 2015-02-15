

source("Problem2.R")
dyn.load("Problem2.so")
#sink("test_results.txt")

for (c in c(10, 50, 100, 200, 300, 400, 500, 800, 1000)){
cat(paste("===== ", "chunksizs: " , c , " ====="))
cat("\nstatic\n")
print(rmandel(8,-1,1,-1,1,0.001,100,"static",c))
cat("dynamic\n")
print(rmandel(8,-1,1,-1,1,0.001,100,"dynamic",c))
cat("guided\n")
print(rmandel(8,-1,1,-1,1,0.001,100,"guided",c))
}


cat("\n@@@@@@@@@@@@@@@@@       @@@@@@@@@@@@@@@@@\n\n")


for (i in c(10, 50, 100, 200, 300, 400, 500, 800, 1000)){
cat(paste("===== ", "iterations: " , i , " ====="))
cat("\nstatic\n")
print(rmandel(8,-1,1,-1,1,0.001,i,"static",100))
cat("dynamic\n")
print(rmandel(8,-1,1,-1,1,0.001,i,"dynamic",100))
cat("guided\n")
print(rmandel(8,-1,1,-1,1,0.001,i,"guided",100))
}


cat("\n@@@@@@@@@@@@@@@@@       @@@@@@@@@@@@@@@@@\n\n")


for (t in c(c(1:8),16, 32, 50)){
cat(paste("===== ", "threads: " , t , " ====="))
cat("\nstatic\n")
print(rmandel(t,-1,1,-1,1,0.001,100,"static",100))
cat("dynamic\n")
print(rmandel(t,-1,1,-1,1,0.001,100,"dynamic",100))
cat("guided\n")
print(rmandel(t,-1,1,-1,1,0.001,100,"guided",100))
}

#sink()
#unlink("test_results.txt")
