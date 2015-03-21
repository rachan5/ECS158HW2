args <- commandArgs(TRUE)
library(phylobase)
library(methods)
#t <- data(geospiza)
dyn.load("OMPancestors.so")
dyn.load("OMPdescendants.so")
source('OMPtreewalk.R')

#plot(as(geospiza,"phylo4"), show.node.label=TRUE)
s <- ape::read.tree(file=args[1])
t <- as(s,"phylo")

for (i in 1:5){
    #x <- sample(labels(geospiza),2,replace=TRUE)
    x <- sample(t$node.label,2)
    #print(class(t))
    #print(t$node.label)
    cat("real answer",x[1], x[2],"\n")
    #print(shortestPath(t, x[1], x[2]))
    #print(shortestPath(geospiza, x[1], x[2]))
    print(system.time(shortestPath(t, x[1], x[2])))
    #print(system.time(shortestPath(geospiza, x[1], x[2])))
    cat("OMP", x[1], x[2],"\n")
    #print(OMPshortestPath(t, x[1], x[2]))
    #print(OMPshortestPath(geospiza, x[1], x[2]))
    print(system.time(OMPshortestPath(t, x[1], x[2])))
    #print(system.time(OMPshortestPath(geospiza, x[1], x[2])))
    cat("\n\n")
}

dyn.unload("OMPancestors.so")
dyn.unload("OMPdescendants.so")

