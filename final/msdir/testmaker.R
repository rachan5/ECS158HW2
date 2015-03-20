#enables command like arguments for RScript
args <- commandArgs(TRUE)
library(phylobase)
library(methods)
#TODO: change to accept input file
s <- ape::read.tree(args[1])
#ensure labels are all unique
#NOTE: tree format is of object type phylo4d !!!
t <- as(s,"phylo4d")
possible_labels <- unique(replicate(nNodes(t), paste(sample(LETTERS, 7, replace=FALSE), collapse=""))) 
while(length(possible_labels) != nNodes(t))
{
    possible_labels <- unique(replicate(nNodes(t), paste(sample(LETTERS, 7, replace=FALSE), collapse=""))) 
}
labels <- possible_labels
nodeLabels(t) <-labels
#NOTE: tree format is of object type phylo !!!
ape::write.tree(as(t,"phylo"), file = args[2])
