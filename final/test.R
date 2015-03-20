dyn.load("shortestPath.so")
library(phylobase)
data(geospiza)

rCudaShortestPath <- function(phy, node1, node2)
{
  nodeID = nodeId(phy)
  ancestor = phy@edge[,1]
  label = labels(phy)
  n1 = names(getNode(phy,node1))
  n2 = names(getNode(phy,node2))
  z <- .Call("cudaShortestPath", nodeID, ancestor, label, n1, n2)
  return (z)
}

rCudaShortestPath(geospiza, "fusca", "N20")
