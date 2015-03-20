rCudaShortestPath <- function(phy, node1, node2)
{
  nodeID = nodeId(phy)
  ancestor = phy@edge[,1]
  label = labels(phy)
  n1 = names(node1)
  n2 = names(node2)
  z <- .Call("cudaShortestPath", nodeID, ancestor, label, n1, n2)
  return (z)
}

rCudaShortestPath(geospiza, "fusca", "fortis")