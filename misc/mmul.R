mmul <- function(cls,u,v){
  rowgrps <- splitIndices(nrow(u), length(cls))
  grpmul <- function(grp) u[grp,] %*% v
  mout <- clusterApply(cls,rowgrps,grpmul)
  Reduce(c,mout)
}
