#f <- function(x){ sin(pi*x*3)}
#processNOW <- list(c(0,1))
#nsubints <- 3
#processLATER <- list()

setmyid <- function(i){
  myid <<- i
}

setinitalinterval <- function(){
  mylow <<- (myid - 1) * chunksize
  myhigh <<-  myid * chunksize
  processNOW <<- list(c(mylow,myhigh))
  processLATER <<- list()
}

work <- function(){
  while(length(processNOW) > 0)
  {
    #pop element off PROCESS-NOW QUEUE
    x <- processNOW[[1]]
    if (length(processNOW) <= 1)
    {
      processNOW <<- list()
    }
    else
    {
      #FUCK THIS FOLLLOWING SYNTAX
      #IT GETS RID OF THE FIRST ELEMENT OF THE LIST
      processNOW <<- processNOW[-1]
    }#check for sign change
    #if HIT then append cutted intervals to the PROCESS-LATER QUEUE
    if ( sign(f(x[1])) != sign(f(x[2])) )
    {
      for (t in 1:nsubints)
      {
        newchunksize <- (x[2] - x[1])/nsubints
        start <- x[1]
        processLATER <<- c(processLATER,list(c(start+(t-1)*newchunksize,start+t*newchunksize))) 
      }
    }##if
  }##while
  #when process now queue is empty,
  #swap PROCESS-NOW QUEUE with PROCESS-LATER QUEUE
  processNOW <<- processLATER
  #RESET FUCKING processLATER
  processLATER <<- list()
  #print(processNOW)
  processNOW
}## work

findroots <- function(cls, f, nsubints, niters)
{
  ncls <- length(cls)
  chunksize <- 1/ncls
  #pass objects needed 
  clusterExport(cls, c("f","nsubints","chunksize","work", "setmyid", "setinitalinterval"), envir=environment())
  #pass first cut of intervals to the nodes
  clusterApply(cls,1:ncls,setmyid)
  clusterEvalQ(cls,setinitalinterval())
  #for each iteration
  for (i in 1:(niters-1))
  {
    #call the work function on PROCESS-NOW QUEUE
    clusterEvalQ(cls,work())
  }
  test <- clusterEvalQ(cls,work())
  #attempt to get rid of repeats, but they are floats...
  unique(Reduce(c,test))
}

