#Authors: Group 5 Raymond S. Chan, Alicia Luu, Bryan Ng


#nth = number of threads

#xl = left limit
#xr = right limit
#yb = bottom limit
#yt = top limit
#inc = distance between ticks on X,Y axes
#maxiters = maximum number of iterations
#sched = quoted string indicating which OMP scheduling method is to be used,
#        ie. "static", "dynamic", or "guided"
#chunksize = OMP chunk size (see README)

#assume dyn.load("Problem2.so") has been called
rmandel_t <- function(nth,xl,xr,yb,yt,inc,maxiters,sched,chunksize){

  #.Call("cmandel", 8.0, -1.0, 1.0, -1.0, 1.0, 0.01, 100, 123, 100)
  if (sched == "static"){
    sched_num <- 1
  }
  else if (sched == "dynamic"){
    sched_num <- 2
  }
  else if (sched == "guided"){
    sched_num <- 3
  }
  else{
    print("ERROR: sched != static, dynamic or guided, please try again.\n")
  }
  z <- .Call("cmandel", nth, xl, xr, yb, yt, inc, maxiters, sched_num, chunksize)
  
  g <- list()
  g$x <- seq(xl,xr,inc)
  g$y <- seq(yb,yt,inc)
  g$z <- matrix(z,nrow = abs(xl-xr)/inc, ncol = abs(yb-yt)/inc)
  #print the image to .png
  png("HW2P2Mandelplot.png")
  image(g)
  dev.off()
}
#NOTE: system.time() should be called here to display elapsed time.
#TODO: start system timer somewhere

rmandel <- function(nth,xl,xr,yb,yt,inc,maxiters,sched,chunksize){
  system.time(rmandel_t(nth,xl,xr,yb,yt,inc,maxiters,sched,chunksize))
}

