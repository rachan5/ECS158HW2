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

rmandel(nth,xl,xr,yb,yt,inc,maxiters,sched,chunksize)

#print the image to .png
#png(HW2P2Mandelplot.png)
#image(g)
#dev.off()

#NOTE: system.time() should be called here to display elapsed time.
#TODO: start system timer somewhere
