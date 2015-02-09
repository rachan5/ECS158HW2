# ECS158HW2

Winter 2015 Team We Love Illias 2.5.5

## Problem1
To compile on CSIF:

	/usr/lib64/mpich/bin/mpic++ Problem1.cpp

To run on CSIF:

	/usr/lib64/mpich/bin/mpiexec -n 8 a.out

**Bryan- What's the current status of the program?**

##Problem 2

## Misc.
### To run RSnowExample.R on CSIF
	Rscript RSnowExample.R

### To run Quiz2Mandelbrot.R 
Open and type the following in R

	library(parallel)
	cls <- makePSOCKcluster(rep("localhost",2))
	source('Quiz2Mandelbrot.R')
	mandelsnow(cls,-1,1,-1,1,0.01,100)
	
A file "Quiz2Mandelbrot.png" should now appear.

Ray uses WINSCP to move the .png to my desktop and use Windows Photoviewer to open it.
