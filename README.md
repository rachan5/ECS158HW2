# ECS158HW2

Winter 2015 Team We Love Illias 2.5.5

## Problem1
To compile on CSIF:

	/usr/lib64/mpich/bin/mpic++ Problem1.cpp

To run on CSIF:

	/usr/lib64/mpich/bin/mpiexec -n 8 a.out

**Bryan- What's the current status of the program?**

Currently, all nodes take a chunk, process it and pass it to the last node. This is very slow I think because sends and receives are expensive. So, some nodes need to receive from others and all the burden cant be on the last node.

##Problem 2

***Ray's current idea is to mirror what Matloff's doing in Quiz 2'R mandelbrot function and use OpenMP instead of RSnow***

### **Scheduling clauses** [(copied from wiki)](http://en.wikipedia.org/wiki/OpenMP#Scheduling_clauses)

*schedule(type, chunk)*: This is useful if the work sharing construct is a do-loop or for-loop. The iteration(s) in the work sharing construct are assigned to threads according to the scheduling method defined by this clause. The three types of scheduling are:

1. *static*: Here, all the threads are allocated iterations before they execute the loop iterations. The iterations are divided among threads equally by default. However, specifying an integer for the parameter chunk will allocate chunk number of contiguous iterations to a particular thread.

2. *dynamic*: Here, some of the iterations are allocated to a smaller number of threads. Once a particular thread finishes its allocated iteration, it returns to get another one from the iterations that are left. The parameter chunk defines the number of contiguous iterations that are allocated to a thread at a time.

3. *guided*: A large chunk of contiguous iterations are allocated to each thread dynamically (as above). The chunk size decreases exponentially with each successive allocation to a minimum size specified in the parameter chunk

###**Interfacing between R and C** [(summary of this)](http://www.biostat.jhsph.edu/~bcaffo/statcomp/files/dotCall.pdf)

#### cd into Problem2/InterfaceRC

#### To compile:

	R CMD SHLIB vecSum.c
	or
	R CMD SHILB ab.c

#### To run (in R):

	> .Call("vecSum", c(1,2,3))
	The value is: 6.000000
	NULL

	> dyn.load("ab.so")
	> .Call("ab", 1, 5)
	[1] 1 2 3 4 5


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
