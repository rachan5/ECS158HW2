# ECS 158 Final Project #

We will be trying to parallel-ize the shortestPath() function. Ray suggests we start with the ancestor() function.

**Ray** is handling the **RSnow** implementation.

**Bryan** is handling the **CUDA** implementation

**Alicia** is handling the **OpenMP** implementation.

------

### Files of concern: ###
phylobase/R contains the R source code (take a look at treewalk.R)

phylocase/src contain the C/C++ source code (take a look at ancestors.c)

------
### List of things to do (in no particular order): ###

* Figure out how to *assemble code in a formal R package, that is, it must build on CSIF using "R CMD INSTALL" command*.

* Find a LARGE test cases for time comparison between the different implementations.

* Start documenting references used 

* Start writing .tex file for report

------

#### To install on CSIF:  ####

While in R, run 

	install.packages("phylobase")

Type in 87 for the mirror you want to use, and let it install (it takes a couple of minutes)

To test your installation (still in R command prompt),run

	library(phylobase)
	data(geospiza)
	nodeLabels(geospiza) <- LETTERS[1:nNodes(geospiza)]
	## NOTE: Plots don't appear while on CSIF
	plot(as(geospiza, "phylo4"), show.node.label=TRUE)
	ancestor(geospiza, "E")
	children(geospiza, "C")
	descendants(geospiza, "D", type="tips")
	descendants(geospiza, "D", type="all")
	ancestors(geospiza, "D")
	MRCA(geospiza, "conirostris", "difficilis", "fuliginosa")
	MRCA(geospiza, "olivacea", "conirostris")
	## shortest path between 2 nodes
	shortestPath(geospiza, "fortis", "fuliginosa")
	shortestPath(geospiza, "F", "L")
	## branch length from a tip to the root
	sumEdgeLength(geospiza, ancestors(geospiza, "fortis", type="ALL"))