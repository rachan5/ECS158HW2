/*
  ancestors.c:
    Identify all ancestors of each node in the input vector. Function
  inputs are derived from a phylo4 edge matrix, which *must* be in
  postorder order. The isAncestor output is an indicator matrix of
  which nodes (rows, corresponding to the decendant vector) are
  ancestors of each input node (columns, corresponding to the nodes
  vector). It will contain 1 for each ancestor of the node, *including
  itself*, and 0 for all other nodes.

  Jim Regetz (NCEAS)
*/

#include <R.h>
#include <Rinternals.h>
#include "omp.h"

int num_thread=8;
OMP_SET_NUM_THREAD(num_thread);

SEXP ancestors(SEXP nod, SEXP anc, SEXP des) {

    int numEdges = length(anc);
    int numNodes = length(nod);

    int* nodes = INTEGER(nod);
    int* ancestor = INTEGER(anc);
    int* descendant = INTEGER(des);

    int parent = 0;
    SEXP isAncestor;

    PROTECT(isAncestor = allocMatrix(INTSXP, numEdges, numNodes));
 #pragma omp parallel for collapse(2)
    for (int n=0; n<numNodes; n++) {
	//#pragma omp parallel for
        for (int i=0; i<numEdges; i++) {
            if (nodes[n]==descendant[i]) {
                INTEGER(isAncestor)[i + n*numEdges] = 1;
            } else {
                INTEGER(isAncestor)[i + n*numEdges] = 0;
            }
        }
    }

#pragma omp parallel for collapse(2)
    for (int n=0; n<numNodes; n++) {
	//#pragma omp parallel for
        for (int i=0; i<numEdges; i++) {
            if (INTEGER(isAncestor)[i + n*numEdges]==1) {
                parent = ancestor[i];
                for (int j=i+1; j<numEdges; j++) {
                    if (descendant[j]==parent) {
                        INTEGER(isAncestor)[j + n*numEdges]=1; 
                    }
                }
            }
        }
    }


    UNPROTECT(1);
    return isAncestor;
}
