#include <stdio.h>
#include <stdlib.h>
#include <R.h>
#include <Rinternals.h>
#include <cuda.h>

//nvcc -c shortestPath.cu -Xcompiler "-fpic" -I/usr/include/R
//R CMD SHLIB shortestPath.o -o shortestPath.so

struct node
{
	int nodeID, ancestor;
	char label[20];
	//nodeType - 0 = root, 1 = internal, 2 = tip
};//node

void setNode(node &phy, int numNodes, int id, int aID, const char * label)
{
	phy.nodeID = id;
	phy.ancestor = aID;
	memset(phy.label, '\0', sizeof(label));
	strcpy(phy.label, label);
}//setNode


__global__ void kernel(node * array, int numNodes, int id, int * ancestorID)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
  if (idx < numNodes)
	{
		int ancestorIndex = 0;
  	for(int i=0; i<numNodes; i++)
		{
			if (array[i].nodeID == id)
			{
				node temp = array[i];
				while (temp.ancestor != 0)
				{
					ancestorID[ancestorIndex++] = temp.ancestor;
					for (int j=0; j<numNodes; i++)
					{
						if (array[j].nodeID == temp.ancestor)
						{
							temp = array[j];
							break;
						}//if
					}//for
				}//while
			}//if
		}//for
	}//if
/*	else if (array[idx].nodeID == id2)
		{
			node temp = array[idx];
			while (temp.ancestor != 0)
			{
				for (int i=0; i<numNodes; i++)
				{
					if (ancestorID2[i] != 0)
						continue;
					ancestorID2[i] = temp.ancestor;
				}//for
				for (int i=0; i<numNodes; i++)
				{
					if (array[i].nodeID == temp.ancestor)
					{
						temp = array[i];
						break;
					}//if
				}//for
			}//while
		}//if
	}//if
*/
}//kernel


int * shortestPath(node * phy, int numNodes, const char * label1, const char * label2)
{
	node * deviceArray;
	int * deviceID;
	int * ancestorID1 = new int[numNodes];
	int * ancestorID2 = new int[numNodes];
	float blockSize = 1024; //num threads per block

	//check if invalid query
	node temp1, temp2;
	for (int i=0; i<numNodes; i++)
	{
		ancestorID1[i] = 0;
		ancestorID2[i] = 0;
		if (strcmp(label1, phy[i].label) == 0)
			temp1 = phy[i];
		else if (strcmp(label2, phy[i].label) == 0)
			temp2 = phy[i];
	}//for

	if ((temp1.ancestor == temp2.nodeID) || (temp2.ancestor == temp1.nodeID))
	{
		return 0;
	}//if

	dim3 dimBlock(blockSize);
	dim3 dimGrid(ceil(numNodes/blockSize));

	cudaMalloc(&deviceArray, sizeof(node) * numNodes);
	cudaMalloc(&deviceID, sizeof(int) * numNodes);
	cudaMemcpy(deviceArray, phy, sizeof(node) * numNodes, cudaMemcpyHostToDevice);	
	cudaMemcpy(deviceID, ancestorID1, sizeof(int) * numNodes, cudaMemcpyHostToDevice);

  kernel <<< dimBlock, dimGrid >>> (deviceArray, numNodes, temp1.nodeID, deviceID);
	cudaMemcpy(ancestorID1, deviceID, sizeof(int) * numNodes, cudaMemcpyDeviceToHost);
  
	int ancestorIndex = 0;
  for(int i=0; i<numNodes; i++)
	{
		if (phy[i].nodeID == temp1.nodeID)
		{
			node temp = phy[i];
			while (temp.ancestor != 0)
			{
				ancestorID1[ancestorIndex++] = temp.ancestor;
				for (int j=0; j<numNodes; i++)
				{
					if (array[j].nodeID == temp.ancestor)
					{
						temp = phy[j];
						break;
					}//if
				}//for
			}//while
		}//if
	}//for
	
	ancestorIndex = 0;
  for(int i=0; i<numNodes; i++)
	{
		if (phy[i].nodeID == temp2.nodeID)
		{
			node temp = phy[i];
			while (temp.ancestor != 0)
			{
				ancestorID2[ancestorIndex++] = temp.ancestor;
				for (int j=0; j<numNodes; i++)
				{
					if (array[j].nodeID == temp.ancestor)
					{
						temp = phy[j];
						break;
					}//if
				}//for
			}//while
		}//if
	}//for
	cudaFree(deviceArray);
	cudaFree(deviceID);

	//find shortest path
	int * path = new int[numNodes];
	int currentPath = ancestorID1[0];
	int pathIndex = 0;
	bool isLCAPath = false;
	for (int i=0; i<numNodes; i++)
	{
		path[i] = 0;
		if (temp1.nodeID == ancestorID2[i])
		{
			for (int j=0; j<i; j++)
				path[j] = ancestorID2[j];
			isLCAPath = true;
			break;
		}//if	
		else if (temp2.nodeID == ancestorID1[i])
		{	
			for (int j=0; j<i; j++)
				path[j] = ancestorID1[j];
			isLCAPath = true;
			break;
		}//else if
	}//for

	if (!isLCAPath)
	{
		for(int i=0; i<numNodes; i++)
		{	
			for (int j=0; j<numNodes; j++)
			{
				if (currentPath == ancestorID2[j])
					break;
				if ((ancestorID2[j] == 0) || (j == numNodes-1))
				{
					path[pathIndex++] = ancestorID1[i];
					currentPath = ancestorID1[i];
					break;
				}//if
			}//for	
		}//for		

		if (pathIndex == 0)	
			path[pathIndex++] = currentPath;
	
		for (int i=0; i<numNodes; i++)
		{
			if (ancestorID2[i] == currentPath)
				break;
			path[pathIndex++] = ancestorID2[i];
		}//for
	}//if

  for(int i =0; i < numNodes; i++ ){
    printf("%d %d\n", ancestorID1[i], ancestorID2[i]);
    //printf("%d %d %s\n", phy[i].nodeID, phy[i].ancestor, phy[i].label);
  }

/*	
	for (int i=0; i<numNodes; i++)
	{
		if (path[i] == 0)
			break;
		for (int j=0; j<numNodes; j++)
		{
			if (path[i] == phy[j].nodeID)
			{
				printf("%s ", phy[j].label);
				break;
			}//if
		}//for
	}//for

	printf("\n");
	for (int i=0; i<numNodes; i++)
	{
		if (path[i] == 0)
			break;
		printf("%d ", path[i]);	
	}//for
	printf("\n");
*/
	delete [] ancestorID1;
	delete [] ancestorID2;
	//delete [] path;
	return path;
}//shortestPath


extern "C" SEXP cudaShortestPath(SEXP nodeIDs, SEXP nodeAncestors, SEXP nodeLabels, SEXP n1, SEXP n2)
{
	nodeIDs = coerceVector(nodeIDs, INTSXP);
	nodeAncestors = coerceVector(nodeAncestors, INTSXP);
	nodeLabels = coerceVector(nodeLabels, STRSXP);
	n1 = coerceVector(n1, STRSXP);
	n2 = coerceVector(n2, STRSXP);
	
	int numNodes = length(nodeIDs);
	node * phy = new node[numNodes];
	
	for (int i=0; i<numNodes; i++){
		//printf("%d %d %s\n", INTEGER(nodeIDs)[i], INTEGER(nodeAncestors)[i], CHAR(STRING_ELT(nodeLabels,i)));
    setNode(phy[i], numNodes, INTEGER(nodeIDs)[i], INTEGER(nodeAncestors)[i], CHAR(STRING_ELT(nodeLabels, i)));
  }
	//test shortest path
	SEXP Rval;
	PROTECT(Rval = allocVector(INTSXP, numNodes));
	//printf("%s\n ", CHAR(STRING_ELT(n1,0)));
  //printf("%d\n", numNodes);
  int * path = shortestPath(phy, numNodes, CHAR(STRING_ELT(n1, 0)), CHAR(STRING_ELT(n2, 0))); 
	
	for (int i=0; i<numNodes; i++){
		//printf("%d\n", path[i]);
    INTEGER(Rval)[i] = path[i];
  }
	delete [] phy;
	delete [] path;
	UNPROTECT(1);
	return Rval;
}//main

