#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

using namespace std;

//to compile: mpic++ Problem1.cpp
//to run: mpiexec -n 8 a.out

#define PIPE_MSG 0
#define END_MSG 1

int nnodes, me;

struct entry
{
	int count;
	int foundIndex;
	int * pattern;
	entry * next;
};//entry

class Hash
{
	int tableSize;
	int patternLength;
	int numPatterns;
	int * itemCount;
	entry ** hashTable;
public:
	Hash() {}	
	Hash(int maxPatterns, int m) : tableSize(maxPatterns), patternLength(m), numPatterns(0)
	{
		hashTable = new entry*[tableSize];
		itemCount = new int[tableSize];
		for (int i=0; i<tableSize; i++)
		{
			hashTable[i] = new entry;
			hashTable[i]->count = 0;
			hashTable[i]->foundIndex = 0;
			hashTable[i]->pattern = NULL;
			hashTable[i]->next = NULL;
		}//for
	}//Constructor
	~Hash()
	{
		delete [] itemCount;
		entry * temp;
		entry * tempNext;
		for(int i=0; i<tableSize; i++)
		{
			temp = hashTable[i];
			while(temp)
			{
				tempNext = temp->next;
				if (temp->pattern)
					delete [] temp->pattern;				
				delete temp;
				temp = tempNext;
			}//while
			hashTable[i] = NULL;
		}//for
		delete [] hashTable;
	}//destructor

	int getNumPatterns() { return numPatterns; }

	int * getItemCount() { return itemCount; }

	entry ** getHashTable() { return hashTable; }

	bool patternMatch(entry * hashEntry, int * pattern, int c)
	{
		entry * ptr = hashEntry;
		while (ptr)
		{
			for (int i=0; i<patternLength; i++)
			{
				if (ptr->pattern[i] != pattern[i])
					break;
				if (i == patternLength-1)
				{
					ptr->count += c;
					return true;
				}//if
			}//for
			ptr = ptr->next;
		}//while

		return false;
	}//patternMatch

	int hashFunction(int * pattern)
	{
		unsigned int index = 0;
		if (patternLength < 5)
		{	
			for (int i=0; i<patternLength; i++)
				index += pow(pattern[i], (i+1));
		}//if
		else
		{
			for (int i=0; i<5; i++)
				index += pow(pattern[i], (i+1));
			for (int i=5; i<patternLength; i++)
				index += pattern[i] * (i+1);
		}//else
		return (index%tableSize);
	}//hashFunction

	void insert(int * pattern, int c, int fIndex)
	{
		int index = hashFunction(pattern);
		if (hashTable[index]->count == 0)
		{
			itemCount[index] = 1;
			hashTable[index]->count += c;
			hashTable[index]->foundIndex = fIndex;
			hashTable[index]->pattern = new int[patternLength];
			for(int i=0; i<patternLength; i++)
				hashTable[index]->pattern[i] = pattern[i];
			numPatterns++;	
		}//if
		else if (patternMatch(hashTable[index], pattern, c))
			return;
		else
		{
			itemCount[index]++;
			entry * ptr = hashTable[index];
			
			entry * newPattern = new entry;
			newPattern->count = c;
			newPattern->foundIndex = fIndex;
			newPattern->pattern = new int[patternLength];
			for (int i=0; i<patternLength; i++)
				newPattern->pattern[i] = pattern[i];
			newPattern->next = NULL;
			while(ptr->next)
				ptr = ptr->next;

			ptr->next = newPattern;
			numPatterns++;
		}//else
	}//insert

};//Hash


void init()
{
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
}//init


int * numcount(int * x, int n, int m)
{	
	init();
	int maxPatterns = n-m+1;
	int chunkIndex = 0;
	int chunkSize = maxPatterns/nnodes;
	int * outputArray = NULL;
	int outputIndex = 0;

	//find chunk index
	for (int i=0; i<me; i++)
	{
		chunkIndex += (maxPatterns)/nnodes;
		if (i < (maxPatterns)%nnodes)
			chunkIndex++;			
	}//for

	//find chunk size	
	if (me < maxPatterns%nnodes)
		chunkSize++;

	//receive previous node's size
	int recvSize = 0;
	MPI_Status status;	
	if (me != 0) //first node has no node before it
		MPI_Recv(&recvSize, 1, MPI_INT, me-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	//creat hash table of size chunkSize plus previous nodes size
	Hash nodeTable(chunkSize+recvSize, m);	

	//count patterns
	int pattern[m];
	for (int i=0; i<chunkSize; i++)
	{
		for (int j=0; j<m; j++)
			pattern[j] = x[chunkIndex+i+j];
		nodeTable.insert(pattern, 1, chunkIndex+i); //insert takes pattern, count, and index found
	}//for

	//receive patterns from previous node
	if (me != 0)
	{
		int recvData[recvSize*2]; //contains foundIndex and count
		int patternIndex = 0;
		MPI_Recv(&recvData, recvSize*2, MPI_INT, me-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		for (int i=0; i<recvSize*2; i+=2)
		{
			for (int j=recvData[i]; j<recvData[i]+m; j++)
				pattern[patternIndex++] = x[j];
			patternIndex = 0;
			nodeTable.insert(pattern, recvData[i+1], recvData[i]); //insert takes pattern, count, and index found
		}//for
	}//if
	
	if (me != nnodes-1)
	{
		//send size to next node
		int sendSize = nodeTable.getNumPatterns();
		MPI_Send(&sendSize, 1, MPI_INT, me+1, PIPE_MSG, MPI_COMM_WORLD);
	
		//send patterns to next node	
		int sendArray[nodeTable.getNumPatterns()*2];
		int sendArrayIndex = 0;	
		for(int i=0; i<chunkSize+recvSize; i++)
		{
			if (nodeTable.getHashTable()[i]->count != 0)		
			{
				entry * ptr = nodeTable.getHashTable()[i];
				for (int j=0; j<nodeTable.getItemCount()[i]; j++)
				{
					sendArray[sendArrayIndex++] = ptr->foundIndex;
					sendArray[sendArrayIndex++] = ptr->count;
					ptr = ptr->next;
				}//for
			}//if
		}//for	
		MPI_Send(&sendArray, nodeTable.getNumPatterns()*2, MPI_INT, me+1, PIPE_MSG, MPI_COMM_WORLD);
	}//if

	//output array
	if (me == nnodes-1)
	{
		outputArray = new int[nodeTable.getNumPatterns()*(m+1) + 1];
		outputArray[outputIndex++] = nodeTable.getNumPatterns();
		//set values of output array
		for (int i=0; i<chunkSize+recvSize; i++)
		{
			if (nodeTable.getHashTable()[i]->count != 0)
			{
				entry * ptr = nodeTable.getHashTable()[i];
				for (int j=0; j<nodeTable.getItemCount()[i]; j++)
				{
					for (int k=0; k<m; k++)
						outputArray[outputIndex++] = ptr->pattern[k];
					outputArray[outputIndex++] = ptr->count;
					ptr = ptr->next;
				}//for
			}//if
		}//for
	}//if

	MPI_Finalize();
	return outputArray;
}//numcount


int main(int argc, char *argv[])
{	
	int n, m;
	ifstream myFile(argv[1]);
	myFile >> n;
	myFile >> m;

	int * x = new int[n];
	for (int i=0; i<n; i++)
		myFile >> x[i];
	
	int * output = numcount(x, n, m);	
	if (output)
	{	
		for (int i=0; i<output[0]*(m+1) + 1; i++)
			cout << output[i] << " ";
 		cout << endl;
	}//if
	delete [] x;
}//main	
