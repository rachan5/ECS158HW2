#include <iostream>
#include <fstream>
#include <cstdlib>
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
		for (int i=0; i<patternLength; i++)
			index += pattern[i]*(i+1);
		return (index%tableSize);
	}//hashFunction

	void insert(int * pattern, int c)
	{
		int index = hashFunction(pattern);
		if (hashTable[index]->count == 0)
		{
			itemCount[index] = 1;
			hashTable[index]->count += c;
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


void nodeWorker(int * x, int n, int m)
{
	int maxPatterns = n-m+1;
	int chunkIndex = 0;
	int chunkSize = maxPatterns/nnodes;

	//find chnunk index
	for (int i=0; i<me; i++)
	{
		chunkIndex += (maxPatterns)/nnodes;
		if (i < (maxPatterns)%nnodes)
			chunkIndex++;			
	}//for

	//find chunk size	
	if (me < maxPatterns%nnodes)
		chunkSize++;

	//chunkSize = chunkSize * (me+1);
	
	Hash nodeTable(chunkSize, m);	

	//count patterns
	int pattern[m];
	for (int i=0; i<chunkSize; i++)
	{
		for (int j=0; j<m; j++)
			pattern[j] = x[chunkIndex+i+j];
		nodeTable.insert(pattern, 1);
	}//for


	int srData[m+1]; //send and receive data
	if (me != 0)
	{
		MPI_Status status;
		while (1)
		{			
			MPI_Recv(&srData, m+1, MPI_INT, me-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == END_MSG)
				break;

			for(int i=0; i<m; i++)
				pattern[i] = srData[i];

			nodeTable.insert(pattern, srData[m]);
		}//while
	}//if	

	//send patterns to other nodes
	for(int i=0; i<chunkSize; i++)
	{
		if (nodeTable.getHashTable()[i]->count != 0)		
		{
			entry * ptr = nodeTable.getHashTable()[i];
			for (int j=0; j<nodeTable.getItemCount()[i]; j++)
			{
				for (int k=0; k<m; k++)
					srData[k] = ptr->pattern[k];
				srData[m] = ptr->count;
				MPI_Send(&srData, m+1, MPI_INT, me+1, PIPE_MSG, MPI_COMM_WORLD);
				ptr = ptr->next;
			}//for
		}//if
	}//for

	//send end message
	MPI_Send(&srData, m+1, MPI_INT, me+1, END_MSG, MPI_COMM_WORLD);
}//nodeWorker


int * numcount(int * x, int n, int m)
{	
	int maxPatterns = n-m+1;
	int chunkIndex = 0;
	int chunkSize = maxPatterns/nnodes;
	int * outputArray;
	int outputIndex = 0;

	//find chnunk index
	for (int i=0; i<me; i++)
	{
		chunkIndex += (maxPatterns)/nnodes;
		if (i < (maxPatterns)%nnodes)
			chunkIndex++;			
	}//for

	//find chunk size	
	if (me < maxPatterns%nnodes)
		chunkSize++;
	
	Hash patternTable(maxPatterns, m);	
	
	//count patterns
	int pattern[m];
	for (int i=0; i<chunkSize; i++)
	{
		for (int j=0; j<m; j++)
			pattern[j] = x[chunkIndex+i+j];
		patternTable.insert(pattern, 1);
	}//for

	int recvData[m+1];
	int recvPattern[m];
	MPI_Status status;

//	for (int i=0; i<nnodes-1; i++)
//	{
		while (1)
		{			
			MPI_Recv(&recvData, m+1, MPI_INT, me-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			if (status.MPI_TAG == END_MSG)
				break;

			for (int j=0; j<m; j++)
				recvPattern[j] = recvData[j];			

			patternTable.insert(recvPattern, recvData[m]);
		}//while
//	}//for
	
	outputArray = new int[patternTable.getNumPatterns()*(m+1) + 1];
	outputArray[outputIndex++] = patternTable.getNumPatterns();
	//set values of output array
	for (int i=0; i<maxPatterns; i++)
	{
		if (patternTable.getHashTable()[i]->count != 0)
		{
			entry * ptr = patternTable.getHashTable()[i];
			for (int j=0; j<patternTable.getItemCount()[i]; j++)
			{
				for (int k=0; k<m; k++)
					outputArray[outputIndex++] = ptr->pattern[k];
				outputArray[outputIndex++] = ptr->count;
				ptr = ptr->next;
			}//for
		}//if
	}//for

	return outputArray;
}//numcount


void init(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nnodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &me);
}//init


int main(int argc, char *argv[])
{	
	int n, m;
	ifstream myFile(argv[1]);
	myFile >> n;
	myFile >> m;

	int * x = new int[n];
	for (int i=0; i<n; i++)
		myFile >> x[i];
	
	init(argc, argv);
	if (me == nnodes-1)
	{
		int * output = numcount(x, n, m);	
//		for (int i=0; i<output[0]*(m+1) + 1; i++)
//			cout << output[i] << " ";
// 		cout << endl;
	}//if
	else
		nodeWorker(x, n, m);

	MPI_Finalize();
	delete [] x;
}//main	
