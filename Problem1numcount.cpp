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
		itemCount = new int[tableSize];	
		hashTable = new entry*[tableSize];
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
		//we found this on the internet - pjw hash
		unsigned int h, g;
		for (int i=0; i<patternLength; i++)
		{
			unsigned int patternIndex = pattern[i];
			h = (h << 4) + patternIndex;
			g = h & 0xf0000000;
			if (g != 0)
			{
				h = h ^ (g >> 24);
				h = h ^ g;
			}//if
		}//for
		return (h%tableSize);
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


bool isPrime(int num)
{
	if (num < 2)
		return false;
	else if (num == 2)
		return true;
	if (num%2 == 0)
		return false;
	for (int i=3; i*i<=num; i+=2)
		if (num%i == 0)
			return false;
	return true;
}//isPrime


int findPrime(int num)
{
	//return prime bigger than num
	if (num < 2)
		num = 2;
	for (int i=num; i<num*2; i++)
	{
		if (isPrime(i))
			return (i);	
	}//for	
}//for


int * numcount(int * x, int n, int m)
{	
	init();
	int maxPatterns = n-m+1;
	int chunkIndex = 0;
	int chunkSize = maxPatterns/(nnodes*1.5); 
	int * outputArray = NULL;
	int outputIndex = 0;

	//find chunk index
	for (int i=0; i<me; i++)
		chunkIndex += maxPatterns/(nnodes*1.5);

	//weight chunk size
	if (me == nnodes-1)
		chunkSize = maxPatterns - chunkSize*(nnodes-1);

	//receive previous node's size
	unsigned int recvSize = 0;
	MPI_Status status;	
	if (me != 0) //first node has no node before it
		MPI_Recv(&recvSize, 1, MPI_UNSIGNED, me-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	//create hash table of size chunkSize plus previous nodes size
	int tableSize = findPrime(chunkSize+recvSize);
	Hash nodeTable(tableSize, m);	

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
		unsigned int recvData[2]; //contains foundIndex and count
		for (int i=0; i<recvSize; i++)
		{
			MPI_Recv(&recvData, 2, MPI_UNSIGNED, me-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			
			for (int j=0; j<m; j++)
				pattern[j] = x[recvData[0]+j];
	
			nodeTable.insert(pattern, recvData[1], recvData[0]); //insert takes pattern, count, and index found
		}//for
	}//if
	
	//only worker nodes send to next node
	if (me != nnodes-1)
	{
		//send size to next node
		unsigned int sendSize = nodeTable.getNumPatterns();
		MPI_Send(&sendSize, 1, MPI_UNSIGNED, me+1, PIPE_MSG, MPI_COMM_WORLD);

		//send patterns to next node
		unsigned int sendData[2]; //contains foundIndex and count
		for (int i=0; i<tableSize; i++)
		{	
			if (nodeTable.getHashTable()[i]->count != 0)		
			{
				entry * ptr = nodeTable.getHashTable()[i];
				for (int j=0; j<nodeTable.getItemCount()[i]; j++)
				{
					sendData[0] = ptr->foundIndex;
					sendData[1] = ptr->count;
					MPI_Send(&sendData, 2, MPI_UNSIGNED, me+1, PIPE_MSG, MPI_COMM_WORLD);
					ptr = ptr->next;
				}//for
			}//if
		}//for
	}//if

	//output array
	if (me == nnodes-1)
	{
		outputArray = new int[nodeTable.getNumPatterns()*(m+1) + 1];
		outputArray[outputIndex++] = nodeTable.getNumPatterns();
		//set values of output array
		for (int i=0; i<tableSize; i++)
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

/*
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
//		for (int i=0; i<output[0]*(m+1) + 1; i++)
//			cout << output[i] << " ";
// 		cout << endl;
		delete [] output;
	}//if
	delete [] x;
}//main
*/
