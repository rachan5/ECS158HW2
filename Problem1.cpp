#include <iostream>
#include <fstream>
#include <cstdlib>
#include <mpi.h>

using namespace std;

#define MSG 0

int NNodes, Me;
int n, m;
int *x;
int chunkIndex, chunkSize;

void init(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NNodes);
	MPI_Comm_rank(MPI_COMM_WORLD, &Me);
/*
	ifstream myFile;
	myFile.open(argv[1]);

	myFile >> n;
	myFile >> m;
		
	x = new int[n];
	for (int i=0; i<n; i++)
		myFile >> x[i];
*/
/*	
	int maxPatterns = n-m+1;
	if (Me == 0)
		chunkIndex = 0;
	else
	{
		for (int i=0; i<NNodes; i++)
		{
			chunkIndex += maxPatterns/NNodes;
			if (i < maxPatterns%NNodes)
				chunkIndex++;
		}//for
	}//else
		
	chunkSize = maxPatterns/NNodes;
	if (Me < maxPatterns%NNodes)
		chunkSize++;
*/
}//init


/*
struct entry
{
	int count;
	entry * next;
	int * pattern;
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
			hashTable[i]->next = NULL;
			hashTable[i]->pattern = NULL;
		}//for
	}//Constructor
	~Hash()
	{
		for(int i=0; i<tableSize; i++)
		{
			if (hashTable[i]->next)
				delete [] hashTable[i]->next;
			if (hashTable[i]->pattern)
				delete [] hashTable[i]->pattern;
			delete [] hashTable[i];
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
			hashTable[index]->pattern = new int[patternLength];
			for(int i=0; i<patternLength; i++)
				hashTable[index]->pattern[i] = pattern[i];
			hashTable[index]->count += c;
			numPatterns++;	
		}//if
		else if (patternMatch(hashTable[index], pattern, c))
			return;
		else
		{
			itemCount[index]++;
			entry * ptr = hashTable[index];
			
			entry * newPattern = new entry;
			newPattern->pattern = new int[patternLength];
			for (int i=0; i<patternLength; i++)
				newPattern->pattern[i] = pattern[i];
			newPattern->count += c;
			newPattern->next = NULL;
			while(ptr->next)
				ptr = ptr->next;

			ptr->next = newPattern;
			numPatterns++;
		}//else
	}//insert

};//Hash


int getChunkIndex(int n, int m, int currentThread, int numThreads)
{
	//find what chunk the thread will be working on
	if (currentThread == 0) return 0;
	unsigned int maxPatterns = n-m+1;
	int chunkIndex = 0;
	for (int i=0; i<currentThread; i++)
	{
		chunkIndex += (maxPatterns)/numThreads;
		if (i < (maxPatterns)%numThreads)
			chunkIndex++;			
	}//for
	return chunkIndex;
}//getChunkIndex


int *numcount(int *x, int n, int m)
{
	//split array into chunks and compute each chunk in parallel
	unsigned int maxPatterns = n-m+1;
	//initialize pattern table
	Hash patternTable(maxPatterns, m);
	int * outputArray;
	int outputIndex = 0;
	
	#pragma omp parallel
	{
		int currentThread = omp_get_thread_num();
		int numThreads = omp_get_num_threads();
		int chunkIndex = getChunkIndex(n, m, currentThread, numThreads);
		int chunkSize = maxPatterns/numThreads;
		if (currentThread < maxPatterns%numThreads)
			chunkSize++;

		Hash threadTable(chunkSize, m);	
	
		//each thread finds patterns in a chunk of the array
		for (int i=0; i<chunkSize; i++)
		{
			int pattern[m];
			for (int j=0; j<m; j++)
				pattern[j] = x[chunkIndex+i+j];
			threadTable.insert(pattern, 1);
		}//for
		
		#pragma omp critical
		{
			for (int i=0; i<chunkSize; i++)
			{
				if (!threadTable.getHashTable()[i]->count == 0)
				{
					entry * ptr = threadTable.getHashTable()[i];
					for (int j=0; j<threadTable.getItemCount()[i]; j++)
					{
						int * patt = ptr->pattern;
						patternTable.insert(patt, ptr->count);
						ptr = ptr->next;
					}//for
				}//if
			}//for
		}//lock		
		#pragma omp barrier
		{
			#pragma omp single
			{
				outputArray = new int[patternTable.getNumPatterns()*(m+1) + 1];		
				outputArray[outputIndex++] = patternTable.getNumPatterns();
			}//single
		}//barrier
	}//omp parallel		

	//set values of output array
	#pragma omp for schedule(dynamic)
	for (int i=0; i<maxPatterns; i++)
	{	
		if (!patternTable.getHashTable()[i]->count == 0)
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
*/

int main(int argc, char *argv[])
{
	init(argc, argv);
	int send = 9;
	int rec;
	MPI_Status status;

	cout << "before send" << endl;
	cout << Me << endl;
	MPI_Send(&send, 1, MPI_INT, 1, MSG, MPI_COMM_WORLD);
	cout << "lol" << endl;
//	MPI_Recv(&rec, 1, MPI_INT, 1, MSG, MPI_COMM_WORLD, &status);

	cout << rec << endl;
	MPI_Finalize();
/*	
	int n, m;
	cin >> n;
	cin >> m;
		
	//get array values
	int *x = new int[n];
	for (int i=0; i<n; i++)
		cin >> x[i];
	
	int * output = numcount(x, n, m);
	for (int i=0; i<output[0]*(m+1) + 1; i++)
		cout << output[i] << " ";
	cout << endl;

	delete[] x;
	return 0;
*/
}//main	
																																			
