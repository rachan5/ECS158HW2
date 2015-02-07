#include <iostream>
#include <mpi.h>

using namespace std;


int main(int argc, char *argv[])
{
	//get size of array and pattern
	int n, m;
	cin >> n;
	cin >> m;
		
	//get array values
	int *x = new int[n];
	for (int i=0; i<n; i++)
		cin >> x[i];
	
	//int * output = numcount(x, n, m);
	//for (int i=0; i<output[0]*(m+1) + 1; i++)
	//	cout << output[i] << " ";
	//cout << endl;

	delete[] x;
	return 0;
}//main	
