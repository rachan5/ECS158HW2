#include <iostream>
#include <cstdlib>
#include "Problem1numcount.cpp"

using namespace std;

int main()
{	
	int n = 1000000;
	int m = 3;

	int x[n];
	for (int i=0; i<n; i++)
		x[i] = i;
	
	int * output = numcount(x, n, m);	
	if (output)
	{	
//		for (int i=0; i<output[0]*(m+1) + 1; i++)
//			cout << output[i] << " ";
// 		cout << endl;
		delete [] output;
	}//if
}//main
