#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

int main()
{
	int n = 2000000;
	int m = 3;
	ofstream myfile;
	myfile.open ("seq2Mil.txt");
	myfile << n << endl;
	myfile << m << endl;
	for (int i=0; i<n; i++)
		myfile << i << endl;
	myfile.close();
	return 0;
}//main
