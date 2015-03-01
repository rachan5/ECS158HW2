#include <stdio.h>
#include <stdlib.h>

struct entry
{
	int origIndex;
	float xValue;
	float yValue;
};//entry

void merge(entry * a, int low, int high)
{
	int pivot = (low+high)/2;
	int i = 0;
	int j = low;
	int k = pivot+1;

	entry * temp = new entry[high-low+1];

	while ((j <= pivot) && (k <= high))
	{
		if (a[j].xValue < a[k].xValue)
			temp[i++] = a[j++];
		else
			temp[i++] = a[k++];
	}//while

	while (j <= pivot)
		temp[i++] = a[j++];

	while (k <= high)
		temp[i++] = a[k++];

	for (int h=low; h<= high; h++)
		a[h] = temp[h-low];
	
	delete [] temp;
}//merge

void mergeSort(entry * a, int low, int high)
{
	int pivot;
	if (low < high)
	{
		pivot = (low+high)/2;
		mergeSort(a, low, pivot);
		mergeSort(a, pivot+1, high);
		merge(a, low, high);
	}//if
}//mergeSort


void smoothc(float * x, float * y, float * m, int n, float h)
{
	entry * array = new entry[n];
	for (int i=0; i<n; i++)
	{
		entry temp;
		temp.origIndex = i;
		temp.xValue = x[i];
		temp.yValue = y[i];
		array[i] = temp;
	}//for
	
	mergeSort(array, 0, n-1);

	delete [] array;
}//smoothc


int main()
{
	int n = 2000000;
	float * x = new float[n];
	float * y = new float[n];
	float * m = new float[n];
	float h = 2;
	
	for (int i=0; i<n; i++)
	{
		x[i] = rand() % 100;
		y[i] = rand() % 100;
	}//for

	smoothc(x, y, m, n, h);
	delete [] x;
	delete [] y;
	delete [] m;
}//main
