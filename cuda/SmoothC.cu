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

	entry temp[high-low+1];

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
	entry array[n];
	for (int i=0; i<n; i++)
	{
		entry temp;
		temp.origIndex = i;
		temp.xValue = x[i];
		temp.yValue = y[i];
		array[i] = temp;
	}//for
	
	mergeSort(array, 0, n-1);

	for (int i=0; i<n; i++)
	{
	}//for
}//smoothc


int main()
{
	int n = 1000;
	float x[n];
	float y[n];
	float m[n];
	float h = 2;
	
	for (int i=0; i<n; i++)
	{
		x[i] = rand() % 100;
		y[i] = rand() % 100;
	}//for

	smoothc(x, y, m, n, h);
}//main
