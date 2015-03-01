#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

struct entry
{
	int origIndex;
	float xValue, yValue;
};//entry


__device__ int binarySearch(entry * data, float val, int n, bool bound)
{
	//return index of xValue that is equal to or closest to value
	//bound - round up or down, T = round down, F = round up
	int left = 0;
	int right = n;
	int mid = (right-left)/2;

	while(1)
	{
		if (data[mid].xValue == val)
			return mid;

		else if (data[mid].xValue > val)
		{
			if (mid == 0)
				return mid;
			else if ((data[mid-1].xValue < val) && (!bound))
				return mid;	
			else if ((data[mid-1].xValue < val) && (bound))
				return mid-1;

			right = mid;
			mid = (right-left)/2;
		}//else if

		else if (data[mid].xValue < val)
		{
			if (mid == n-1)
				return mid;
			else if ((data[mid+1].xValue > val) && (!bound))
				return mid+1;
			else if ((data[mid+1].xValue > val) && (bound))
				return mid;

			left = mid;
			mid = (right+left)/2;
		}//else if
	}//while
}//binarySearch


__global__ void kernel(entry * array, int n, float h)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int lowerBound = binarySearch(array, array[idx].xValue-h, n, 0); //T-round down, F-round up
	int upperBound = binarySearch(array, array[idx].xValue+h, n, 1);

	//check for duplicates
	if (lowerBound != 0)
	{
		while (array[lowerBound-1].xValue == array[lowerBound].xValue)
		{
			lowerBound--;
			if (lowerBound == 0)
				break;
		}//while
	}//if

	if (upperBound != n-1)
	{
		while (array[upperBound+1].xValue == array[upperBound].xValue)
		{
			upperBound++;
			if (upperBound == n-1)
				break;
		}//while
	}//if

	float avg = 0;
	//calculate y average
	for (int i=lowerBound; i<upperBound+1; i++)
		avg += array[i].yValue;
	avg = avg/((float) (upperBound-lowerBound+1));

	array[idx].yValue = avg;	
}//kernel


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
	entry * deviceArray;
	float blockSize = 1024;

	//creat array of structs	
	for (int i=0; i<n; i++)
	{
		entry temp;
		temp.origIndex = i;
		temp.xValue = x[i];
		temp.yValue = y[i];
		array[i] = temp;
	}//for
	
	//sort by x values
	mergeSort(array, 0, n-1);

	cudaMalloc(&deviceArray, sizeof(entry) * n);
	cudaMemcpy(deviceArray, array, sizeof(entry) * n, cudaMemcpyHostToDevice);

	dim3 dimBlock(blockSize);
	dim3 dimGrid(ceil(n/blockSize));

	//stores smoothed average in yValue
	kernel <<< dimGrid, dimBlock >>>(deviceArray, n, h);

	cudaMemcpy(array, deviceArray, sizeof(entry) * n, cudaMemcpyDeviceToHost);
	
	//rearrange array in original order
	for (int i=0; i<n; i++)
	{
		m[array[i].origIndex] = array[i].yValue;
	}//for

	cudaFree(deviceArray);
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
