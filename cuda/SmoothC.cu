#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define GIGABYTE 1000000000

struct entry
{
	int origIndex;
	float xValue, yValue;
};//entry


int h_binarySearchLB(entry * data, float val, int n)
{
	//return index of greatest leftmost xValue that is greater than val
	int left = 0;
	int right = n;
	int mid;

	while (left != right)
	{
		mid = (left+right)/2;
		
		if (data[mid].xValue <= val)
			left = mid + 1;
		else
			right = mid;
	}//while
	return left;
}//binarySearchLB


int h_binarySearchUB(entry * data, float val, int n)
{
	//return index of greatest leftmost xValue that is greater than val
	int left = 0;
	int right = n;
	int mid;

	while (left != right)
	{
		mid = (left+right)/2;
		
		if (data[mid].xValue >= val)
			right = mid;
		else
			left = mid + 1;
	}//while

	return left;
}//binarySearchUB


__device__ int binarySearchLB(entry * data, float val, int n)
{
	//return index of greatest leftmost xValue that is greater than val
	int left = 0;
	int right = n;
	int mid;

	while (left != right)
	{
		mid = (left+right)/2;
		
		if (data[mid].xValue <= val)
			left = mid + 1;
		else
			right = mid;
	}//while
	return left;
}//binarySearchLB


__device__ int binarySearchUB(entry * data, float val, int n)
{
	//return index of greatest leftmost xValue that is greater than val
	int left = 0;
	int right = n;
	int mid;

	while (left != right)
	{
		mid = (left+right)/2;
		
		if (data[mid].xValue >= val)
			right = mid;
		else
			left = mid + 1;
	}//while

	return left;
}//binarySearchUB


__global__ void kernel1(entry * array, int n, float h)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int lowerBound = binarySearchLB(array, array[idx].xValue-h, n);
	int upperBound = binarySearchUB(array, array[idx].xValue+h, n);

	float avg = 0;
	//calculate y average
	for (int i=lowerBound; i<upperBound; i++)
		avg += array[i].yValue;
	avg = avg/((float) (upperBound-lowerBound));

	//yValue stores the avg
	array[idx].yValue = avg;	
}//kernel1


__global__ void kernel2(entry * array, int n)
{
	float avg = 0;
	for (int i=0; i<n; i++)
		avg += array[i].yValue;
	avg = avg / (float) n;

	array[0].yValue = avg;			
}//kernel2


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
	int * countArray = new int[n];
	int blockSize = 1024;

	//creat array of structs	
	for (int i=0; i<n; i++)
	{
		entry temp;
		temp.origIndex = i;
		temp.xValue = x[i];
		temp.yValue = y[i];
		array[i] = temp;
	}//for

	//sort by xValue
	mergeSort(array, 0, n-1);

	if (n < GIGABYTE/sizeof(entry))
	{
		cudaMalloc(&deviceArray, sizeof(entry) * n);
		cudaMemcpy(deviceArray, array, sizeof(entry) * n, cudaMemcpyHostToDevice);

		dim3 dimBlock(blockSize);
		dim3 dimGrid(ceil(n/blockSize));

		//stores smoothed average in yValue
		kernel1 <<< dimGrid, dimBlock >>> (deviceArray, n, h);
		cudaMemcpy(array, deviceArray, sizeof(entry) * n, cudaMemcpyDeviceToHost);
		
	//rearrange array in original order
		for (int i=0; i<n; i++)
			m[array[i].origIndex] = array[i].yValue;
		cudaFree(deviceArray);
	}//if
	else //have to chunk up data
	{
		int lb, ub;
		for (int i=0; i<n; i++)
		{
			lb = h_binarySearchLB(array, array[i].xValue-h, n);
			ub = h_binarySearchUB(array, array[i].xValue+h, n);
			entry * chunkArray = new entry[ub-lb];
	
			for (int j=0; j<ub-lb; j++)
				chunkArray[j] = array[lb+j];
			
			cudaMalloc(&deviceArray, sizeof(entry) * (ub-lb));
			cudaMemcpy(deviceArray, chunkArray, sizeof(entry) * (ub-lb), cudaMemcpyHostToDevice);

			kernel2 <<< 1, 1 >>> (deviceArray, ub-lb);
			cudaMemcpy(chunkArray, deviceArray, sizeof(entry) * (ub-lb), cudaMemcpyDeviceToHost);

			m[array[i].origIndex] = chunkArray[0].yValue;
			cudaFree(deviceArray);
			delete [] chunkArray;
		}//for
	}//else
	
	delete [] array;
}//smoothc

/*
int main()
{
	int n = 200;
	float * x = new float[n];
	float * y = new float[n];
	float * m = new float[n];
	float h = 2;
	
	for (int i=0; i<n; i++)
	{
		x[i] = rand() % 100;
		y[i] = rand() % 100;
	}//for
	float x[20] = {1, 1,2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9, 10,10};
	float y[20] = {11,11, 12,12, 13,13, 14,14, 15,15, 16,16, 17,17, 18,18, 19,19, 20,20};
	float m[20];
	int n = 20;
	float h = 2;
	smoothc(x, y, m, n, h);
//	delete [] x;
//	delete [] y;
//	delete [] m;
}//main
*/
