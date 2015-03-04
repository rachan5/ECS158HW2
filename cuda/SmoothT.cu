#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <iostream>


struct entry
{
	int origIndex;
	float xValue, yValue;
};//entry

struct smoother_functor
{
  const float h;
  const float x_i;
  smoother_functor(float _h, float _x_i): h(_h), x_i(_x_i){}
  __host__ __device__
  float operator()(const float&x, const float& y)const //FILL IN FROM TUTORIAL
  {
    return (fabs(x-x_i) < (float)h) ? 1.0 : 0.0;
  }
}; //struct smoother_functor

void smootht(float *x, float *y, float *m, int n, float h)
{
  thrust::device_vector<float> dx(n,0.0);
  thrust::device_vector<float> dy(n,0.0);
  thrust::device_vector<float> dm(n,0.0);
  thrust::device_vector<float> dtemp(n,0.0);
  thrust::device_vector<float> dtemp2(n,0.0);
  //gotta initalize these arrays!
  //std::cout << "Check initalization" << std::endl;
  for (int i = 0; i < n; i++)
  {
    *(dx.begin()+i) = x[i];
    *(dy.begin()+i) = y[i];
    //std::cout << *(dx.begin()+i) << " " << *(dy.begin()+i) << std::endl;
  }
  
  //iterate through x for x_i, pass into smoother_functor...
  for (int i = 0; i < dx.size(); i++)
  {
    float total = 0.0;
    int count = 0;
    thrust::transform(dx.begin(),dx.end(), dy.begin(), dtemp.begin(),smoother_functor(h,*(dx.begin()+i)));
    //std::cout << "CHECK AVERAGING" << std::endl;
   
    //calculate averages
    /*
    for (int j = 0; j < dtemp.size(); j++)
    {
      if (*(dtemp.begin()+j) == 1)
      {
        total += *(dy.begin()+j);
        count++;
      }
    }//end for j
    *(dm.begin()+i) = (total/count);
    */
    thrust::transform(dy.begin(), dy.end(), dtemp.begin(), dtemp2.begin(), thrust::multiplies<float>());
    total = thrust::reduce(dtemp2.begin(),dtemp2.end());
    count = thrust::reduce(dtemp.begin(),dtemp.end());
    *(dm.begin()+i) = (total/count);
    
    //std::cout << total << " " << count << " " << total/count << std::endl;
  }//end for i 
  thrust::copy(dm.begin(), dm.end(), std::ostream_iterator<float>(std::cout, "\n"));

}//smootht


int main()
{
  	
  int a, n = 2000000;
	float * x = new float[n];
	float * y = new float[n];
	float * m = new float[n];
	float h = 2;
	
	a=rand();//range of float [-a, a]
	srand(time(NULL));//init rand() seed
	for (int i=0; i<n; i++)
	{
		x[i] = ((float)rand()/(float)(RAND_MAX)*2*a - a);
		y[i] = ((float)rand()/(float)(RAND_MAX)*2*a - a);
	}//generate random floats for x and y
  
	
  /*
  float x[20] = {1, 1,2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9, 10,10};
	float y[20] = {11,11, 12,12, 13,13, 14,14, 15,15, 16,16, 17,17, 18,18, 19,19, 20,20};
	float m[20];
	int n = 20;
	float h = 2;
  */
	smootht(x, y, m, n, h);
}//main
