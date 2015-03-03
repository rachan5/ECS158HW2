#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <iostream>


struct entry
{
	int origIndex;
	float xValue, yValue;
};//entry

struct smoother_functor
{
  const float h;
  //const float sum;
  //const int nums_in_sum;
  //smoother_functor(float _h, float _sum, int _nums_in_sum): h(_h), sum(_sum), nums_in_sum(_nums_in_sum){}
  const float x_i;
  smoother_functor(float _h, float _x_i): h(_h), x_i(_x_i){}
  //template <typename Tuple>
  __host__ __device__
  //float operator()(Tuple t)
  float operator()(const float&x, const float& y)const //FILL IN FROM TUTORIAL
  {
    //if (abs(thrust::get<0>(t) - thrust::get<1>(t)) < h)
    //{
    //  //Y_j is stored here
    //  sum += thrust::get<2>(t);
    //  //nums_in_sum++;
    //}
    //else
    //{
    //  sum += 0;
    //  nums_in_sum += 0;
    //}

    //answer will go here
    //thrust::get<3>(t) = sum/nums_in_sum; 
    //return 0;
    return (abs(x-x_i) > h) ? 1 : 0;
  }
}; //struct smoother_functor

void smootht(float *x, float *y, float *m, int n, float h)
{
  thrust::device_vector<float> dx(n,0.0);
  thrust::device_vector<float> dy(n,0.0);
  thrust::device_vector<float> dm(n,0.0);
  thrust::device_vector<int> dtemp(n,0);
  thrust::device_vector<float> dtemp2(n,0.0);
  //gotta initalize these arrays!
  for (int i = 0; i < n; i++)
  {
    *(dx.begin()+i) = x[i];
    *(dy.begin()+i) = y[i];
    std::cout << *(dx.begin()+i) << " " << *(dy.begin()+i) << std::endl;
  }
  //iterate through x for x_i, pass into smoother_functor...
  for (int i = 0; i < dx.size(); i++)
  {
    float total = 0.0;
    int count =0;
    thrust::transform(dx.begin(),dx.end(), dtemp.begin(), dtemp.begin(),smoother_functor(h,*(dx.begin()+i)));
    for (int j = 0; j < dtemp.size(); j++)
    {
      if (*(dtemp.begin()+j))
      {
        total += *(dy.begin()+j);
        count++;
      }
    }//end for j
    *(dm.begin()+i) = total/count;
    std::cout << total << " " << count << " " << total/count << std::endl;
  //  //thrust::fill(dtemp.begin(), dtemp.end(), dx.begin()+i);
  //  //thrust::transform(dx.begin(),dx.end(), dtemp.begin(), dtemp2.end(),smoother_functor(h,));
  }//end for i 
  //thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(dx.begin(), dx.begin(), dy.begin(), dm.begin())),thrust::make_zip_iterator(thrust::make_tuple(dx.end(),   dx.end(),   dy.end(),   dm.end())),smoother_functor(h,0.0,0));

  thrust::copy(dm.begin(), dm.end(), std::ostream_iterator<float>(std::cout, "\n"));

}//smootht


int main()
{
	/*
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
*/
	
  
  float x[20] = {1, 1,2,2, 3,3, 4,4, 5,5, 6,6, 7,7, 8,8, 9,9, 10,10};
	float y[20] = {11,11, 12,12, 13,13, 14,14, 15,15, 16,16, 17,17, 18,18, 19,19, 20,20};
	float m[20];
	int n = 20;
	float h = 2;

	smootht(x, y, m, n, h);
}//main
