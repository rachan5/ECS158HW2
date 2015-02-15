//Authors: Group 5 Raymond S. Chan, Alicia Luu, Bryan Ng

//Comments are from Quiz2's mandelbrot
//
//
//
//
//
//
//
//
//return answer
//

#include <iostream>
#include <complex>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "omp.h"

extern "C" SEXP cmandel(SEXP Rnth, SEXP Rxl, SEXP Rxr, SEXP Ryb, SEXP Ryt, SEXP Rinc, SEXP Rmaxiters, SEXP Rsched, SEXP Rchunksize){
//int* cmandel(int nth, double xl, double xr, double yb, double yt, double inc, int maxiters, std::string sched, int chunksize){
  //determine where the tick marks go on the X and Y axes
  //possibly unneed functions here...
  SEXP Rval;
  double xl, xr, yb, yt, inc; 
  int nth, sched, maxiters, chunksize;
  Rnth = coerceVector(Rnth,INTSXP);
  Rxl = coerceVector(Rxl,REALSXP);
  Rxr = coerceVector(Rxr,REALSXP);
  Ryb = coerceVector(Ryb,REALSXP);
  Ryt = coerceVector(Ryt,REALSXP);
  Rinc = coerceVector(Rinc,REALSXP);
  Rmaxiters = coerceVector(Rmaxiters,INTSXP);
  Rchunksize = coerceVector(Rchunksize,INTSXP);
  Rsched = coerceVector(Rsched,INTSXP);
  nth = INTEGER(Rnth)[0];
  xl = REAL(Rxl)[0];
  xr = REAL(Rxr)[0];
  yb = REAL(Ryb)[0];
  yt = REAL(Ryt)[0];
  inc = REAL(Rinc)[0];
  sched = INTEGER(Rsched)[0];
  maxiters = INTEGER(Rmaxiters)[0];
  chunksize = INTEGER(Rchunksize)[0];
  
  int numXticks = abs((xr-xl))/inc;  
  int numYticks = abs((yb-yt))/inc;  
  double xticks[numXticks];
  double yticks[numYticks];
  
  //intialize Xticks and Ytick arrays return value matrix 
  for (int i = 0; i <= numXticks; i++){
    xticks[i] = xl +(inc*(double)i);
  }//for i
  
  for (int i = 0; i <= numYticks; i++){
    yticks[i] = yb +(inc*(double)i);
  }//for i

  //int * m = (int *) malloc(sizeof(int)*numXticks*numYticks);
  PROTECT(Rval = allocVector(INTSXP,(numXticks)*(numYticks)));

  for (int i = 0; i < numXticks; i++){
    for (int j = 0; j < numYticks; j++){
      //*(m+(numXticks*i)+j) = 0;
      INTEGER(Rval)[(i*numXticks)+j] = 0;
    }//for i
  }//for j

  //iteratate through the entire grid, 
  //setting c to each gtrid point and then
  //seeing when z goes a result
  double xti, ytj;
  std::complex<double> cpt;
  std::complex<double> z;
  
  //SWITCH STATEMENT SHOULD GO HERE!!!!!!!!!!!!!!!!!! 
  switch(sched){
    case 1:
      #pragma omp parallel num_threads(nth)
      {
      #pragma omp for schedule(static,chunksize)
      for(int i = 0; i < numXticks; i++){
        xti = xticks[i];
        for (int j = 0; j < numYticks; j++){
          ytj = yticks[j];
          cpt = std::complex<double>(xti,ytj);
          z = std::complex<double>(cpt);
          for (int k = 0; k <= maxiters; k++){
            z = (z*z) + cpt;
            if(sqrt((z.real()*z.real()+z.imag()*z.imag())) > 2.0){
              break;
            }
            if(k == maxiters){
              //*(m+(i*numXticks)+j) = 1;
              INTEGER(Rval)[(i*numXticks)+j] = 1;
            }
          }// end for k, maxiters
        }//end for j, numYticks
      }//end for i, numXticks
      UNPROTECT(1);
      return Rval;
      //return m;
      break;
      }
    case 2:
      #pragma omp parallel num_threads(nth)
      {
      #pragma omp for schedule(dynamic,chunksize)
      for(int i = 0; i < numXticks; i++){
        xti = xticks[i];
        for (int j = 0; j < numYticks; j++){
          ytj = yticks[j];
          cpt = std::complex<double>(xti,ytj);
          z = std::complex<double>(cpt);
          for (int k = 0; k <= maxiters; k++){
            z = (z*z) + cpt;
            if(sqrt((z.real()*z.real()+z.imag()*z.imag())) > 2.0){
              break;
            }
            if(k == maxiters){
              //*(m+(i*numXticks)+j) = 1;
              INTEGER(Rval)[(i*numXticks)+j] = 1;
            }
          }// end for k, maxiters
        }//end for j, numYticks
      }//end for i, numXticks
      UNPROTECT(1);
      return Rval;
      //return m;
      break;
      }
    case 3:
      #pragma omp parallel num_threads(nth)
      {
      #pragma omp for schedule(guided, chunksize)
      for(int i = 0; i < numXticks; i++){
        xti = xticks[i];
        for (int j = 0; j < numYticks; j++){
          ytj = yticks[j];
          cpt = std::complex<double>(xti,ytj);
          z = std::complex<double>(cpt);
          for (int k = 0; k <= maxiters; k++){
            z = (z*z) + cpt;
            if(sqrt((z.real()*z.real()+z.imag()*z.imag())) > 2.0){
              break;
            }
            if(k == maxiters){
              //*(m+(i*numXticks)+j) = 1;
              INTEGER(Rval)[(i*numXticks)+j] = 1;
            }
          }// end for k, maxiters
        }//end for j, numYticks
      }//end for i, numXticks
      UNPROTECT(1);
      return Rval;
      //return m;
      break;
      }
    default:
      std::cout << "error in sched" << std::endl;
  }
  return 0;
}//end cmandel()



//=============================================================================
/*
int main(){
  //operators for complex type:  =, +=, -=, *=, /=, +, -, *, /, ==, !=, <<, >>
  std::complex<double> first (2.1,2.0);
  std::complex<double> second (first);
  std::complex<double> third;
  third = std::complex<double>(1.0,2.3);
  //std::cout << third << std::endl;
  //std::cout << "first: " << first*first << "second: " << second << std::endl;
  int * answer;
  int numXticks = ceil(abs((1.0-(-1.0)))/0.01);  
  int numYticks = ceil(abs((1.0-(-1.0)))/0.01);  
  
  //int* cmandel(int nth, double xl, double xr, double yb, double yt, double inc, int maxiters, std::string sched, int chunksize){
  answer =  cmandel(8, -1.0, 1.0,-1.0, 1.0, 0.01, 100, "lalala", 100);
  for (int i = 0; i <= numXticks; i++){
    for (int j = 0; j <= numYticks; j++){
      std::cout << " " << *(answer+(numXticks*i)+j);
    }
    std::cout << std::endl;
  }

  
  //cmandelbrot
}
*/
