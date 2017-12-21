#ifndef _CORR2MEXCUDA_H_
#define _CORR2MEXCUDA_H_

typedef struct{
  int searchY;
  int searchX;
  int kernelY;
  int kernelX;
  float overlap; //between 0-1
  int numX;
  int numY;
} params ;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;

struct SoA_Corr{
  float *Corr_Points;
} ; // Structure of Array for Storing Cross-corr values 

void CorrelationOnDevice(const Matrix Pre, const Matrix Post, float *CorrH, params parameters,float *quality,float *dpX, float *dpY,int *dpX_H,int *dpY_H);

#endif // _CORR2MEXCUDA_H_