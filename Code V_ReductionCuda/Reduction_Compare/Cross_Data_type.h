#ifndef _DATATYPE_H_
#define _DATATYPE_H_

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



#endif // _DATATYPE_H_