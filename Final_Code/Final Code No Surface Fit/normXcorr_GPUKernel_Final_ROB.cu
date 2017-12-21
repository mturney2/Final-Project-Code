#ifndef _NORM_XCORR_KERNEL_H_
#define _NORM_XCORR_KERNEL_H_

//#include "Cross_Data_type.h"
#include "corr2Mex.h"
////// Kernel For Computing CC Values //////

__global__ void normXcorr_GPU(Matrix Pre, Matrix Post, float *Corr, params P,float *preMean,float *preVar,float *postMean,float *postVar)
{
	int imgWidth_pre = Pre.width;
	int imgWidth_post = Post.width;
	int kerX = P.kernelX; // 5
	int kerY = P.kernelY; // 5
	int sX = P.searchX; // 3
	int sY = P.searchY; // 3
	int shiftX = sX/2;
	int shiftY = sY/2;
	float overlap = P.overlap;
	int stepY;
	if( overlap  < 100){
		stepY = kerY*((100.0 - overlap)/100)+1;
	}
	else{
		stepY = kerY*((100.0 - overlap)/100);
	}
	
  
	int by = blockIdx.y;
	int bx = blockIdx.x;
  
	//int start_row = by*stepY - halfY;
	//int start_col = bx - halfX ;
	
	int start_row = by*stepY + shiftY;
	int start_col = bx + shiftX;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

/////////PRE//////////////	
	if(tx==0&&ty==0){
		//Mean
		float sumPre = 0.0;
		float sum2Pre = 0.0;
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
					sumPre += Pre.elements[(i+start_col)+(j+start_row)*imgWidth_pre];
			}
		}
		preMean[bx+by*gridDim.x] = sumPre/(kerX*kerY);
		
		//Variance
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
					sum2Pre += (Pre.elements[(i+start_col)+(j+start_row)*imgWidth_pre]-preMean[bx+by*gridDim.x])*(Pre.elements[(i+start_col)+(j+start_row)*imgWidth_pre]-preMean[bx+by*gridDim.x]);
			}
		}
		preVar[bx+by*gridDim.x] = sum2Pre;
	}
__syncthreads();


//////Post////////
if(tx<sX && ty<sY){	
	//int post_start_row = start_row + ty - shiftY;
	//int post_start_col = start_col + tx - shiftX;
	
	int post_start_row = start_row + ty - shiftY ;
	int post_start_col = start_col + tx - shiftX ;
	
	//Global Index for PostMean and PostVariance
	int bId =  blockIdx.x + blockIdx.y*gridDim.x;
	int ptId = bId*(blockDim.x * blockDim.y)+(threadIdx.y*blockDim.x)+threadIdx.x;
	
	
	float sumPost = 0.0;
	float sum2Post = 0.0;
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
					sumPost += Post.elements[(post_start_row+j)*(imgWidth_post)+(post_start_col+i)];
			}
		}
		//postMean[tx+ty*sX] = sumPost/((kerY)*(kerX));
		postMean[ptId] = sumPost/((kerY)*(kerX));
	
	for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
					sum2Post += (Post.elements[(post_start_row+j)*(imgWidth_post)+(post_start_col+i)]-postMean[ptId])*(Post.elements[(post_start_row+j)*(imgWidth_post)+(post_start_col+i)]-postMean[ptId]);	
			}
		}
		//postVar[tx+ty*sX]=sum2Post;	
		postVar[ptId]=sum2Post;
		
__syncthreads();
	
/////Cross-correlation Calculation/////

		
	float CC = 0.0;
	
	
	for(int j=0;j<kerY;j++){			
		for(int i=0;i<kerX;i++){				
				CC += (Pre.elements[(i+start_col)+(j+start_row)*imgWidth_pre]-preMean[bx+by*gridDim.x])*(Post.elements[(post_start_row+j)*(imgWidth_post)+(post_start_col+i)]-postMean[ptId]);		
		}
	}
		
	int xx = bx+by*gridDim.x;
	//Corr[tx+sX*(ty+sY*xx)] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]);
	Corr[(xx*sY+ty)*sX+tx] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]); // Row Major Update
	//Corr[(xx*sX+tx)*sY+ty] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]); // Column Major Update	
	if (Corr[(xx*sX+ty)*sY+tx] != Corr[(xx*sX+ty)*sY+tx]) Corr[(xx*sX+ty)*sY+tx] = -1.00;
}
}

__device__ __forceinline__ void maxE(volatile float *a, volatile float *b,volatile int* c,volatile int index) {
  if(*a<*b) {*a=*b;*c=index;}
}

__device__ __forceinline__ void maxE(volatile float *a, volatile float *b,volatile int* c,volatile int* d) {
  if(*a<*b) {*a=*b;*c=*d;}
}


__device__ __forceinline__ void warpReduce(volatile float *sdata,volatile int* sindex, unsigned int tid) {
	maxE(&sdata[tid],&sdata[tid+32],&sindex[tid],&sindex[tid+32]);
  maxE(&sdata[tid],&sdata[tid+16],&sindex[tid],&sindex[tid+16]);
	maxE(&sdata[tid],&sdata[tid+8],&sindex[tid],&sindex[tid+8]);
	maxE(&sdata[tid],&sdata[tid+4],&sindex[tid],&sindex[tid+4]);
	maxE(&sdata[tid],&sdata[tid+2],&sindex[tid],&sindex[tid+2]);
  maxE(&sdata[tid],&sdata[tid+1],&sindex[tid],&sindex[tid+1]);
}

__device__ __forceinline__ void maximize(float *sdata,int* sindex, unsigned int tid) {

    if (tid < 512) { maxE(&sdata[tid],&sdata[tid+512],&sindex[tid],&sindex[tid+512]); } 
  __syncthreads();  
	if (tid < 256) { maxE(&sdata[tid],&sdata[tid+256],&sindex[tid],&sindex[tid+256]); } 
  __syncthreads();
	if (tid < 128) { maxE(&sdata[tid],&sdata[tid+128],&sindex[tid],&sindex[tid+128]); } 
  __syncthreads();
	if (tid < 64) { maxE(&sdata[tid],&sdata[tid+64],&sindex[tid],&sindex[tid+64]); } 
  __syncthreads();
	if (tid < 32) warpReduce(sdata,sindex, tid);
	
}

__global__
void  MaxElement(float *Corr, params P,float *quality, int *dpX, int *dpY){
	int by = blockIdx.y;
	int bx = blockIdx.x;
  int xx = bx+by*gridDim.x;
  int sX = P.searchX;
	int sY = P.searchY;
  int n = sX*sY;
  extern __shared__ char array[];
  float *sdata=(float*)array;
  int* sindex=(int*)(array+1024*sizeof(float));
  int tx=threadIdx.x;
  
  if(tx<n)
  sdata[tx] = Corr[(xx*sY)*sX+tx];
  if(tx>=n)
  sdata[tx]=-1; // this is very low and wont get picked
  
  sindex[tx]=tx;
  for(int i=tx+blockDim.x;i<n;i+=blockDim.x)
		maxE(&sdata[tx],&Corr[(xx*sY)*sX+i],&sindex[tx],i);	
  __syncthreads();

  maximize(sdata,sindex,tx);
  __syncthreads();
  if(tx==0){
  quality[xx] = sdata[tx];
  dpX[xx] = (sindex[tx]%sX)-(sX/2);
  dpY[xx] = (sindex[tx]/sX)-(sY/2);
  }
  __syncthreads();
}

#endif // #ifndef _NORM_XCORR_KERNEL_H_