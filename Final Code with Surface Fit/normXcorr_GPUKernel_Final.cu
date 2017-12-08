#ifndef _NORM_XCORR_KERNEL_H_
#define _NORM_XCORR_KERNEL_H_

//#include "Cross_Data_type.h"
#include "corr2Mex.h"
////// Kernel For Computing CC Values //////

__global__ void normXcorr_GPU(Matrix Pre, Matrix Post, float *Corr, params P,float *preMean,float *preVar,float *postMean,float *postVar)
{

	extern __shared__ float sharedPreElements[];

	int imgWidth_pre = Pre.width; // 244
	int imgWidth_post = Post.width; // 244
	int kerX = P.kernelX; // 9
	int kerY = P.kernelY; // 161
	int sX = P.searchX; // 17
	int sY = P.searchY; // 49
	int shiftX = sX/2; // 8
	int shiftY = sY/2;// 24
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

	int start_row = by*stepY + shiftY;
	int start_col = bx + shiftX;

	int tx = threadIdx.x;
	int ty = threadIdx.y;
//====================================================================
//========================Shared Memory Initialization================
//====================================================================

//IMPORTANT!!! This scheme only works when kerX < number of threads in x direction.
if(kerX<=blockDim.x){
	if(tx<kerX && ty==0){
		for (int j=0; j<(kerY); j++) //161
		{
		sharedPreElements[tx+j*kerX]=Pre.elements[(tx+start_col)+(j+start_row)*imgWidth_pre];
		}
	}
}else{
	int iter = (kerX+blockDim.x-1)/blockDim.x;
	for(int q=0; q<iter; q++){
		int TX = tx+q*blockDim.x;
	if(TX<kerX && ty==0){
		for (int j=0; j<(kerY); j++) //161
		{
		sharedPreElements[TX+j*kerX]=Pre.elements[(TX+start_col)+(j+start_row)*imgWidth_pre];
		}
	}
}
}
//__syncthreads();

/////////PRE//////////////
	if(tx==0&&ty==1){
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

	//====================================================================
	//===============================Post Stats===========================
	//====================================================================
	if(tx<sX && ty<sY){

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
		postMean[ptId] = sumPost/((kerY)*(kerX));

	for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
					sum2Post += (Post.elements[(post_start_row+j)*(imgWidth_post)+(post_start_col+i)]-postMean[ptId])*(Post.elements[(post_start_row+j)*(imgWidth_post)+(post_start_col+i)]-postMean[ptId]);
			}
		}
		postVar[ptId]=sum2Post;

		__syncthreads();

	//====================================================================
	//===========================CC Calculation===========================
	//====================================================================
	float CC = 0.0;

	for(int j=0;j<kerY;j++){
		for(int i=0;i<kerX;i++){
				CC += (sharedPreElements[i+j*kerX]-preMean[bx+by*gridDim.x])*(Post.elements[(post_start_row+j)*(imgWidth_post)+(post_start_col+i)]-postMean[ptId]);
		}
	}
	
	int xx = bx+by*gridDim.x;
	if(postVar[ptId]>0.00000001) Corr[(xx*sY+ty)*sX+tx] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]); // Row Major Update
	else Corr[(xx*sY+ty)*sX+tx] = 0.00;
	//Corr[(xx*sY+ty)*sX+tx] = postVar[ptId]; // Row Major Update
	//Corr[(xx*sX+tx)*sY+ty] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]); // Column Major Update
	//if (Corr[(xx*sX+tx)*sY+ty] != Corr[(xx*sX+tx)*sY+ty]) Corr[(xx*sX+tx)*sY+ty] = -1.00;
	//if (isnan(Corr[(xx*sX+tx)*sY+ty]) == 1) Corr[(xx*sX+tx)*sY+ty] = -1.00;
	//if (isinf(Corr[(xx*sX+tx)*sY+ty]) == 1) Corr[(xx*sX+tx)*sY+ty] = -1.00;
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
  //dpX[xx] = (sindex[tx]%sX)-(sX/2);
  //dpY[xx] = (sindex[tx]/sX)-(sY/2);
  dpX[xx] = (sindex[tx]%sX);
  dpY[xx] = (sindex[tx]/sX);
  }
  __syncthreads();
}

//====================================================================
//===========================Kernel for QuadSurface Fit===============
//====================================================================

/*__device__ void subsamplefit(float* vector,float temp){
	temp = 0.0f;
	float tempx = 0.0;
	if(vector[0]!=vector[2]){
		tempx = (vector[0]-vector[2])/(2*(vector[0]+vector[2]-2*vector[1]));
		if(abs(tempx)<1){
			temp = tempx;
		}
	}
}*/

__device__ void subsamplefitx(float* vector,float *temp){
	temp[0] = 0.0f;
	float tempx = 0.0;
	if(vector[0]!=vector[2]){
		tempx = (vector[0]-vector[2])/(2*(vector[0]+vector[2]-2*vector[1]));
		if(abs(tempx)<1){
			temp[0] = tempx;
		}
	}
}


__device__ void subsamplefity(float* vector,float *temp){
	temp[1] = 0.0f;
	float tempx = 0.0;
	if(vector[0]!=vector[2]){
		tempx = (vector[0]-vector[2])/(2*(vector[0]+vector[2]-2*vector[1]));
		if(abs(tempx)<1){
			temp[1] = tempx;
		}
	}
}
__device__ void quadsurfacefit(float *SSD,float *temp){
	float tempSum = 0.0, sumSSD = 0.0 ;
	float b,c,e,d,f;
	float vector[3];
	float sumVer[3];
	float sumHor[3];
	
	int size = 9;
	for(int i=0;i<size;i++){
	tempSum +=SSD[i];
	}
	sumSSD = tempSum/3;



	for(int col=0;col<3;col++){
		for(int row=0;row<3;row++){
			sumVer[col]+=SSD[row*3+col];
		}
	}

	for(int row=0;row<3;row++){
		for(int col=0;col<3;col++){
			sumHor[row]+=SSD[row*3+col];
		}
	}

	b = (sumVer[2] - sumVer[0])/6.0;
	c = (sumHor[2] - sumHor[0])/6.0;
	e = (SSD[0]-SSD[2]-SSD[6]+SSD[8])*0.25;
	d = (sumVer[0]+sumVer[2])*0.5 - sumSSD;
	f = (sumHor[0]+sumHor[2])*0.5 - sumSSD;

	temp[0] = (2*b*f-c*e)/(e*e-4*d*f);
	temp[1] = (-2*d*temp[0]-b)/e;
	
	if(abs(temp[0])>=1){
		for(int i=0;i<3;i++)
		{
			vector[i]=SSD[3+i];
		}
		subsamplefitx(vector,temp);
	}
	
	if(abs(temp[1])>=1){
		for(int i=0;i<3;i++)
		{
			vector[i]=SSD[1+i*3];
		}
		subsamplefity(vector,temp);
	}

	//temp[0] = xo;
	//temp[1] = yo;
}	


	
__global__ void subsample(float *Corr, params P,float *quality, int *dpX, int *dpY,float *dpX_s,float *dpY_s){
	
	float SSD[9];
	float temp[2];
	float vector[3];
	int n = P.numX*P.numY;
	int sXX = P.searchX;
	int sYY = P.searchY;
	int k =  blockIdx.x*blockDim.x + threadIdx.x;
	if(k<n){
		if(dpY[k]>0 && dpY[k]<sYY-1 && dpX[k]>0 && dpX[k]<sXX-1){
		for(int r=0;r<3;r++){
			for(int p=0;p<3;p++){
				SSD[r*3+p] = Corr[(k*sYY+dpY[k]-1+r)*sXX+(dpX[k]-1+p)];
				}
			}
			quadsurfacefit(SSD,temp);
		}
		else if(dpY[k]>0 && dpY[k]<sYY-1){
			temp[0] = 0.0f;
			for(int i=0;i<3;i++)
			{
				vector[i]=Corr[(k*sYY+dpY[k]-1+i)*sXX+(dpX[k])];
			}
			subsamplefity(vector,temp);
		}
		else if(dpX[k]>0 && dpX[k]<sXX-1){
			temp[1] = 0.0f;
			for(int i=0;i<3;i++)
			{
				vector[i]=Corr[(k*sYY+dpY[k])*sXX+(dpX[k]-1+i)];
			}
			subsamplefitx(vector,temp);
		}
		else{
			temp[0] = 0.0f;
			temp[1] = 0.0f;
		}
		dpX_s[k] = float(dpX[k]) -(sXX/2) + temp[0];
		dpY_s[k] = float(dpY[k]) -(sYY/2) + temp[1];
	}
}

#endif // #ifndef _NORM_XCORR_KERNEL_H_
