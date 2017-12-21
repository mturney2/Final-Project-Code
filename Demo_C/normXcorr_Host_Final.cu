// Host Side Code for Cross-correlation in GPU

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include "corr2cuda.h"
#include "normXcorr_GPUKernel_Cuda.cu"

using namespace std;


Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix_Pre(int height, int width,int init);
Matrix AllocateMatrix_Post(int height, int width,int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void CorrelationOnDevice(const Matrix M, const Matrix N, float *CorrH, params parameters,float *quality,float *dpX, float *dpY,int *dpX_H,int *dpY_H);


int main(int argc,char** argv) {
	// Input Parameters

	if(argc!=11) // Both Pre and Post Padded Image Size should be provided as Input
	{
		printf("Usage %s Parameters missing\n",argv[0]);
		return 1;
	}

	int imageWidth_pre = atoi(argv[1]);
	int imageHeight_pre = atoi(argv[2]);
	int imageWidth_post = atoi(argv[3]);
	int imageHeight_post = atoi(argv[4]);
	int SEARCH_X = atoi(argv[5]);
	int SEARCH_Y = atoi(argv[6]);
	int KERNEL_X = atoi(argv[7]);
	int KERNEL_Y = atoi(argv[8]);
	int numX = atoi(argv[9]);
	int numY = atoi(argv[10]);
	int DisplacementSize = numX*numY;
	int Corr_size = SEARCH_X*SEARCH_Y;
	Matrix Pre;
	Matrix Post;

	float OVERLAP = 50.0;

	params parameters = {SEARCH_Y,SEARCH_X,KERNEL_Y,KERNEL_X,OVERLAP,numX,numY};
	Pre  = AllocateMatrix_Pre(imageHeight_pre,imageWidth_pre, 1);
	Post  = AllocateMatrix_Post(imageHeight_post,imageWidth_post, 1);
	float gpuTime=0.f;

	float *CorrH;
	cudaMallocHost((void**)&CorrH, Corr_size*DisplacementSize*sizeof(float));
	//CorrH = (float*)malloc(Corr_size*DisplacementSize*sizeof(float));

	float *quality;
	quality = (float*)malloc(DisplacementSize*sizeof(float));
	float *dpX;
	dpX = (float*)malloc(DisplacementSize*sizeof(float));
	float *dpY;
	dpY = (float*)malloc(DisplacementSize*sizeof(float));
	
	int* dpX_H;
	dpX_H = (int*)malloc(sizeof(int)*parameters.numX*parameters.numY);
	
	int* dpY_H;
	dpY_H = (int*)malloc(sizeof(int)*parameters.numX*parameters.numY);

	float  elapsedTime_inc;
	cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventRecord(startEvent_inc,0); // starting timing for inclusive

	CorrelationOnDevice(Pre, Post, CorrH, parameters,quality,dpX,dpY,dpX_H,dpY_H); // Execution Model for GPU is set up in this function


  cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);
	gpuTime = elapsedTime_inc;

	

	
	//for(int h=0;h<DisplacementSize;h++){
	/*int h = DisplacementSize - 1;
		for(int g=0;g<SEARCH_Y;g++){
			for(int z=0;z<SEARCH_X;z++){
				printf("%f ",CorrH[(h*SEARCH_Y+g)*SEARCH_X+z]);
			}
		printf("\n");
		}
		printf("\n");*/
	//}
	
  float cp_dpY = 0.1464;
  float cp_dpX = -.0992; 
	int u = DisplacementSize -1; 
  printf("dpY_CPU = %0.4f\ndpY_GPU = %0.4f\ndpX_CPU = %0.4f\ndpX_GPU = %0.4f\n",cp_dpY,dpY[u],cp_dpX,dpX[u]);
	
	/*for(int u=0;u<DisplacementSize;u++)
	{
		printf(" %d %f  %f  %f\n",u,quality[u],dpY[u],dpX[u]);
	}*/
	
	/*for(int u=0;u<parameters.numX*parameters.numY;u++)
	{
		printf("%d  %d\n",dpY_H[u],dpX_H[u]);
	}*/
	printf("Elasped Time = %f\n",gpuTime);

	// Free matrices
	FreeMatrix(&Pre);
	FreeMatrix(&Post);
	return 0;

}

//// Cuda Kernel Call //////

void CorrelationOnDevice(const Matrix Pre, const Matrix Post, float *CorrH, params parameters,float *quality,float *dpX, float *dpY,int *dpX_H,int *dpY_H)
{
	// Load Pre and Post to the device
	Matrix Pred = AllocateDeviceMatrix(Pre);
	CopyToDeviceMatrix(Pred, Pre);
	Matrix Postd = AllocateDeviceMatrix(Post);
	CopyToDeviceMatrix(Postd, Post);

	// Allocate Space for Pre-Mean
	float *preMean;
	float *preVar;
	cudaMalloc((void **)&preMean,sizeof(float)*parameters.numX*parameters.numY);
	cudaMalloc((void **)&preVar,sizeof(float)*parameters.numX*parameters.numY);

	//Allocate Space for Post-mean
	float *postMean;
	float *postVar;
	cudaMalloc((void **)&postMean,sizeof(float)*parameters.searchX*parameters.searchY*parameters.numX*parameters.numY);
	cudaMalloc((void **)&postVar,sizeof(float)*parameters.searchX*parameters.searchY*parameters.numX*parameters.numY);


	// Device Memory Allocation for Cross-correlation Result
	float *CorrD;
	cudaMalloc((void **)&CorrD,sizeof(float)*parameters.numX*parameters.numY*parameters.searchX*parameters.searchY);

	//Initialize Values for Displacement Results
	float *qualityD;
	cudaMalloc((void **)&qualityD,sizeof(float)*parameters.numX*parameters.numY);
	

	int *dpX_D;
	cudaMalloc((void **)&dpX_D,sizeof(int)*parameters.numX*parameters.numY);

	int *dpY_D;
	cudaMalloc((void **)&dpY_D,sizeof(int)*parameters.numX*parameters.numY);
	
	float *dpX_sD;
	cudaMalloc((void **)&dpX_sD,sizeof(float)*parameters.numX*parameters.numY);

	float *dpY_sD;
	cudaMalloc((void **)&dpY_sD,sizeof(float)*parameters.numX*parameters.numY);


	// Setup the execution configuration

	dim3 dimBlock(parameters.searchX, parameters.searchY);
	dim3 dimGrid(parameters.numX, parameters.numY);
	// Launch the device computation threads!
	
	// Kernel Call for NCC Calculation
	normXcorr_GPU<<<dimGrid, dimBlock, parameters.kernelX*parameters.kernelY*sizeof(float)>>>(Pred,Postd,CorrD,parameters,preMean,preVar,postMean,postVar);
	
	// Kernel Call for Peak Finding for CC Results
	int smemSize = 1024*sizeof(int)+1024*sizeof(float);
	MaxElement<<<dimGrid,1024,smemSize>>>(CorrD,parameters,qualityD,dpX_D,dpY_D);
	
	// Subsample Estimation
	int numthreads = 512;
	int numblocks = (parameters.numX*parameters.numY + numthreads -1)/numthreads;
	subsample<<<numblocks,numthreads>>>(CorrD,parameters,qualityD,dpX_D,dpY_D,dpX_sD,dpY_sD);
	
	// Copying the Results from GPU to CPU
	//cudaMemcpy(CorrH,CorrD,sizeof(float)*parameters.numX*parameters.numY*parameters.searchX*parameters.searchY,cudaMemcpyDeviceToHost);
	cudaMemcpy(quality,qualityD,sizeof(float)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	cudaMemcpy(dpX_H,dpX_D,sizeof(int)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	cudaMemcpy(dpY_H,dpY_D,sizeof(int)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	cudaMemcpy(dpX,dpX_sD,sizeof(float)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	cudaMemcpy(dpY,dpY_sD,sizeof(float)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	
	// Free device matrices
	FreeDeviceMatrix(&Pred);
	FreeDeviceMatrix(&Postd);
	cudaFree(CorrD);
	cudaFree(qualityD);
	cudaFree(dpY_D);
	cudaFree(dpX_D);
	cudaFree(dpY_sD);
	cudaFree(dpX_sD);
	
}



// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

Matrix AllocateMatrix_Pre(int height, int width,int init) // 1 is file read/ 0 is just allocation
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;
	FILE *fp;
	fp = fopen("Pre_RF.inp","r");
	cudaMallocHost((void**)&M.elements, size*sizeof(float));

	if(init)
	{
		for(unsigned int i = 0; i < M.width * M.height; i++)
		{
			fscanf(fp,"%f",&M.elements[i]);
		}
	}
	return M;
}


Matrix AllocateMatrix_Post(int height, int width,int init) // 1 is file read/ 0 is just allocation
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;
	FILE *fp;
	fp = fopen("Post_RF.inp","r");
	cudaMallocHost((void**)&M.elements, size*sizeof(float));

	if(init)
	{
		for(unsigned int i = 0; i < M.width * M.height; i++)
		{
			fscanf(fp,"%f",&M.elements[i]);
		}
	}
	return M;
}


// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size,
			cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size,
			cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix(Matrix* M)
{
	cudaFree(M->elements);
	M->elements = NULL;
}

// Free a host Matrix
void FreeMatrix(Matrix* M)
{
	//free(M->elements);
	cudaFreeHost(M->elements);
	M->elements = NULL;
}
