// Host Side Code for Cross-correlation in GPU

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
//#include "Cross_Data_type.h"
#include "corr2Mex.h" 
#include "normXcorr_GPUKernel_Final.cu"

using namespace std;


Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width,int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

//// Cuda Kernel Call //////

void CorrelationOnDevice(const Matrix Pre, const Matrix Post, params parameters,float *quality,float *dpX, float *dpY)
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

	normXcorr_GPU<<<dimGrid, dimBlock, parameters.kernelX*parameters.kernelY*sizeof(float)>>>(Pred,Postd,CorrD,parameters,preMean,preVar,postMean,postVar);

	int smemSize = 1024*sizeof(int)+1024*sizeof(float);
	MaxElement<<<dimGrid,1024,smemSize>>>(CorrD,parameters,qualityD,dpX_D,dpY_D);
	
	int numthreads = 512;
	int numblocks = (parameters.numX*parameters.numY + numthreads -1)/numthreads;
	subsample<<<numblocks,numthreads>>>(CorrD,parameters,qualityD,dpX_D,dpY_D,dpX_sD,dpY_sD);
	
	//cudaMemcpy(CorrH,CorrD,sizeof(float)*parameters.numX*parameters.numY*parameters.searchX*parameters.searchY,cudaMemcpyDeviceToHost);
	cudaMemcpy(quality,qualityD,sizeof(float)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
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
	cudaFree(preMean);
	cudaFree(preVar);
	cudaFree(postMean);
	cudaFree(postVar);

	//FreeDeviceMatrix(&Corrd);

}



// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

Matrix AllocateMatrix(int height, int width,int init) // 1 is file read/ 0 is just allocation
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;
	FILE *fp;
	fp = fopen("trialNumbers.inp","r");
	// don't allocate memory on option 2
	M.elements = (float*) malloc(size*sizeof(float));
	
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
	free(M->elements);
	M->elements = NULL;
}



