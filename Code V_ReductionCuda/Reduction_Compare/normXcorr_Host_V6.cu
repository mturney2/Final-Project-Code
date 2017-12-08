// Host Side Code for Cross-correlation in GPU

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include "corr2Mex.h"
#include "normXcorr_GPUKernel_V4.cu"

using namespace std;


Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width,int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void CorrelationOnDevice(const Matrix M, const Matrix N, float *CorrH, params parameters,float *qualityH);


int main(int argc,char** argv) {
	// Input Parameters
	
	if(argc!=9)
	{
		printf("Usage %s Parameters missing\n",argv[0]);
		return 1;
	}
	
	int imageWidth = atoi(argv[1]);
	int imageHeight = atoi(argv[2]);
	int SEARCH_X = atoi(argv[3]);
	int SEARCH_Y = atoi(argv[4]);
	int KERNEL_X = atoi(argv[5]);
	int KERNEL_Y = atoi(argv[6]);
	int numX = atoi(argv[7]);
	int numY = atoi(argv[8]);
	
	/*
	int imageWidth = 31;
	int imageHeight = 31;
	int SEARCH_X = 5;
	int SEARCH_Y = 5;
	int KERNEL_X = 11;
	int KERNEL_Y = 11;
	int numX = 1;
	int numY = 1;
	*/
	
	
	int DisplacementSize = numX*numY;
	int Corr_size = SEARCH_X*SEARCH_Y;
	Matrix Pre;
	Matrix Post;
	
	float OVERLAP = 50.0;
	
	params parameters = {SEARCH_Y,SEARCH_X,KERNEL_Y,KERNEL_X,OVERLAP,numX,numY};
	Pre  = AllocateMatrix(imageHeight,imageWidth, 1);
	Post  = AllocateMatrix(imageHeight,imageWidth, 1);	
	float gpuTime=0.f;
	
	// Allocating Host-side Memory for Cross-correlation

	float *CorrH;
	//CorrH = (float*)malloc(Corr_size*DisplacementSize*sizeof(float));
	cudaMallocHost((void**)&CorrH, Corr_size*DisplacementSize*sizeof(float));

	float *qualityH;
	qualityH = (float*) malloc(sizeof(float)*parameters.numX*parameters.numY);	

	float  elapsedTime_inc;
	cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventRecord(startEvent_inc,0); // starting timing for inclusive  
	
	CorrelationOnDevice(Pre, Post, CorrH, parameters, qualityH); // Execution Model for GPU is set up in this function

	
  cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);   
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);
	gpuTime = elapsedTime_inc;
	
	// Printing Cross-correlation Matrix for Block:0
	//for(int h=0;h<DisplacementSize;h++){
	/*int h =DisplacementSize - 1;
		for(int z=0;z<SEARCH_X;z++){
			for(int g=0;g<SEARCH_Y;g++){
				printf("%0.4f ",CorrH[g+SEARCH_X*(z+SEARCH_Y*h)]);
			}
		printf("\n");	
		}
		printf("\n");
	//}*/
	
	for(int h=0;h<DisplacementSize;h++){
		for(int g=0;g<SEARCH_Y;g++){
			for(int z=0;z<SEARCH_X;z++){
				printf("%f ",CorrH[(h*SEARCH_Y+g)*SEARCH_X+z]);
			}
		printf("\n");	
		}
		printf("\n");
	}
	
	
	
	printf("\n");
	
	// Printing for Quality Verification
	printf("%f\n",qualityH[0]);
	
	printf("\n");
	printf("Elasped Time = %f\n",gpuTime);
	
	// Free matrices
	FreeMatrix(&Pre);
	FreeMatrix(&Post);
	cudaFreeHost(&CorrH);
	cudaFreeHost(&qualityH);
	return 0;
	
}

//// Cuda Kernel Call //////

void CorrelationOnDevice(const Matrix Pre, const Matrix Post, float *CorrH, params parameters,float *qualityH)
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
	
		
	// Temporary host corr to find max
	float *tempCorrHost;
	int modx = parameters.searchX%2;
	int mody = parameters.searchY%2;
	cudaMalloc((void **)&tempCorrHost,sizeof(float)*(parameters.searchX+modx)*(parameters.searchY+mody)*parameters.numX*parameters.numY);
	
  // CC Value Matrix
	float *qualityD;
	cudaMalloc((void **)&qualityD,sizeof(float)*parameters.numX*parameters.numY);
	
	// Device Memory Allocation for Cross-correlation Result
	float *CorrD;
	cudaMalloc((void **)&CorrD,sizeof(float)*parameters.numX*parameters.numY*parameters.searchX*parameters.searchY);
	
	//cudaMalloc((SoA_Corr **)&CorrD,sizeof(SoA_Corr)*parameters.numX*parameters.numY);

	// Setup the execution configuration

	dim3 dimBlock(parameters.searchX, parameters.searchY);
	dim3 dimGrid(parameters.numX, parameters.numY); 
	// Launch the device computation threads!
	
	normXcorr_GPU<<<dimGrid, dimBlock>>>(Pred,Postd,CorrD,parameters,preMean,preVar,postMean,postVar,tempCorrHost,qualityD);
	cudaMemcpy(CorrH,CorrD,sizeof(float)*parameters.numX*parameters.numY*parameters.searchX*parameters.searchY,cudaMemcpyDeviceToHost);
	cudaMemcpy(qualityH,qualityD,sizeof(float)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	
	// Free device matrices
	FreeDeviceMatrix(&Pred);
	FreeDeviceMatrix(&Postd);
	cudaFree(CorrD);
    
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
	//fp = fopen("trialNumbers.inp","r");
	fp = fopen("Real_Data_US.inp","r");
	// don't allocate memory on option 2
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
	//cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice,0);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	//cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost,0);
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
	cudaFreeHost(M->elements);
	M->elements = NULL;
}



