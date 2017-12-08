// Host Side Code for Cross-correlation in GPU

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
//#include "Cross_Data_type.h"
#include "corr2Mex.h"
#include "normXcorr_GPUKernel.cu"

using namespace std;


Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width,int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void CorrelationOnDevice(const Matrix M, const Matrix N, float *CorrH, params parameters);


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
	/*SoA_Corr *CorrH;
	CorrH = (SoA_Corr *)malloc(sizeof(SoA_Corr)*DisplacementSize);
	for(int k=0;k<DisplacementSize;k++){
		CorrH[k].Corr_Points = (float*) malloc(Corr_size*sizeof(float));
	}*/
	float *CorrH;
	CorrH = (float*)malloc(Corr_size*DisplacementSize*sizeof(float));
	

	float  elapsedTime_inc;
	cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventRecord(startEvent_inc,0); // starting timing for inclusive  
	
	CorrelationOnDevice(Pre, Post, CorrH, parameters); // Execution Model for GPU is set up in this function

	
    cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);   
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);
	gpuTime = elapsedTime_inc;
	
	// Printing Cross-correlation Matrix for Block:0
	for(int h=0;h<DisplacementSize;h++){
		for(int z=0;z<SEARCH_X;z++){
			for(int g=0;g<SEARCH_Y;g++){
				printf("%f ",CorrH[g+SEARCH_X*(z+SEARCH_Y*h)]);
			}
		printf("\n");	
		}
		printf("\n");
	}
	
	printf("\n");
	
	// Free matrices
	FreeMatrix(&Pre);
	FreeMatrix(&Post);
	return 0;
	
}

//// Cuda Kernel Call //////

void CorrelationOnDevice(const Matrix Pre, const Matrix Post, float *CorrH, params parameters)
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
	
	// Allocate SoA on the device ?????
	float *CorrD;
	cudaMalloc((void **)&CorrD,sizeof(float)*parameters.numX*parameters.numY*parameters.searchX*parameters.searchY);
	
	//cudaMalloc((SoA_Corr **)&CorrD,sizeof(SoA_Corr)*parameters.numX*parameters.numY);

	// Setup the execution configuration

	dim3 dimBlock(parameters.searchX, parameters.searchY);
	//dim3 dimBlock(2*parameters.searchX, 2*parameters.searchY);
	dim3 dimGrid(parameters.numX, parameters.numY); 
	int sharedmemsize = 2*parameters.searchX*parameters.searchY*sizeof(float);
	// Launch the device computation threads!
	
    normXcorr_GPU<<<dimGrid, dimBlock,sharedmemsize>>>(Pred,Postd,CorrD,parameters,preMean,preVar);
  
	//Copting SoA from Device to Host
	//CopyFromDeviceMatrix(Corr, Corrd); 
	//cudaMemcpy(CorrH,CorrD,sizeof(SoA_Corr)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	cudaMemcpy(CorrH,CorrD,sizeof(float)*parameters.numX*parameters.numY*parameters.searchX*parameters.searchY,cudaMemcpyDeviceToHost);
	
	// Free device matrices
	FreeDeviceMatrix(&Pred);
	FreeDeviceMatrix(&Postd);
	cudaFree(CorrD);
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



