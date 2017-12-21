// Host Side Code for Cross-correlation in GPU

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
//#include "Cross_Data_type.h"
#include "corr2Mex.h"
#include "reduce.cu"

using namespace std;


Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width,int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

int main(int argc,char** argv) {
	// Input Parameters
	
	Matrix M;
	
	M  = AllocateMatrix(7,3, 1);	
	
	// Load Pre and Post to the device
	Matrix Md = AllocateDeviceMatrix(M);
	CopyToDeviceMatrix(Md, M);
    
	// Allocate Space for Pre-Mean
	int *dpY;
	int *dpX;
	float *val;
	float *tempCorrHost;
	cudaMalloc((void **)&dpY,sizeof(float)*1);
	cudaMalloc((void **)&dpX,sizeof(float)*1);
	cudaMalloc((void **)&val,sizeof(float)*1);
	cudaMalloc((void **)&tempCorrHost,sizeof(float)*8*4);
		
	// Device Memory Allocation for Cross-correlation Result
	

	// Setup the execution configuration

	dim3 dimBlock(3,7);
	//dim3 dimBlock(2*parameters.searchX, 2*parameters.searchY);
	dim3 dimGrid(1,1); 
	//int sharedmemsize = 2*parameters.searchX*parameters.searchY*sizeof(float);
	// Launch the device computation threads!
	
    //normXcorr_GPU<<<dimGrid, dimBlock,sharedmemsize>>>(Pred,Postd,CorrD,parameters,preMean,preVar,postMean,postVar);
	reduce<<<dimGrid, dimBlock>>>(Md,tempCorrHost,val,dpX,dpY);
	//Copting SoA from Device to Host
	//CopyFromDeviceMatrix(Corr, Corrd); 
	//cudaMemcpy(CorrH,CorrD,sizeof(SoA_Corr)*parameters.numX*parameters.numY,cudaMemcpyDeviceToHost);
	float *corr;
	int *dp_X;
	int *dp_Y;
	dp_X = (int*)malloc(sizeof(int));
	dp_Y = (int*)malloc(sizeof(int));
	corr = (float*)malloc(sizeof(float));
	cudaMemcpy(dp_Y,dpY,sizeof(float)*1,cudaMemcpyDeviceToHost);
	cudaMemcpy(dp_X,dpX,sizeof(float)*1,cudaMemcpyDeviceToHost);
	cudaMemcpy(corr,val,sizeof(float)*1,cudaMemcpyDeviceToHost);
	
	printf("Y = %d.\n X = %d.\n Correlation is %f",dp_Y[0],dp_X[0],corr[0]);
	
	
	return 0;

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
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
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



