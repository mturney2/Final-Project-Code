
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
// includes, project
using namespace std;

typedef struct{
  int searchY;
  int searchX;
  int kernelY;
  int kernelX;
  float overlap; //between 0-1
} params ;

typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float* elements;
} Matrix;
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"

Matrix normXcorr_CPU(Matrix Pre, Matrix Post, Matrix Corr, params P);
Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width,int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
bool CompareResults(float* A, float* B, int elements, float eps);
void FreeDeviceMatrix(Matrix* M);
void FreeMatrix(Matrix* M);

void CorrelationOnDevice(const Matrix M, const Matrix N, Matrix P, params parameters);

////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
////////////////////////////////////////////////////////////////////////////////
__global__ void CorrelationKernel(Matrix Pre, Matrix Post, Matrix Corr, params P)
{
	int imgWidth = Pre.width;
	int imgHeight = Pre.height;
	int kerX = P.kernelX;
	int kerY = P.kernelY;
	int sX = P.searchX;
	int sY = P.searchY;
	int shiftX = sX/2;
	int shiftY = sY/2;
	float overlap = P.overlap;

	int xB = (blockIdx.x*blockDim.x)*overlap;
	int yB = (blockIdx.y*blockDim.y)*overlap;
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	/*volatile __shared__ float sh[kerX*kerY+(kerY+shiftY)*(kerX+shiftX)];
	
	float *preFrame = sh[0];
	float *postFrame = sh[kerX*kerY]
	
	if */
	
	if(x<sX&&y<sY){
	
	int x1 = x-shiftX;
	int y1 = y-shiftY;
		
	float CC = 0.0;
	
		for(int j=0;j<kerY;j++){			
			for(int i=0;i<kerX;i++){				
				if((x1+i+xB)<imgWidth&&(y1+j+yB)<imgHeight&&(x1+i+xB)>=0&&(y1+j+yB)>=0)					
					CC += Pre.elements[(i+xB)+(j+yB)*imgWidth]*Post.elements[(y1+j+yB)*imgWidth+(x1+i+xB)];
					
				else
					CC += 0.0;
			}
		}
		__syncthreads();
		Corr.elements[x+y*sX] = CC;
	}
}

__global__ void CorrelationKernel_Shared(Matrix Pre, Matrix Post, Matrix Corr, params P)
{
	int imgWidth = Pre.width;
	int imgHeight = Pre.height;
	int kerX = P.kernelX;
	int kerY = P.kernelY;
	int sX = P.searchX;
	int sY = P.searchY;
	int shiftX = sX/2;
	int shiftY = sY/2;
	float overlap = P.overlap;

	int xB = (blockIdx.x*blockDim.x)*overlap;
	int yB = (blockIdx.y*blockDim.y)*overlap;
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	extern __shared__ float sh[];
	
	float *preKern = &sh[0];
	float *postKern = &sh[kerX*kerY];
	
	if(x==0&&y==0){
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				if((i+xB)<imgWidth&&(j+yB)<imgHeight&&(i+xB)>0&&(j+yB)>0)
					preKern[i+j*kerX] = Pre.elements[(i+xB)+(j+yB)*imgWidth]; 
				else
					preKern[i+j*kerX]=0;
			}
		}
	}
	if(x==1&&y==1){
		for (int j=-shiftY; j<(kerY+shiftY); j++)
		{
			for (int i=-shiftX;i<(kerX+shiftX);i++){
				if((i+xB)<imgWidth&&(j+yB)<imgHeight&&(i+xB)>0&&(j+yB)>0)
					postKern[i+j*(kerX+sX)] = Post.elements[(i+xB)+(j+yB)*imgWidth]; 
				else
					postKern[i+j*kerX]=0;
			}
		}		
	}
	__syncthreads();
	
	if(x<sX&&y<sY){
	
	int x1 = x-shiftX;
	int y1 = y-shiftY;
		
	float CC = 0.0;
	
		for(int j=0;j<kerY;j++){			
			for(int i=0;i<kerX;i++){				
					CC += Pre.elements[(i+xB)+(j+yB)*kerX]*Post.elements[(y1+j)*(kerX+sX)+(x1+i)];
			}
		}
		__syncthreads();
		Corr.elements[x+y*sX] = CC;
	}
}

__global__ void normXcorr_GPU(Matrix Pre, Matrix Post, Matrix Corr, params P)
{
	int imgWidth = Pre.width;
	int imgHeight = Pre.height;
	int kerX = P.kernelX;
	int kerY = P.kernelY;
	int sX = P.searchX;
	int sY = P.searchY;
	int shiftX = sX/2;
	int shiftY = sY/2;
	float overlap = P.overlap;

	int xB = (blockIdx.x*blockDim.x)*overlap;
	int yB = (blockIdx.y*blockDim.y)*overlap;
	int x = threadIdx.x;
	int y = threadIdx.y;
	
	extern __shared__ float sh[];
	
	float *preMean = &sh[0];
	float *preVar = &sh[1];
	float *postMean = &sh[2];
	float *postVar = &sh[2+sX*sY];
/////////PRE//////////////	
	if(x==0&&y==0){
		//Mean
		float sumPre = 0.0;
		float sum2Pre = 0.0;
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				sumPre += Pre.elements[(i+xB)+(j+yB)*imgWidth];
			}
		}
		preMean[0] = sumPre/(kerX*kerY);
		
		//Variance
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				sum2Pre += (Pre.elements[(i+xB)+(j+yB)*imgWidth]-preMean[0])*(Pre.elements[(i+xB)+(j+yB)*imgWidth]-preMean[0]);
			}
		}
		preVar[0] = sum2Pre;
	}
__syncthreads();
	
	if(x<sX&&y<sY){
	int x1 = x-shiftX;
	int y1 = y-shiftY;
	
	float sumPost = 0.0;
	float sum2Post = 0.0;
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				if((y1+j+yB)<imgHeight&&(y1+j+yB)>=0&&(x1+i+xB)<imgWidth&&(x1+i+xB)>=0)
					sumPost += Post.elements[(y1+j+yB)*(imgWidth)+(x1+i+xB)];
				else
					sumPost += 0;
			}
		}
		postMean[x+y*sX] = sumPost/((kerY)*(kerX));
		
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				if((y1+j+yB)<imgHeight&&(y1+j+yB)>=0&&(x1+i+xB)<imgWidth&&(x1+i+xB)>=0)
					sum2Post += (Post.elements[(y1+j+yB)*(imgWidth)+(x1+i+xB)]-postMean[x+y*sX])*(Post.elements[(y1+j+yB)*(imgWidth)+(x1+i+xB)]-postMean[x+y*sX]);
				else
					sum2Post += (postMean[x+y*sX])*(postMean[x+y*sX]);
			}
		}
		postVar[x+y*sY]=sum2Post;	
	}
	__syncthreads();
	
	if(x<sX&&y<sY){
	
	int x1 = x-shiftX;
	int y1 = y-shiftY;
		
	float CC = 0.0;
	
		for(int j=0;j<kerY;j++){			
			for(int i=0;i<kerX;i++){	
				if((y1+j+yB)<imgHeight&&(y1+j+yB)>=0&&(x1+i+xB)<imgWidth&&(x1+i+xB)>=0)			
					CC += (Pre.elements[(i+xB)+(j+yB)*imgWidth]-preMean[0])*(Post.elements[(y1+j+yB)*(imgWidth)+(x1+i+xB)]-postMean[y*sX+x]);
				else
					CC += (preMean[0])*(postMean[x+y*sX]);
			}
		}
		Corr.elements[x+y*sX] = CC/sqrt(preVar[0]*postVar[y*sX+x]);
		//Corr.elements[x+y*sX] = postVar[y*sX+x];
	}
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
	//int imageWidth = 5;
	//int imageHeight = 5;
	float OVERLAP = 1.0;
	Matrix  Pre;
	Matrix  Post;
	Matrix  Corr;


/*	if(argc != 4) 
	{
		printf("Usage %s Size\n",argv[0]);
		return 1;

	}*/

	int imageWidth = atoi(argv[1]);
	int imageHeight = atoi(argv[2]);
	int SEARCH_X = atoi(argv[3]);
	int SEARCH_Y = atoi(argv[4]);
	int KERNEL_X = atoi(argv[5]);
	int KERNEL_Y = atoi(argv[6]);
	imageWidth = 5;
	imageHeight = 5;
	SEARCH_X = 3;
	SEARCH_Y = 3;
	KERNEL_X = 5;
	KERNEL_Y = 5;
	params parameters = {SEARCH_X,SEARCH_Y,KERNEL_X,KERNEL_Y,OVERLAP};
	Pre  = AllocateMatrix(imageHeight,imageWidth, 1);
	Post  = AllocateMatrix(imageHeight,imageWidth, 1);		
	Corr  = AllocateMatrix(SEARCH_Y,SEARCH_X, 0);
	float gpuTime=0.f;

	// M * N on the device
	float  elapsedTime_inc;
	cudaEvent_t startEvent_inc, stopEvent_inc;
	cudaEventCreate(&startEvent_inc);
	cudaEventCreate(&stopEvent_inc);
	cudaEventRecord(startEvent_inc,0); // starting timing for inclusive  
	
	CorrelationOnDevice(Pre, Post, Corr, parameters);

	
  cudaEventRecord(stopEvent_inc,0);  //ending timing for inclusive
	cudaEventSynchronize(stopEvent_inc);   
	cudaEventElapsedTime(&elapsedTime_inc, startEvent_inc, stopEvent_inc);
	gpuTime = elapsedTime_inc;
	
	float sum = 0.0;
	for(int k =0;k<SEARCH_Y*SEARCH_X;k++)
		sum += Corr.elements[k]*Corr.elements[k];
	
	float norm = sqrt(sum);
	
	printf("Time is %f\n\n",gpuTime);
	
	for(int j=0;j<SEARCH_Y;j++){
		printf("\n");	
		for(int i=0;i<SEARCH_X;i++){
			printf("%.3f ",Corr.elements[i+j*SEARCH_X]);
		}
	}
		//printf("%f\n",Corr.elements[i]/norm);

	// Free matrices
	FreeMatrix(&Pre);
	FreeMatrix(&Post);
	FreeMatrix(&Corr);
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void CorrelationOnDevice(const Matrix Pre, const Matrix Post, Matrix Corr, params parameters)
{
	// Load M and N to the device
	Matrix Pred = AllocateDeviceMatrix(Pre);
	CopyToDeviceMatrix(Pred, Pre);
	Matrix Postd = AllocateDeviceMatrix(Post);
	CopyToDeviceMatrix(Postd, Post);

	// Allocate P on the device
	Matrix Corrd = AllocateDeviceMatrix(Corr);
	CopyToDeviceMatrix(Corrd, Corr);

	// Setup the execution configuration

dim3 dimBlock(parameters.searchX, parameters.searchY);
dim3 dimGrid(parameters.kernelX, parameters.kernelY);
//int smemSizeLarge = parameters.kernelX*parameters.kernelY+(parameters.kernelY+parameters.searchY)*(parameters.kernelX+parameters.searchX)*sizeof(float);
int smemSize = 2+2*parameters.searchX*parameters.searchY*sizeof(float);
	// Launch the device computation threads!
	
  normXcorr_GPU<<<dimGrid, dimBlock,smemSize>>>(Pred,Postd,Corrd,parameters);
  
	// Read P from the device
	CopyFromDeviceMatrix(Corr, Corrd); 

	// Free device matrices
	FreeDeviceMatrix(&Pred);
	FreeDeviceMatrix(&Postd);
	FreeDeviceMatrix(&Corrd);

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
		for(unsigned int i = 0; i < M.height * M.width; i++)
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

//compare the data stored in two arrays on the host
bool CompareResults(float* A, float* B, int elements, float eps)
{
	for(unsigned int i = 0; i < elements; i++){
		float error = A[i]-B[i];
		if(error>eps){
			return false;
		} 
	}
	return true;
}



