#ifndef _NORM_XCORR_KERNEL_H_
#define _NORM_XCORR_KERNEL_H_

//#include "Cross_Data_type.h"
#include "corr2Mex.h"
////// Kernel For Computing CC Values //////

__global__ void normXcorr_GPU(Matrix Pre, Matrix Post, float *Corr, params P,float *preMean,float *preVar,float *postMean,float *postVar,float *tempCorr,float *quality)
{
	int imgWidth = Pre.width; // Total width of Pre and Post Compression Image (In this case I will try 10)
	int imgHeight = Pre.height; // Total Height of Pre and Post Compression Image (In this case I will try 10)
	int kerX = P.kernelX; // 5
	int kerY = P.kernelY; // 5
	int halfX = kerX/2;
	int halfY = kerY/2;
	int sX = P.searchX; // 3
	int sY = P.searchY; // 3
	/*int numx = P.numX;
	int numy = P.numY;
	int corrsize = numx*numy;*/
	int shiftX = sX/2;
	int shiftY = sY/2;
	//int shiftX = sX;
	//int shiftY = sY;
	float overlap = P.overlap;
	//int stepY = kerY*((100.0 - overlap)/100);
	int stepY;
	if( overlap  < 100){
		stepY = kerY*((100.0 - overlap)/100)+1;
	}
	else{
		stepY = kerY*((100.0 - overlap)/100);
	}
	
  
	int by = blockIdx.y;
	int bx = blockIdx.x;
  
	int start_row = by*stepY - halfY;
	int start_col = bx - halfX ;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	//Corr[bx+by*gridDim.x].Corr_Points = (float*) malloc(corrsize*sizeof(float));
	// Figure it out in Host Side: Appropiately allocate memory
	//extern __shared__ float sh[];
	//float *preMean = &sh[0];
	//float *preVar = &sh[1];
	//float *postMean = &sh[2];
	//float *postVar = &sh[2+sX*sY];
	//float *postMean = &sh[0];
	//float *postVar = &sh[sX*sY];

/////////PRE//////////////	
	if(tx==0&&ty==0){
		//Mean
		float sumPre = 0.0;
		float sum2Pre = 0.0;
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				if((start_row+j)<imgHeight && (start_row+j)>=0 && (start_col+i)<imgWidth && (start_col+i)>=0)
					sumPre += Pre.elements[(i+start_col)+(j+start_row)*imgWidth];
				else
					sumPre += 0;
			}
		}
		preMean[bx+by*gridDim.x] = sumPre/(kerX*kerY);
		
		//Variance
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				if((start_row+j)<imgHeight && (start_row+j)>=0 && (start_col+i)<imgWidth && (start_col+i)>=0)
					sum2Pre += (Pre.elements[(i+start_col)+(j+start_row)*imgWidth]-preMean[bx+by*gridDim.x])*(Pre.elements[(i+start_col)+(j+start_row)*imgWidth]-preMean[bx+by*gridDim.x]);
				else
					sum2Pre += (preMean[bx+by*gridDim.x])*(preMean[bx+by*gridDim.x]);
			}
		}
		preVar[bx+by*gridDim.x] = sum2Pre;
	}
__syncthreads();


//////Post////////
if(tx<sX && ty<sY){	
	int post_start_row = start_row + ty - shiftY;
	int post_start_col = start_col + tx - shiftX;
	
	//Global Index for PostMean and PostVariance
	int bId =  blockIdx.x + blockIdx.y*gridDim.x;
	int ptId = bId*(blockDim.x * blockDim.y)+(threadIdx.y*blockDim.x)+threadIdx.x;
	
	
	float sumPost = 0.0;
	float sum2Post = 0.0;
		for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				if((post_start_row+j)<imgHeight&&(post_start_row+j)>=0&&(post_start_col+i)<imgWidth&&(post_start_col+i)>=0)
					sumPost += Post.elements[(post_start_row+j)*(imgWidth)+(post_start_col+i)];
				else
					sumPost += 0; 
			}
		}
		//postMean[tx+ty*sX] = sumPost/((kerY)*(kerX));
		postMean[ptId] = sumPost/((kerY)*(kerX));
	
	for (int j=0; j<(kerY); j++)
		{
			for (int i=0;i<(kerX);i++){
				if((post_start_row+j)<imgHeight&&(post_start_row+j)>=0&&(post_start_col+i)<imgWidth&&(post_start_col+i)>=0)
					sum2Post += (Post.elements[(post_start_row+j)*(imgWidth)+(post_start_col+i)]-postMean[ptId])*(Post.elements[(post_start_row+j)*(imgWidth)+(post_start_col+i)]-postMean[ptId]);
				else
					sum2Post += (postMean[ptId])*(postMean[ptId]);
			}
		}
		//postVar[tx+ty*sX]=sum2Post;	
		postVar[ptId]=sum2Post;
		
__syncthreads();
	
/////Cross-correlation Calculation/////

	//int post_start_row = start_row + ty - shiftY;
	//int post_start_col = start_col + tx - shiftX;
		
	float CC = 0.0;
	
	/*for(int j=0;j<kerY;j++){			
		for(int i=0;i<kerX;i++){	
			if((post_start_row+j)<imgHeight&&(post_start_row+j)>=0&&(post_start_col+i)<imgWidth&&(post_start_col+i)>=0)			
				CC += (Pre.elements[(i+start_col)+(j+start_row)*imgWidth]-preMean[bx+by*gridDim.x])*(Post.elements[(post_start_row+j)*(imgWidth)+(post_start_col+i)]-postMean[tx+ty*sX]);
			else
				CC += (Pre.elements[(i+start_col)+(j+start_row)*imgWidth] - preMean[bx+by*gridDim.x])*(0.00 - postMean[tx+ty*sX]);
				
		}
	}*/
	
	for(int j=0;j<kerY;j++){			
		for(int i=0;i<kerX;i++){	
			if((post_start_row+j)<imgHeight&&(post_start_row+j)>=0&&(post_start_col+i)<imgWidth&&(post_start_col+i)>=0&&(start_row+j)<imgHeight&&(start_row+j)>=0&&(start_col+i)<imgWidth&&(start_col+i)>=0)			
				CC += (Pre.elements[(i+start_col)+(j+start_row)*imgWidth]-preMean[bx+by*gridDim.x])*(Post.elements[(post_start_row+j)*(imgWidth)+(post_start_col+i)]-postMean[ptId]);
			else if((start_row+j)<imgHeight && (start_row+j)>=0 && (start_col+i)<imgWidth && (start_col+i)>=0)
					CC += (Pre.elements[(i+start_col)+(j+start_row)*imgWidth] - preMean[bx+by*gridDim.x])*(0.00 - postMean[ptId]);
			else if((post_start_row+j)<imgHeight&&(post_start_row+j)>=0&&(post_start_col+i)<imgWidth&&(post_start_col+i)>=0)
					CC += (0.00 - preMean[bx+by*gridDim.x])*(Post.elements[(post_start_row+j)*(imgWidth)+(post_start_col+i)]-postMean[ptId]);
			else
					CC += (0.00 - preMean[bx+by*gridDim.x])*(0.00 - postMean[ptId]);
			
				
		}
	}
		
	//Corr[bx+by*gridDim.x].Corr_Points[ty*sX+tx] = CC/sqrt(preVar[by+bx*gridDim.x]*postVar[ty*sX+tx]); // Cross-correlation Matrix for Each location of Displacement Image
	int xx = bx+by*gridDim.x;
	//Corr[tx+sX*(ty+sY*xx)]=1.00;
	//Corr[tx+sX*(ty+sY*xx)] = CC;
	//Corr[tx+sX*(ty+sY*xx)] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]);
	Corr[(xx*sY+ty)*sX+tx] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]); // Row Major Update
	//Corr[(xx*sX+tx)*sY+ty] = CC/sqrt(preVar[bx+by*gridDim.x]*postVar[ptId]); // Column Major Update	
	if (Corr[(xx*sX+tx)*sY+ty] != Corr[(xx*sX+tx)*sY+ty]) Corr[(xx*sX+tx)*sY+ty] = 0.0;
	//Corr[tx+sX*(ty+sY*xx)] = preMean[xx];
	
	
	//Reduction Ex 3 for Peak Finding
	int numThreadsX = sX;
	int numThreadsY = sY;
	int modY;
	int modX;
	
	
	//Before finding max, we need to pad tempCorr to make even numbers
	if((tx==0)&&(ty==0)){
		//First make dimensions even
		if(numThreadsX%2==1){
			numThreadsX = numThreadsX+1;
		}
		if(numThreadsY%2==1){
			numThreadsY = numThreadsY+1;
		}
		//Populate tempCorr with Corr values, pad with zeros for positions outside of original Corr.
		for(int ys=0; ys<numThreadsY; ys++){
			for(int xs=0; xs<numThreadsX; xs++){
				if((xs<sX)&&(ys<sY)){
					//tempCorr[ys*numThreadsX+xs]=test.elements[xs+ys*sX];
					tempCorr[(xx*numThreadsY+ys)*numThreadsX+xs] = Corr[(xx*sY+ys)*sX+xs];
				}
				else{
					tempCorr[(xx*numThreadsY+ys)*numThreadsX+xs] = 0.0;
				}
			}
		}
	}
	__syncthreads();
	
	//Find Max of each column
	//Cant use changing numThreads based on indexing above, so new variables initialized for reduction operation.
	int nTY = numThreadsY;
	int nTX = numThreadsX;
	while(nTY>1){
		modY = nTY%2;
		nTY = nTY/2 + modY;
		//if( (ty < numThreadsY) && (tempCorr[tx+ty*numThreadsX] < tempCorr[tx+(ty+numThreadsY)*numThreadsX]) ){
		if( (ty < nTY) && (tempCorr[(xx*numThreadsY+ty)*numThreadsX+tx] < tempCorr[(xx*numThreadsY+ty+nTY)*numThreadsX+tx]) ){
  	// push the bigger element to the top half of each column
  	tempCorr[(xx*numThreadsY+ty)*numThreadsX+tx] = tempCorr[(xx*numThreadsY+ty+nTY)*numThreadsX+tx];
  	//tempCorr[tx+ty*sX] = tempCorr[(ty+numThreadsY)*sX+tx];
  	}
	}
	__syncthreads();
	//The max of each column should be at the top of each column now. Therefore we can find max of just the first row.
	if(ty==0){
		while(nTX>1){
			modX = nTX%2;
			nTX = nTX/2 + modX;
			//ty = 0
			if((tx<nTX)&&(tempCorr[(xx*numThreadsY+ty)*numThreadsX+tx]<tempCorr[(xx*numThreadsY+ty)*numThreadsX+tx+nTX])){
				//tempCorr[tx] = tempCorr[tx+numThreadsX];
				tempCorr[(xx*numThreadsY+ty)*numThreadsX+tx]=tempCorr[(xx*numThreadsY+ty)*numThreadsX+tx+nTX];
			}
		}
	}
	__syncthreads();
	//Now the max value should reside in the top left index of tempCorr. tempCorr[0]
	//Now we need to find which index originally held that value.
	if(tempCorr[0]==tempCorr[(xx*numThreadsY)*numThreadsX]){
		quality[0]=tempCorr[0];
		//dP_X[0] = tx;
		//dP_Y[0] = ty;
	}
	__syncthreads();
	/*
	if(tempCorr[(xx*numThreadsY)*numThreadsX]==Corr[(xx*sY+ty)*sX+tx]){
		quality[(xx*numThreadsY)*numThreadsX]=tempCorr[(xx*numThreadsY)*numThreadsX];
		dP_X[(xx*numThreadsY)*numThreadsX] = tx;
		dP_Y[(xx*numThreadsY)*numThreadsX] = ty;
	}
	*/	
}
}

#endif // #ifndef _NORM_XCORR_KERNEL_H_