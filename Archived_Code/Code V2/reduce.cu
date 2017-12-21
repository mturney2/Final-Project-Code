__global__ void normXcorr_GPU(Matrix M,float *quality, float *dP_X, float *dP_Y)
{

	float *tempCorr;
	int numThreadsX = 5;
	int numThreadsY = 5;
	
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
		for(int ys; ys<numThreadsY; ys++){
			for(int xs; xs<numThreadsX; xs++){
				if((xs<sX)&&(yx<sY)){
					tempCorr[ys*numThreadsX+xs]=M.elements[xs+ys*sX];
				}
				else{
					tempCorr[ys*numThreadsX+xs]=0.0;
				}
			}
		}
	}
	__syncthreads();
	
	//Find Max of each column
	while(numThreadsY>1){
		numThreadsY = numThreadsY/2;
		if( (ty < numThreadsY) && (tempCorr[ty*sX] < tempCorr[(ty+numThreadsY)*sX]) ){
  	// push the bigger element to the top half of each column
  	tempCorr[ty] = tempCorr[(ty+numThreads)*sX];
  	}
	}
	__syncthreads();
	//The max of each column should be at the top of each column now. Therefore we can find max of just the first row.
	if(ty==0){
		while(numThreadsX>1){
			numThreadsX = numThreadsX/2;
			if((tx<numThreadsX)&&(tempCorr[tx]<tempCorr[tx+numThreadsX])){
				tempCorr[tx] = tempCorr[tx+numThreadsX];
			}
		}
	}
	__syncthreads();
	//Now the max value should reside in the top left index of tempCorr. tempCorr[0]
	//Now we need to find which index originally held that value.
	if(tempCorr[0]==Corr[tx+ty*sX]){
		quality[bx+by*gridDim.x]=tempCorr[0];
		dP_X[bx+by*gridDim.x] = tx;
		dP_Y[bx+by*gridDim.x] = ty;
	}
	__syncthreads();
}