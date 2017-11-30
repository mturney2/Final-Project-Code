__global__ void reduce(Matrix test,float *tempCorr, float *quality, int *dP_X, int *dP_Y)
{

	int sX = 3; // 3
	int sY = 7; // 3
  
  
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//tempCorr = &test.elements[0];
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
					tempCorr[ys*numThreadsX+xs]=test.elements[xs+ys*sX];
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
		modY = numThreadsY%2;
		numThreadsY = numThreadsY/2 + modY;
		if( (ty < numThreadsY) && (tempCorr[tx+ty*sX] < tempCorr[tx+(ty+numThreadsY)*sX]) ){
  	// push the bigger element to the top half of each column
  	tempCorr[tx+ty*sX] = tempCorr[(ty+numThreadsY)*sX+tx];
  	}
	}
	__syncthreads();
	//The max of each column should be at the top of each column now. Therefore we can find max of just the first row.
	if(ty==0){
		while(numThreadsX>1){
			modX = numThreadsX%2;
			numThreadsX = numThreadsX/2 + modX;
			if((tx<numThreadsX)&&(tempCorr[tx]<tempCorr[tx+numThreadsX])){
				tempCorr[tx] = tempCorr[tx+numThreadsX];
			}
		}
	}
	__syncthreads();
	//Now the max value should reside in the top left index of tempCorr. tempCorr[0]
	//Now we need to find which index originally held that value.
	if(tempCorr[0]==test.elements[tx+ty*sX]){
		quality[0]=tempCorr[0];
		dP_X[0] = tx;
		dP_Y[0] = ty;
	}
	__syncthreads();
}