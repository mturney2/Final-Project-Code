#include "mex.h"
#include "corr2Mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs!=9)
        mexErrMsgTxt("Invalid Number of input arguments");
    
    if(nlhs!=3)
        mexErrMsgTxt("Invalid Number of outputs");
    
    if(!mxIsSingle(prhs[0]) && !mxIsSingle(prhs[1]))
        mexErrMsgTxt("Input Vector Data type must ne single");
    
    int numRowsPre = (int)mxGetM(prhs[0]);
    int numColsPre = (int)mxGetN(prhs[0]);
    int numRowsPost = (int)mxGetM(prhs[1]);
    int numColsPost = (int)mxGetN(prhs[1]);
    
    // Geting Data from the Pre and Post Matrix
    float* pre = (float*)mxGetData(prhs[0]);
    float* post = (float*)mxGetData(prhs[1]);
	
	// Creating Matrix Structure reuired for the Correlation on Device Function
	///////// Pre Matrix //////////
	Matrix Pre;
	Pre.width = Pre.pitch = numColsPre;
	Pre.height = numRowsPre;
	int sizepre = Pre.width * Pre.height;
	Pre.elements = NULL;
	Pre.elements = (float*) malloc(sizepre*sizeof(float));
	
	int index = 0;
	for(int i=0;i<Pre.height;i++){
		for(int j=0;j<Pre.width;j++){
			Pre.elements[index] = pre[j*Pre.height+i]; //Pre is in cloumn major but read in a row major fashion
			index++;
		}
	}
	
	///////// Post Matrix //////////
    Matrix Post;
	Post.width = Post.pitch = numColsPre;
	Post.height = numRowsPost;
	int sizepost = Post.width * Post.height;
	Post.elements = NULL;
	Post.elements = (float*) malloc(sizepost*sizeof(float));
	
	int index1 = 0;
	for(int i=0;i<Post.height;i++){
		for(int j=0;j<Post.width;j++){
			Post.elements[index1] = post[j*Post.height+i]; //Post is in cloumn major but read in a row major fashion
			index1++;
		}
	}
	
	//////////// Input Parameters /////////
	// parameter structure containing Algorithm Parameters
	int SEARCH_X = (int)mxGetScalar(prhs[2]);
	int SEARCH_Y = (int)mxGetScalar(prhs[3]);
	int KERNEL_X = (int)mxGetScalar(prhs[4]);
	int KERNEL_Y = (int)mxGetScalar(prhs[5]);
	int numX = (int)mxGetScalar(prhs[6]);
	int numY = (int)mxGetScalar(prhs[7]);
	float OVERLAP = (float)mxGetScalar(prhs[8]);
	int DisplacementSize = numX*numY;
	int Corr_size = SEARCH_X*SEARCH_Y;
	
	// Printing for verification of Parameters
	mexPrintf("Search_x = %d\n",SEARCH_X);
	mexPrintf("Search_y = %d\n",SEARCH_Y);
	mexPrintf("Kernel_x = %d\n",KERNEL_X);
	mexPrintf("Kernel_y = %d\n",KERNEL_Y);
	mexPrintf("numX = %d\n",numX);
	mexPrintf("numY = %d\n",numY);
	mexPrintf("Overlap = %f\n",OVERLAP);
	////// Printing Ends
	
	params parameters = {SEARCH_Y,SEARCH_X,KERNEL_Y,KERNEL_X,OVERLAP,numX,numY};
	
	int ndim = 2;
	int dims[2] = {DisplacementSize,1};
	
	
    //plhs[0]=mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS,mxREAL);
    //float* CorrH = (float*)mxGetData(plhs[0]);
    plhs[0]=mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS,mxREAL);
    float* quality = (float*)mxGetData(plhs[0]);
	plhs[1]=mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS,mxREAL);
    float* dpX = (float*)mxGetData(plhs[1]);
	plhs[2]=mxCreateNumericArray(ndim, dims, mxSINGLE_CLASS,mxREAL);
    float* dpY = (float*)mxGetData(plhs[2]);
    
    CorrelationOnDevice(Pre, Post, parameters,quality, dpX, dpY);
    //computeGold(result, image, mask, numRowsA, numColsA);
}