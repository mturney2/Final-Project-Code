#include "mex.h"
#include "corr2Mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]){
    if(nrhs!=1)
        mexErrMsgTxt("Invalid Number of input arguments");
    
    //if(nlhs!=1)
    //    mexErrMsgTxt("Invalid Number of outputs");
    
    if(!mxIsSingle(prhs[0]))
        mexErrMsgTxt("Input Vector Data type must ne single");
    
    int numRowsPre = (int)mxGetM(prhs[0]);
    int numColsPre = (int)mxGetN(prhs[0]);
    
    // Geting Data from the Pre and Post Matrix
    float* pre = (float*)mxGetData(prhs[0]);
    
	// Creating Matrix Structure required for the Correlation on Device Function
	Matrix Pre;
	Pre.width = Pre.pitch = numColsPre;
	Pre.height = numRowsPre;
	int size = Pre.width * Pre.height;
	Pre.elements = NULL;
	Pre.elements = (float*) malloc(size*sizeof(float));
	
	mexPrintf("number of rows = %d\n",Pre.height);
	mexPrintf("number of cols = %d\n",Pre.width);
	
	int index = 0;
	for(int i=0;i<Pre.height;i++){
		for(int j=0;j<Pre.width;j++){
			Pre.elements[index] = pre[j*Pre.height+i]; //Pre is in cloumn major but read in a row major fashion
			index++;
		}
	}
	
	//Printing for verification
	/*for(int i=0;i<Pre.height;i++){
		for(int j=0;j<Pre.width;j++){
			mexPrintf("%f\n",Pre.elements[i*Pre.width+j]);
			}
		mexPrintf("\n");
	}
    
	//Printing for verification
	/*for(int i=0;i<size;i++){
		mexPrintf("%f\n",Pre.elements[i]);
	}*/
	
	mexPrintf("number of elements = %d\n",size);
	FILE *fp;
	fp = fopen("Real_Data_US.inp","w");
	for(int i=0;i<size;i++){
		fprintf(fp,"%f\n",Pre.elements[i]);
	}
	fclose(fp);
	
    plhs[0]=mxCreateNumericMatrix(size,1,mxSINGLE_CLASS,mxREAL);
    float* result = (float*)mxGetData(plhs[0]);
	for(int i=0;i<size;i++){
		result[i] = Pre.elements[i];
	}
    //plhs[1]=mxCreateNumericMatrix(1,1,mxSINGLE_CLASS,mxREAL);
    //float* inc = (float*)mxGetData(plhs[1]);*/
    
    //void CorrelationOnDevice(const Matrix Pre, const Matrix Post, float *CorrH, params parameters);
    //computeGold(result, image, mask, numRowsA, numColsA);
}