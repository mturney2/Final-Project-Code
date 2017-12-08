#include <stdio.h>
#include <stdlib.h>


// C-Code for Quad Surface Fit
//Fit function z=a+bx+cy+dx^2+exy+fy^2
//The parameter a to f should be determined by input SSD value
//Of course SSD matrix can be replaced by cross correlation matrix,
//It should be a 3*3 matrix

//The Output:
//x0,y0 are the position with fitted minimium SSD value (maxium xcorr value)

float SSD[9] = {.3,.5,.4,.1,.92,.4,.22,.123,.2};
//float sumVer[3] = {0.0,0.0,0.0};
//float sumHor[3] = {0.0,0.0,0.0};
int size = 3*3;


float subsamplefit(float* vector){
	float delta = 0.0;
	float temp = 0.0;
	if(vector[0]!=vector[0]){
		temp = (vector[0]-vector[2])/(2*(vector[0]+vector[2]-2*vector[1]));
		if(abs(temp)<1){
			delta = temp;
		}
	}
	return delta;
}


int main()
{
	

	float tempSum = 0.0, sumSSD = 0.0 ;
	float b,c,e,d,f,xo,yo;
	float *vector,*sumVer,*sumHor;
	
	vector = (float*)malloc(3*sizeof(float));
	sumVer = (float*)malloc(3*sizeof(float));
	sumHor = (float*)malloc(3*sizeof(float));

	for(int i=0;i<size;i++){
	tempSum +=SSD[i];
	}
	sumSSD = tempSum/3;



	for(int col=0;col<3;col++){
		for(int row=0;row<3;row++){
			sumVer[col]+=SSD[row*3+col];
		}
	}

	for(int row=0;row<3;row++){
		for(int col=0;col<3;col++){
			sumHor[row]+=SSD[row*3+col];
		}
	}

	b = (sumVer[2] - sumVer[0])/6.0;
	c = (sumHor[2] - sumHor[0])/6.0;
	e = (SSD[0]-SSD[2]-SSD[6]+SSD[8])*0.25;
	d = (sumVer[0]+sumVer[2])*0.5 - sumSSD;
	f = (sumHor[0]+sumHor[2])*0.5 - sumSSD;

	xo = (2*b*f-c*e)/(e*e-4*d*f);
	yo = (-2*d*xo-b)/e;
	
	if(abs(xo)>=1){
		for(int i=0;i<3;i++)
		{
			vector[i]=SSD[3+i];
		}
		xo = subsamplefit(vector);
	}
	
	if(abs(yo)>=1){
		for(int i=0;i<3;i++)
		{
			vector[i]=SSD[1+i*3];
		}
		yo = subsamplefit(vector);
	}

	printf("%f\n",xo);
	printf("%f\n",yo);
	
	return 0;
}



