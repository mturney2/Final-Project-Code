# Accelerating Ultrasound Elasticity Imaging with a CUDA-MATLAB Approach

## Repository Contents  
* Archived Code: Initial implementation and versions from a major update. 
* Final Code: The final version of the code which will be used in our lab. Available with and without surface fitting. 
* Demo_C, Demo_MATLAB: The final version of the code made available as demo. 
* Miscellaneous: Literature resources, meeting notes, report files, image files for this page, images. 

# Instructions for Executing Demo Code

## Prerequisites 
1. There are two options to execute the project code. On GPU with MATLAB or by executing directly on the GPU as we have typically done for this course. MATLAB is perferred to maintain a similar algorithm framework. Both of the codes have been developed in the Varghese Ultrasound Lab's GPU, the Tesla K40. However, we have also tested the C version in the euler99 node(sample image below). The timings and accuracy reported in the course paper is based on Tesla K40. 

## Steps to Run MATLAB Demo (preferred)
1. Clone the code repository `./Demo_MATLAB` and download the files to a local directory. 
2. Demo_MATLAB contains a compiled MEX binary `Corr2GPUMex_Final.mexa64` which is called in `main.m` for displacement estimation. The MEX was compiled for a 64-bit Linux OS. 
3. If neccessary, the MEX can be recompiled using the following two lines: 
```
   > nvcc -O0 -std=c++11 -c normXcorr_Host_Mex_Final.cu -Xcompiler -fPIC -I/usr/local/MATLAB/R2013a/extern/include;
   > mex Corr2GPUMex_Final.cpp normXcorr_Host_Mex_Final.o -lcudart -L"/local/linux/cuda"
```
4. Run `main.m` in MATLAB. `main.m` will load two pre/post frames of ultrasound images and perform the displacement estimation using the GPU code.  
5. Sample Output:  
![Image of CODE_1](https://github.com/mturney2/Final-Project-Code/blob/master/Miscellaneous/Images/matlabOutput.PNG)

  
## Steps to Run C/CUDA Demo
0. A C demo is also included in case the system doesn't have MATLAB support. 
1. Clone the code repository `./Demo_C` and download the files to a local directory. `Demo_C` contains the neccessary input files. 
2. `Demo_C` contains a precompiled binary `normXcorr2_FC` which can be executed using the following syntax: 
```
   > ./normXcorr2_FC 244 6224 244 6224 17 49 9 161 220 75
```

The input arguments associated with the execution call are: 
 * Width of pre image
 * Height of pre image
 * Width of post image
 * Height of post image
 * Shift in X direction
 * Shift in Y direction
 * Kernel width
 * Kernel height
 * Number of displacement points in X direction
 * Number of displacement points in Y direction
3. `Demo_C` also contains a Makefile which can be used to recompile the code, if neccessary. 

4. Sample Output:  
![Image of CODE_1](https://github.com/mturney2/Final-Project-Code/blob/master/Miscellaneous/Images/eulerOut.PNG)

## Sample Displacement Estimation Results

![Image of CODE_1](https://github.com/mturney2/Final-Project-Code/blob/master/Miscellaneous/Images/displacement_estimated.png)

