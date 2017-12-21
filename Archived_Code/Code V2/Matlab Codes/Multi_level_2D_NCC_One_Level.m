function [ dpY, dpX, quality, CC_surface, geomRf, geomRf_pad] = Multi_level_2D_NCC_One_Level( pre,post,searchParams,y_spacing,x_spacing)
%Multi_Level_2D_NCC - Two-dimensional Multi-Level Strain Estimation for
%                     Discontinuous Tissue - Shi & Varghese
% 
% Input Args: 
% pre and post are identically sized arrays for motion tracking.
% preEnvelope and postEnvelope are B_mode images after filtering using
% Bayesian NLM filter and anisotropic diffusion
% Search Params is a structure with the following fields:
%     overlap                 => percentage, 75 gives a 75% overlap
%     kernelY                 => pixels, must be an odd number, to have a well
%                                defined center pixel.  Array with four entries.  Kernel size at all 4 levels.
%     kernelX                 => pixels, must be an odd number.  Array with four entries.  Kernel size at all 4 levels.
%     medianfiltY,medianfiltX => Dimension of 2-D Median Filter at all four levels; Example: medianfiltY = [3,3,3,3];
%     threshold               => Cross-correlation threshold Value; Example: threshold=[.3,.5,.6,.75];
%     axial_ey                => Axial Strain Limit in y direction; Example: axial_ey=[.9,.8,.4,.3];
%     axial_ex                => Axial Strain Limit in x direction; Example: axial_ex=[.9,.8,.4,.3];
%     lateral_ey              => Lateral Strain Limit in x direction; Example: lateral_ey=[2,2,2,2];
%     lateral_ex              => Lateral Strain Limit in x direction; Example: lateral_ex=[2,2,2,2];
%     maxRepeatNum            => Iterative strain limit smooth number; Example: maxRepeatNum=[32,16,8,8];
%     splineParam             => %cubic spline smooth parameter; Example: splineParam=.4;

% Output Args:
%   Axial_disp   => Cell arary containing displacement axially in mm for all four levels
%   Lateral_disp => Cell arary containing displacement laterally in mm for all four levels
%   Quality      => Cell array containing the value of cross-correlation for all four levels
%   CC_surface   => Regional 2-D CC values for all levels; Used for
%                   invesgating the CC function
%   geomRF       => start and stop for all levels of displacements
%   geom         => start and stop for making the plot

% Current Limitation: For the Cardaic Simulation work, no downsampling is
% done at the first level. In future, the downsampling part will be updated


%% Parameter section
l = 1;
rangeY = round(searchParams.kernelY*.25);
rangeY(1)=120;
if rangeY(4) < 2
    rangeY(4) = 2;
end
rangeX = [8,4,4,4];

halfY = (searchParams.kernelY - 1)/2;  
halfX = (searchParams.kernelX - 1)/2;

%median filter size
medianfiltY = searchParams.medianfiltY; 
medianfiltX = searchParams.medianfiltX;

%CC threshold 
threshold=searchParams.threshold; 

%strain limit
%axial_ey=searchParams.axial_ey;
%axial_ex=searchParams.axial_ex;
%lateral_ey=searchParams.lateral_ey;
%lateral_ex=searchParams.lateral_ex;

%iterative strain limit smooth number
maxRepeatNum=searchParams.maxRepeatNum;

%cubic spline smooth parameter
splineParam= searchParams.splineParam;

%% Boundary Considerations
% Work out the boundaries of the displacement grids so tracking doesn't go
% out of bounds, there is no zero padding

%rfY =  size(pre, 1);
rfX = size(pre,2);

maxRangeY = sum(rangeY(1));
maxRangeX = sum(rangeX(1));
stepY = round(searchParams.kernelY*(100-searchParams.overlap)/100 );


% Boundary Considerations and Padding

pad_level_y = maxRangeY + halfY(1);
pad_level_x = maxRangeX + halfX(1);

% Padded data for 2-D NCC
pad_pre_rf=padarray(pre,[pad_level_y,pad_level_x]);
pad_post_rf=padarray(post,[pad_level_y,pad_level_x]);


startYrf_pad = 1 + maxRangeY + halfY(1);
startX_pad = 1 + maxRangeX + halfX(1);
stopYmax_pad = size(pad_pre_rf,1) - halfY(1) - maxRangeY;


numY = floor( (stopYmax_pad - startYrf_pad) / stepY(1) ) + 1;


stopYrf_pad = startYrf_pad + stepY(1)*(numY-1);
stopX_pad = size(pad_pre_rf,2) - halfX(1) - maxRangeX;

numX = stopX_pad - startX_pad + 1;

kernCenterY_pad = startYrf_pad:stepY:stopYrf_pad;
kernCenterX_pad = startX_pad:stopX_pad;

% Original Kernel locations for interpolation
startYrf=1; 
stopYrf = 1 + stepY*(numY-1);
startX=1;
stopX=rfX;
%kernCenterY = 1:stepY(l):stopYrf;
%kernCenterX = 1:rfX;
        
        
geomRf = struct( 'stopY', stopYrf, 'startY', startYrf, 'stepY', stepY, 'startX', startX, 'stopX', stopX );
geomRf_pad = struct( 'stopY', stopYrf_pad, 'startY', startYrf_pad, 'stepY', stepY, 'startX', startX_pad, 'stopX', stopX_pad );

iniDpY = zeros(numY, numX);
iniDpX = zeros(numY, numX);
dpY = zeros(numY, numX);
dpX = zeros(numY, numX);
quality = zeros(numY, numX);
CC_surface=cell(numY,numX);
%% Perform cross correlations using 2-D NCC

tic;

% Call kernel here --> CC_surface

for y = 1:numY  %y,x are coordinates on dp grid
    for x = 1:numX

           %use envelope data with largest kernel size,
           %papers show envelope performs better with large kernel size
           %RF data performs better with smaller kernel size

            
            CC = normxcorr2_mex...
                   (pad_pre_rf( kernCenterY_pad(y) - halfY(1) : kernCenterY_pad(y) + halfY(1), kernCenterX_pad(x) - halfX(1) : kernCenterX_pad(x) + halfX(1))...
                   , pad_post_rf(kernCenterY_pad(y) + iniDpY(y,x) - halfY(1) - rangeY(1) : kernCenterY_pad(y) + iniDpY(y,x) + halfY(1) + rangeY(1), kernCenterX_pad(x) + iniDpX(y,x) - halfX(1) - rangeX(1) : kernCenterX_pad(x) + iniDpX(y,x) + halfX(1) + rangeX(1)), 'valid' );
          


        %find correlation function peak, then obtain subsample
        %displacement
        
         CC_surface{y,x}=CC; % Stroing the cross-correlation functions
        
        if sum(CC(:))==0
            dpY(y,x) = 0;  %convert index to dp and add in subsample fit
            dpX(y,x) = 0;
            quality(y,x) = 1;
        else
        %[tmpQual, tempIndex] = max(CC(:));
        [tmpQual] = max(CC(:));
        [subY,subX]=find(CC==tmpQual);
        %[subY, subX] = ind2sub( size(CC), tempIndex );

        if subY >1 && subY < size(CC,1) && subX > 1 && subX < size(CC,2)     %In the middle of the CC Matrix
            [deltaX, deltaY] = QuadSurfaceFit (CC(subY-1:subY+1, subX - 1:subX + 1));
        elseif subY >1 && subY < size(CC,1)  %y is in the middle, x at edge
            deltaX=0;
            deltaY = subSampleFit (CC(subY-1:subY+1, subX)');
        elseif subX > 1 && subX < size(CC,2)  %x is in the middle, y at edge
            deltaX = subSampleFit (CC(subY, subX-1:subX+1));
            deltaY=0;
        else
            deltaX=0;
            deltaY=0;
        end

        dpY(y,x) = subY - rangeY(1) - 1 + deltaY + iniDpY(y,x);  %convert index to dp and add in subsample fit
        dpX(y,x) = subX - rangeX(1) - 1 + deltaX + iniDpX(y,x);
        quality(y,x) = tmpQual;
        end
    end
end
toc;
    
    

% CC-Filtering 

[m1,n1]=size(dpY);

%to do the CC filtering, all maxCC value less than threshold will cause
%corresponding displacement estimation be replaced by surrounding
%values' average
mask=ones(m1,n1);
mask(quality<threshold(1))=0;
findMaskPos=find(mask==1);  
[y,x] = ind2sub(size(dpY),findMaskPos); %keep track of which entries are above the cut

xi=(1:n1);
yi=(1:m1)';

%remove the entries that are nan (not a number)
dpY(isnan(dpY)) = zeros(size(dpY(isnan(dpY))));
dpX(isnan(dpX)) = zeros(size(dpX(isnan(dpX))));

try
    %make griddata using the valid dp estimates
    tempDpY2=griddata(x,y,dpY(findMaskPos), xi,yi);
    tempDpX2=griddata(x,y,dpX(findMaskPos), xi,yi);

    %replace the nans with 9999
    tempDpY2(isnan(tempDpY2)) = 99999*ones(size(dpY(isnan(tempDpY2))));
    tempDpX2(isnan(tempDpX2)) = 99999*ones(size(dpX(isnan(tempDpX2))));

    %replace the 
    dpY=tempDpY2.*(abs(tempDpY2)<10000)+dpY.*(abs(tempDpY2)>10000);
    dpX=tempDpX2.*(abs(tempDpX2)<10000)+dpX.*(abs(tempDpX2)>10000);

    tempDpY2=Filter2D_Copy(dpY,[medianfiltY(l),medianfiltX(l)]);
    tempDpX2=Filter2D_Copy(dpX,[medianfiltY(l),medianfiltX(l)]);
catch
    tempDpY2=Filter2D_Copy(dpY,[medianfiltY(l),medianfiltX(l)]);
    tempDpX2=Filter2D_Copy(dpX,[medianfiltY(l),medianfiltX(l)]);
end

%To do the strain filtering, all strain values larger than limit will
%be filtered
flag=1;
repeatNum=1;
while repeatNum < maxRepeatNum(l) && flag==1
    try
        dy_y=tempDpY2(2:m1,:)-tempDpY2(1:m1-1,:);
        dy_y(m1,:)=0;
        dy_x=tempDpY2(:,2:n1)-tempDpY2(:,1:n1-1);
        dy_x(:,n1)=0;
        maskY=(abs(dy_y)<ady).*(abs(dy_x)<adx);

        dx_y=tempDpX2(2:m1,:)-tempDpX2(1:m1-1,:);
        dx_y(m1,:)=0;
        dx_x=tempDpX2(:,2:n1)-tempDpX2(:,1:n1-1);
        dx_x(:,n1)=0;
        maskX=(abs(dx_y)<ldy).*(abs(dx_x)<ldx);

        mask=maskY.*maskX;
        findMaskPos=find(mask==1);   
        x=ceil(findMaskPos/m1);  %x positions of all the mask values
        y=findMaskPos-floor((findMaskPos-0.5)/m1)*m1;  %y positions of the mask values
        xi=(1:n1);          %all possible x
        yi=(1:m1)';         %all possible y
        tempDpY2(isnan(tempDpY2)) = zeros(size(tempDpY2(isnan(tempDpY2))));
        tempDpX2(isnan(tempDpX2)) = zeros(size(tempDpX2(isnan(tempDpX2))));

        tempDpY3=griddata(x,y,tempDpY2(findMaskPos), xi,yi);
        tempDpX3=griddata(x,y,tempDpX2(findMaskPos), xi,yi);
        tempDpY3(isnan(tempDpY2)) = 99999*ones(size(tempDpY3(isnan(tempDpY3))));
        tempDpX3(isnan(tempDpX2)) = 99999*ones(size(tempDpX3(isnan(tempDpX3))));
        tempDpY2=tempDpY3.*(abs(tempDpY3)<10000)+tempDpY2.*(abs(tempDpY3)>10000);
        tempDpX2=tempDpX3.*(abs(tempDpX3)<10000)+tempDpX2.*(abs(tempDpX3)>10000);
    catch
        flag=0;
    end         
    repeatNum=repeatNum+1;
end

% Cubic Spline Smoothing
[m2,n2]=size(tempDpY2);
xxx={(1:m2),(1:n2)};
dpY2 = csaps(xxx,tempDpY2, splineParam(l),xxx);
dpX2 = csaps(xxx,tempDpX2, splineParam(l),xxx); 

dpY=y_spacing.*dpY2;
dpX=x_spacing.*dpX2;


end


