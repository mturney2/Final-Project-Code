function [ Axial_disp, Lateral_disp, Quality, CC_surface, geomRf, geomRf_pad, geom ] = Multi_Level_2D_NCC_In_Vivo_Pad( pre,post,preEnvelope,postEnvelope,searchParams,y_spacing,x_spacing)
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


% 
% Update History: 
% 1. Fixed the out of bound tracking issue

%% Parameter section

rangeY = round(searchParams.kernelY*.15);
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
axial_ey=searchParams.axial_ey;
axial_ex=searchParams.axial_ex;
lateral_ey=searchParams.lateral_ey;
lateral_ex=searchParams.lateral_ex;

%iterative strain limit smooth number
maxRepeatNum=searchParams.maxRepeatNum;

%cubic spline smooth parameter
splineParam= searchParams.splineParam;

%% Boundary Considerations
% Work out the boundaries of the displacement grids so tracking doesn't go
% out of bounds, there is no zero padding

rfY =  size(pre, 1);
rfX = size(pre,2);

maxRangeY = sum(rangeY(1));
maxRangeX = sum(rangeX(1));

stepY = round(searchParams.kernelY*(100-searchParams.overlap)/100 );
startYrf = zeros(4,1);
stopYrf = zeros(4,1);
startX = zeros(4,1);
stopX = zeros(4,1);
numY = zeros(4,1);
numX = zeros(4,1);


for l = 1:4
    
    if l == 1
        pad_level_y{l} = maxRangeY + halfY(1);
        pad_level_x{l} = maxRangeX + halfX(1);
        
        % Padded data for 2-D NCC
        pad_pre_rf{l}=padarray(pre,[pad_level_y{l},pad_level_x{l}]);
        pad_pre_env{l}=padarray(preEnvelope,[pad_level_y{l},pad_level_x{l}]);
        
        pad_post_rf{l}=padarray(post,[pad_level_y{l},pad_level_x{l}]);
        pad_post_env{l}=padarray(postEnvelope,[pad_level_y{l},pad_level_x{l}]);
        
        startYrf_pad(l) = 1 + maxRangeY + halfY(1);
        startX_pad(l) = 1 + maxRangeX + halfX(1);
        stopYmax_pad = size(pad_pre_rf{l},1) - halfY(1) - maxRangeY;
        
        
        numY(l) = floor( (stopYmax_pad - startYrf_pad(l)) / stepY(1) ) + 1;
        
        
        stopYrf_pad(l) = startYrf_pad(l) + stepY(l)*(numY(l)-1);
        stopX_pad(l) = size(pad_pre_rf{l},2) - halfX(1) - maxRangeX;
        
        numX(l) = stopX_pad(l) - startX_pad(l) + 1;
        
        kernCenterY_pad{l} = startYrf_pad(l):stepY(l):stopYrf_pad(l);
        kernCenterX_pad{l} = startX_pad(l):stopX_pad(l);
        
        % Original Kernel locations for interpolation
        startYrf(l)=1; 
        stopYrf(l) = 1 + stepY(l)*(numY(l)-1);
        startX=1;
        stopX=rfX;
        kernCenterY{l} = 1:stepY(l):stopYrf(l);
        kernCenterX{l} = 1:rfX;
        
        
    else
        
        pad_level_y{l} = sum(rangeY(1:l)) + halfY(l);
        pad_level_x{l} = sum(rangeX(1:l)) + halfX(l);
        
        % Padded data for 2-D NCC
        pad_pre_rf{l}=padarray(pre,[pad_level_y{l},pad_level_x{l}]);
        pad_pre_env{l}=padarray(preEnvelope,[pad_level_y{l},pad_level_x{l}]);
        
        pad_post_rf{l}=padarray(post,[pad_level_y{l},pad_level_x{l}]);
        pad_post_env{l}=padarray(postEnvelope,[pad_level_y{l},pad_level_x{l}]);
        
        startYrf_pad(l) = 1 + sum(rangeY(1:l)) + halfY(l);
        startX_pad(l) = 1 + sum(rangeX(1:l)) + halfX(l);
        stopYmax_pad = size(pad_pre_rf{l},1) - halfY(l) - sum(rangeY(1:l));
        
        
        numY(l) = floor( (stopYmax_pad - startYrf_pad(l)) / stepY(l) ) + 1;
        
        
        stopYrf_pad(l) = startYrf_pad(l) + stepY(l)*(numY(l)-1);
        stopX_pad(l) = size(pad_pre_rf{l},2) - halfX(l) - sum(rangeX(1:l));
        
        numX(l) = stopX_pad(l) - startX_pad(l) + 1;
        
        kernCenterY_pad{l} = startYrf_pad(l):stepY(l):stopYrf_pad(l);
        kernCenterX_pad{l} = startX_pad(l):stopX_pad(l);
        
        % Original Kernel locations for interpolation
        startYrf(l)=1; 
        stopYrf(l) = 1 + stepY(l)*(numY(l)-1);
        startX=1;
        stopX=rfX;
        kernCenterY{l} = 1:stepY(l):stopYrf(l);
        kernCenterX{l} = 1:rfX;
        
        % Old one
%         startYrf(l) = startYrf(l-1) + sum(rangeY(1:l-1));
%         startX(l) = startX(l-1) +  sum(rangeX(1:l-1)) ;
%         
%         stopYmax = kernCenterY{l-1}(end) - halfY(l) -sum(rangeY(1:l-1));
%         numY(l) = floor( (stopYmax - startYrf(l)) / stepY(l) ) + 1;
%         
%         stopYrf(l) = startYrf(l) + stepY(l)*(numY(l)-1);
% %         stopX(l) = rfX - halfX(1) - maxRangeX;
%         stopX(l) = rfX - halfX(l) - sum(rangeX(1:l-1));
%         numX(l) = stopX(l) - startX(l) + 1;
%         
%         kernCenterY{l} = startYrf(l):stepY(l):stopYrf(l);
%         kernCenterX{l} = startX(l):stopX(l);
        
    end
    
end

geomRf = struct( 'stopY', stopYrf, 'startY', startYrf, 'stepY', stepY, 'startX', startX, 'stopX', stopX );
geomRf_pad = struct( 'stopY', stopYrf_pad, 'startY', startYrf_pad, 'stepY', stepY, 'startX', startX_pad, 'stopX', stopX_pad );


%% Perform cross correlations using 2-D NCC

for l = 1:4
    
    preEnvelope = pad_pre_env{l};
    postEnvelope = pad_post_env{l};
      
    pre =  pad_pre_rf{l};
    post =  pad_post_rf{l};
      
    if l == 1
                
        iniDpY = zeros(numY(1), numX(1));
        iniDpX = zeros(numY(1), numX(1) );
       
    end
        dpY = zeros(numY(l) , numX(l) );
        dpX = zeros(numY(l) , numX(l) );
        quality = zeros(numY(l), numX(l) );
    tic;
    for y = 1:numY(l)  %y,x are coordinates on dp grid
        for x = 1:numX(l)
            
               %use envelope data with largest kernel size,
               %papers show envelope performs better with large kernel size
               %RF data performs better with smaller kernel size
              
%                 if l == 1
%                  
%                  CC = normxcorr2_mex...
%                        (preEnvelope( kernCenterY_pad{l}(y) - halfY(l) : kernCenterY_pad{l}(y) + halfY(l), kernCenterX_pad{l}(x) - halfX(l) : kernCenterX_pad{l}(x) + halfX(l)  )...
%                        , postEnvelope(kernCenterY_pad{l}(y) + iniDpY(y,x) - halfY(l) - rangeY(l) : kernCenterY_pad{l}(y) + iniDpY(y,x) + halfY(l) + rangeY(l), kernCenterX_pad{l}(x) + iniDpX(y,x) - halfX(l) - rangeX(l) : kernCenterX_pad{l}(x) + iniDpX(y,x) + halfX(l) + rangeX(l) ), 'valid' );
%                    
%                else
                   CC = normxcorr2_mex...
                       (pre( kernCenterY_pad{l}(y) - halfY(l) : kernCenterY_pad{l}(y) + halfY(l), kernCenterX_pad{l}(x) - halfX(l) : kernCenterX_pad{l}(x) + halfX(l)  )...
                       , post(kernCenterY_pad{l}(y) + iniDpY(y,x) - halfY(l) - rangeY(l) : kernCenterY_pad{l}(y) + iniDpY(y,x) + halfY(l) + rangeY(l), kernCenterX_pad{l}(x) + iniDpX(y,x) - halfX(l) - rangeX(l) : kernCenterX_pad{l}(x) + iniDpX(y,x) + halfX(l) + rangeX(l) ), 'valid' );
%                end
              
            
            %find correlation function peak, then obtain subsample
            %displacement
            
            CC_store{y,x}=CC; % Stroing the cross-correlation functions
            
            if sum(CC(:))==0
                dpY(y,x) = 0;  %convert index to dp and add in subsample fit
                dpX(y,x) = 0;
                quality(y,x) = 1;
            else
            [tmpQual, tempIndex] = max(CC(:));
            [subY, subX] = ind2sub( size(CC), tempIndex );
            
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
                
            dpY(y,x) = subY - rangeY(l) - 1 + deltaY + iniDpY(y,x);  %convert index to dp and add in subsample fit
            dpX(y,x) = subX - rangeX(l) - 1 + deltaX + iniDpX(y,x);
            quality(y,x) = tmpQual;
            end
        end
    end
    toc;
    
    
    % Now perform error correction, quality filter, derivative filter.
%     if l==1
%         dpY2=dpY;
%         dpX2=dpX;
%     else
        
    % CC-Filtering 
    
    [m1,n1]=size(dpY);
    
    %to do the CC filtering, all maxCC value less than threshold will cause
    %corresponding displacement estimation be replaced by surrounding
    %values' average
    mask=ones(m1,n1);
    mask(quality<threshold(l))=0;
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
    
%     end
    %now calculate initial dp for the next level, and reset the dp estimates
    %if l ~= 4  %as long as it isn't the final level
    
    if l ~= 4
    
	%calculate initial displacements for next level.
    [iniDpY, iniDpX] = makeIniDp(dpY2, dpX2, geomRf_pad, l);  %iniDpY, DpX should have integer values.

    %get rid of any displacements that are too large for the current level
    iniDpY(iniDpY > sum(rangeY(1:l))) = sum(rangeY(1:l));
    iniDpY(iniDpY < -sum(rangeY(1:l))) = -sum(rangeY(1:l));
    iniDpX(iniDpX > sum(rangeX(1:l)) ) = sum(rangeX(1:l));
    iniDpX(iniDpX < -sum(rangeX(1:l)) ) = -sum(rangeX(1:l));
    
    end
    
    Axial_disp{l}=dpY2;
    Lateral_disp{l}=dpX2;
    Quality{l}=quality;
    CC_surface{l}=CC_store;
  
end

for l=1:4
    Axial_disp{l}=y_spacing.*Axial_disp{l};
    Lateral_disp{l}=x_spacing.*Lateral_disp{l};
end

spacing = stepY(4);
geom = struct('startY', geomRf.startY(4), 'stopY', geomRf.stopY(4), 'startX', geomRf.startX, 'stopX', geomRf.stopX, 'stepY', spacing);

end



