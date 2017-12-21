%% Multi-Level Strain Estimation for in-vivo mouse long axis data
% Run using updated Multi-Level_2D_NCC function
% Complied by Rashid-Al-Mukaddim, UW-Madison
% Update Date: 8-3-2017
% email: mukaddim@wisc.edu

clear all
close all

cur_dir=pwd;
addpath(strcat(cur_dir,'/BlockMatch_Hao'));
addpath(strcat(cur_dir,'/normxcorr2_mex_ALL'));

%filterfolder='E:\Rashid\Multi_Level_Update\Hybrid Multilevel\OBNLMpackage\';
%addpath(genpath(filterfolder));

matDataFolder='/export/home/rmukaddim/LA Paper Long Axis Data/';  
addpath(matDataFolder);

vevoRfFolder='/export/home/rmukaddim/RfPackage/';
addpath(genpath(vevoRfFolder));

% resultFolder='E:\Rashid\Multi_Level_Update\2D NCC Mouse 00\';
% addpath(resultFolder);

fnameBase='Mouse 03 LA.iq';
inname=[matDataFolder fnameBase];

%graphicFolder='E:\Rashid\Multi_Level_Update\Hybrid Multilevel\altmany-export_fig-5be2ca4\';
%addpath(graphicFolder);

% Drops the extension from filename and preserves onlythe name
len=length(fnameBase);
Res_name=fnameBase(1:len-3); % DROP EXTENSION


%% Physical Dimension Generation

frameNumber = 163;
[ rf,Idata,Qdata,sampleFreq,centerFreq,param] = IQ2RF(inname, frameNumber );

Depth=param.BmodeDepth-param.BmodeDepthOffset; %mm
y=linspace(0,Depth,size(rf,1));
halfwidth=param.BmodeWidth/2;
x=linspace(-halfwidth,halfwidth,size(rf,2));
y_spacing=mean(diff(y));
x_spacing=mean(diff(x));

%% Parameter Definitions
c=1540; %m/s
lamda = (c/centerFreq)*1000; % mm


% Kernel Dimensions
searchParams.kernelY(1)=round((5*lamda)/y_spacing); % previous value : 06
searchParams.kernelY(2)=round((10*lamda)/y_spacing);
searchParams.kernelY(3)=round((4*lamda)/y_spacing);
searchParams.kernelY(4)=round((2*lamda)/y_spacing);
ind=find(mod(searchParams.kernelY,2)==0);
searchParams.kernelY(ind)=searchParams.kernelY(ind)+1; % Making sure the kernel dimension is an odd number
searchParams.kernelX=[9 7 5 3]; % Number of Beam Lines
searchParams.overlap=50; % Overlap

%median filter size
searchParams.medianfiltY = [3,3,5,5];
searchParams.medianfiltX = [3,3,7,7];

%CC threshold 
searchParams.threshold=[.3,.5,.6,.75]; 

%strain limit
searchParams.axial_ey=[.9,.8,.4,.3];
searchParams.axial_ex=[.9,.8,.4,.3];
searchParams.lateral_ey=[2,2,2,2];
searchParams.lateral_ex=[2,2,2,2];

%iterative strain limit smooth number
searchParams.maxRepeatNum=[32,16,8,8];

%cubic spline smooth parameter
searchParams.splineParam= [.8,.5,.4,.4];



%% Main Loop for calculating Incremental Displacements
startFrame = 161;
endFrame = 162;
nstep = 1;
index=1;
for curFrame=startFrame:nstep:endFrame-1
    fprintf('processing %d and %d/%d ', curFrame,curFrame+1,endFrame);
    fprintf('\n');
    % Pre-RF Data
    pre_frame=curFrame;
    pre=IQ2RF(inname, pre_frame );
 
    % Post-RF Data
    post_frame=curFrame+1;
    post=IQ2RF(inname, post_frame );

    
    % Multi-Level-Estimation
    [ dpY, dpX, quality, CC_surface, geomRf, geomRf_pad] = Multi_level_2D_NCC_One_Level( pre,post,searchParams,y_spacing,x_spacing);
    

    % Creating Structure before saving the result

    field1 = 'axial_displacement';      value1 = dpY;
    field2 = 'lateral_displacement';    value2 = dpX;
    field3 = 'cc_map';                  value3 = quality;
    field4 = 'rf_geom';                 value4 = geomRf ;
%     field5 = 'geom';                    value5 = geom;

    multi_level_results = struct(field1,value1,field2,value2,field3,value3,field4,value4);
%     cc_final_sur = CC_surface;
    
    % Saving screen shots
    
    h=figure;
    set(h,'units','inches','position',[0 0 16 6]);
    subplot(1, 2, 1); 
    imagesc(x(geomRf.startX:geomRf.stopX),y(geomRf.startY:geomRf.stepY:geomRf.stopY),dpY);
    title(sprintf('Inc. Axial Disp. Frame: %d and %d',curFrame,curFrame+1),'FontSize',16);
    xlabel('Width (mm)','FontWeight','bold','FontSize',16);
    ylabel('Depth (mm)','FontWeight','bold','FontSize',16);
    colormap(jet); colorbar;
    
    subplot(1, 2, 2); 
    imagesc(x(geomRf.startX:geomRf.stopX),y(geomRf.startY:geomRf.stepY:geomRf.stopY),dpX);
    title(sprintf('Inc. Lateral Disp. Frame: %d and %d',curFrame,curFrame+1),'FontSize',16);
    xlabel('Width (mm)','FontWeight','bold','FontSize',16);
    ylabel('Depth (mm)','FontWeight','bold','FontSize',16);
    colormap(jet); colorbar;
%     export_fig(h, sprintf('%sDisp_Frame_%d.jpg',resultFolder, curFrame+1));
%     close(h);
    
    % Saving the results
    
%     outFileName=sprintf('%sDisp_Frame_%d.mat', resultFolder, curFrame+1);
%     save(outFileName,'multi_level_results' );
%     outFileName=sprintf('%sCC_Surface_Frame_%d.mat', resultFolder, curFrame+1);
%     save(outFileName,'cc_final_sur' );
    
    fprintf('Processing Finished betn Frame: %d and %d',curFrame,curFrame+1);
    fprintf('\n');
    
end
