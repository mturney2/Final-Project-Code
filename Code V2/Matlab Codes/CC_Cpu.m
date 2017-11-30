%%

clear all
close all

fileId = fopen('trialNumbers.inp');
c = textscan(fileId,'%f');
data = c{1,1};

row = 5;
col = 5;
pre = zeros(row,col);

j = 1;

for i=1:row
    pre(i,1:col)=data(j:j-1+col);
    j = j + col;
end

post = pre;

kerX = 5;
kerY = 5;
sX = 3;
rangeX = floor(sX/2);
sY = 5;
rangeY = floor(sY/2);
numX = 5;
numY = 1;

pad_pre_y = floor(kerY/2); 
halfY = floor(kerY/2);
pad_pre_x = floor(kerX/2);
halfX = floor(kerX/2);

pad_post_y = pad_pre_y + floor(sY/2);
pad_post_x = pad_pre_x + floor(sX/2);

pad_pre = padarray(pre,[pad_pre_y,pad_pre_x]);
pad_post = padarray(post,[pad_post_y,pad_post_x]);

kernelcenY = 2 + rangeY;
kernelcenX = 2 + rangeX;
for y=1:numY
    for x=1:numX
        CC = normxcorr2_mex...
                       (pad_pre( kernelcenY - halfY: kernelcenY + halfY,kernelcenX + x - halfX - 1:kernelcenX + x + halfX - 1)...
                       , pad_post(kernelcenY - halfY - rangeY + 1 : 1 + kernelcenY + halfY + rangeY, kernelcenX + x - halfX - rangeX : kernelcenX + x + halfX + rangeX), 'valid' );
    CC_store{y,x}=CC; 
    Pre_mean(y,x) = mean(mean(pad_pre( kernelcenY - halfY: kernelcenY + halfY,kernelcenX + x - halfX - 1:kernelcenX + x + halfX - 1)));
    end
end