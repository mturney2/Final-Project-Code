% Testing the mex routine

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
pre = single(pre);
post = single(post);

%% Estimation parameters

% GPU Implementation
kerX = 5;
kerY = 5;
sX = 3;
sY = 5;
numX = 5;
numY = 1;
overlap = 50.0;

CorrResult = Corr2GPUMex(pre,post,sX,sY,kerX,kerY,numX,numY,overlap);


