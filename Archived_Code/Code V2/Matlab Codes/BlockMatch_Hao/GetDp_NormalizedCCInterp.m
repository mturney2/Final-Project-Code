function Dp=GetDp_NormalizedCCInterp(PreData, PostData, matchPara, Dp, InterpolateBeamFactor)

if nargin~=5
    disp('number of function input arguments error!\n');
    return;
end
[nPoint, nLine]=size(PreData);

% tempMarker=zeros(nPoint, nLine);
% tempMarker=Marker(1:matchPara.BmodeFactor:end,:);
% for k=2:matchPara.BmodeFactor
%     tempMarker=tempMarker+Marker(k:matchPara.BmodeFactor:end,:);
% end
% Marker=tempMarker;

firstLineMovingBeamLine=30;

winLenPoint_1=2*matchPara.halfWinLenPoint;
winBeamLine_1=2*matchPara.halfWinBeamLine;
winBeamLine_MoveRange=matchPara.winBeamLine+2*matchPara.scatterMovingBeamLine;

Win = hanning(matchPara.winLenPoint); 
%Win = ones(matchPara.winLenPoint, 1); 
sWin = repmat(Win, 1, matchPara.winBeamLine+2*matchPara.scatterMovingBeamLine);
Win = repmat(Win,1,nLine);            
nSize=2^(ceil(log10(matchPara.winLenPoint)/log10(2)));
nSize_2=nSize/2;
nBoundaryLine=ceil(matchPara.halfWinBeamLine/InterpolateBeamFactor);

tmp1=zeros(nSize, nLine);
tmp2=zeros(nSize, nLine);

    mCC=2*matchPara.scatterMovingBeamLine+1;
    regionCC=zeros(nSize, mCC);
   
    Dp.yIndex=Dp.yIndex-matchPara.halfWinLenPoint;
    nDpPoint=length(Dp.yIndex);

%     Dp.cc=zeros(3, 3, nDpPoint, nLine);
    ttcc=zeros(3, 3);

    for i=1:nDpPoint
        yStart=Dp.yIndex(i); yEnd=Dp.yIndex(i)+winLenPoint_1;
        
        ttData=PreData(yStart:yEnd,:);
        if matchPara.IsBmode==1
            ttData=ttData-mean(ttData(:));
        end
        preRegionData=ttData.*Win;
        tmp1=fft(preRegionData, nSize);
        PreDataPower=sum(preRegionData.^2); clear preRegionData;
        for curLine=nBoundaryLine+1:360-nBoundaryLine % 360 ******
%             if curLine==359
%             curLine
%             end
            j=InterpolateBeamFactor*(curLine-1)+1;
            xStart=j-matchPara.halfWinBeamLine; xEnd=j+matchPara.halfWinBeamLine;
            tt1=tmp1(:, xStart:xEnd);
            prePower=sum(PreDataPower(xStart:xEnd));
            vStart=xStart-matchPara.scatterMovingBeamLine+Dp.x(i,curLine);
            vEnd=xStart+matchPara.scatterMovingBeamLine+winBeamLine_1+Dp.x(i,curLine);
            uStart=Dp.yIndex(i)-Dp.y(i,curLine);
            uEnd=uStart+winLenPoint_1;
            if vStart>=1 & vEnd<=nLine
                if uStart<1
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(matchPara.winLenPoint-uEnd+1:matchPara.winLenPoint, :)=PostData(1:uEnd, vStart:vEnd);
                elseif uEnd>nPoint
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(1:nPoint-uStart+1, :)=PostData(uStart:nPoint, vStart:vEnd);
                else
                    tmpData=PostData(uStart:uEnd, vStart:vEnd);
                end
            elseif vStart>nLine | vEnd<1
                tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
            elseif vStart<1
                if uStart<1
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(matchPara.winLenPoint-uEnd+1:matchPara.winLenPoint, ...
                        winBeamLine_MoveRange-vEnd+1:winBeamLine_MoveRange)=PostData(1:uEnd, 1:vEnd);
                elseif uEnd>nPoint
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(1:nPoint-uStart+1, winBeamLine_MoveRange-vEnd+1:winBeamLine_MoveRange) ...
                        =PostData(uStart:nPoint, 1:vEnd);
                else
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(:, winBeamLine_MoveRange-vEnd+1:winBeamLine_MoveRange)=PostData(uStart:uEnd, 1:vEnd);
                end
            elseif vEnd>nLine
                if uStart<1
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(matchPara.winLenPoint-uEnd+1:matchPara.winLenPoint, ...
                        1:nLine-vStart+1)=PostData(1:uEnd, vStart:nLine);
                elseif uEnd>nPoint
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(1:nPoint-uStart+1, 1:nLine-vStart+1) ...
                        =PostData(uStart:nPoint, vStart:nLine);
                else
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    tmpData(:, 1:nLine-vStart+1)=PostData(uStart:uEnd, vStart:nLine);
                end
            end
            if matchPara.IsBmode==1
                tmpData=tmpData-mean(tmpData(:));
            end
            postRegionData=tmpData.*sWin;
            tmp2=fft(postRegionData, nSize);
            PostDataPower=sum(postRegionData.^2); clear postRegionData;
            for v=1:mCC
                tt2=tmp2(:, v:v+winBeamLine_1);
                postPower=sum(PostDataPower(v:v+winBeamLine_1));
                regionCC(:, v)=sum(abs(ifft(tt1.*conj(tt2))),2)/sqrt(prePower*postPower);
            end
            regionCC(matchPara.scatterMovingRangePoint+1:end-matchPara.scatterMovingRangePoint, :)=0;
            regionCC=circshift(regionCC, nSize_2);
            [yy, xx]=find(regionCC==max(regionCC(:)));
            yy=fix(median(yy)); xx=fix(median(xx));
            if (yy>=1 & yy<=nSize & xx>=1 & xx<=mCC)
%                 ttcc = regionCC(yy-1:yy+1, xx-1:xx+1);
                if yy>1 & yy<nSize & xx>1 & xx<mCC
                    [deltaX, deltaY] = QuadSurfaceFit (regionCC(yy-1:yy+1, xx-1:xx+1));
                elseif yy>1 & yy<nSize
                    deltaX=0;
                    deltaY=subSampleFit (regionCC(yy-1:yy+1, xx));
                elseif  xx>1 & xx<mCC
                    deltaX=subSampleFit (regionCC(yy, xx-1:xx+1));
                    deltaY=0;
                else
                    deltaX=0;
                    deltaY=0;
                end
                Dp.y(i,curLine)=yy+deltaY;
                Dp.x(i,curLine)=xx+deltaX;
                Dp.y(i,curLine)=Dp.y(i,curLine)-nSize_2;
                Dp.x(i,curLine)=Dp.x(i,curLine)-(matchPara.scatterMovingBeamLine+1);
            else
                ttcc = zeros(3, 3);
                Dp.y(i,curLine)=0;
                Dp.x(i,curLine)=0;
            end            
%             Dp.cc(:,:,i,j)=ttcc;
%             end
        end
    end
    Dp.yIndex=Dp.yIndex+matchPara.halfWinLenPoint;




%--------------------------------------------------------------------
function [x0,y0]= QuadSurfaceFit (SSD)
%Fit function z=a+bx+cy+dx^2+exy+fy^2
%The parameter a to f should be determined by input SSD value
%Of course SSD matrix can be replaced by cross correlation matrix,
%It should be a 3*3 matrix
%
%The Output:
%x0,y0 are the position with fitted minimium SSD value (maxium xcorr value)
%The derivation is in my note book, but there I forget that Matlab uses
%column based index, so here all indeics are transpose indeics of the
%ones in my notes.


sumSSD=sum(SSD(:))/3;
sumVer=sum(SSD);
sumHor=sum(SSD,2);

b=(sumVer(3)-sumVer(1))/6;
c=(sumHor(3)-sumHor(1))/6;
e=(SSD(1)-SSD(3)-SSD(7)+SSD(9))*0.25;
d=(sumVer(1)+sumVer(3))*0.5-sumSSD;
f=(sumHor(1)+sumHor(3))*0.5-sumSSD;

x0=(2*b*f-c*e)/(e*e-4*d*f);
y0=(-2*d*x0-b)/e;

if abs(x0)>=1
    x0=subSampleFit(SSD(2,:));
end

if abs(y0)>=1
    y0=subSampleFit(SSD(:,2)');;
end

%--------------------------------------------------------------------------
function delta=subSampleFit(vector)
delta=0;
if vector(1) ~= vector(3)
    temp= (vector(1)-vector(3))/(2*(vector(1)+vector(3)-2*vector(2)));
    if abs(temp)<1
        delta=temp;
    end
end
