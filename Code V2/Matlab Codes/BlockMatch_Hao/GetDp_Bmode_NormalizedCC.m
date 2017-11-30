function Dp=GetDp_Bmode_NormalizedCC_Marker(PreData, PostData, matchPara, marker)

if nargin~=4
    disp('number of function input arguments error!\n');
    return;
end
[nPoint, nLine]=size(PreData);

winLenPoint_1=2*matchPara.halfWinLenPoint;
winBeamLine_1=2*matchPara.halfWinBeamLine;

% Win = hanning(matchPara.winLenPoint); 
% Win = repmat(Win,1,matchPara.winBeamLine);            
Win = ones(matchPara.winLenPoint, matchPara.winBeamLine);
nBoundaryLine=matchPara.halfWinBeamLine+matchPara.scatterMovingBeamLine;
nBoundary4MovingBeam=2*nBoundaryLine;

Dp.yIndex=matchPara.halfWinLenPoint+1:matchPara.shiftPixel:(nPoint-matchPara.halfWinLenPoint);
nDpPoint=length(Dp.yIndex);
Dp.x=zeros(nDpPoint, nLine);
Dp.y=zeros(nDpPoint, nLine);

nCC=2*matchPara.scatterMovingRangePoint+1;
mCC=2*matchPara.scatterMovingBeamLine+1;
regionCC=zeros(nCC, mCC);
for i=1:nDpPoint %-1
    if i>100000
        xMovingBeamLine=round(mean(Dp.x(i-1, nBoundary4MovingBeam+1:nLine-nBoundary4MovingBeam)));
        yMovingPoint=round(mean(Dp.y(i-1, nBoundary4MovingBeam+1:nLine-nBoundary4MovingBeam)));
    else
        xMovingBeamLine=0;
        yMovingPoint=0;
    end
    currentPoint=Dp.yIndex(i);

    yStart=currentPoint-matchPara.halfWinLenPoint; yEnd=currentPoint+matchPara.halfWinLenPoint;
    yyStart=yStart-yMovingPoint; yyEnd=yEnd-yMovingPoint;
    for j=nBoundaryLine+1:nLine-nBoundaryLine
        xStart=j-matchPara.halfWinBeamLine; xEnd=j+matchPara.halfWinBeamLine;
        if sum(sum(marker(yStart:yEnd, xStart:xEnd)))>0
            tt1=PreData(yStart:yEnd, xStart:xEnd);
            tt1=tt1-mean(tt1(:));
            preRegionData=tt1.*Win;
            PreDataPower=sum(preRegionData(:).^2);
            iy=1;
            xxStart=xStart+xMovingBeamLine; xxEnd=xEnd+xMovingBeamLine;
            for ySearch=-matchPara.scatterMovingRangePoint:matchPara.scatterMovingRangePoint
                ix=1;
                for xSearch=-matchPara.scatterMovingBeamLine:matchPara.scatterMovingBeamLine
                    ypStart=yyStart+ySearch; ypEnd=yyEnd+ySearch;
                    xpStart=xxStart+xSearch; xpEnd=xxEnd+xSearch;

                    if (ypStart>=1 & ypEnd<=nPoint)
                        tt2=PostData(ypStart:ypEnd, xpStart:xpEnd);
                        tt2=tt2-mean(tt2(:));
                        postRegionData=tt2.*Win;
                        PostDataPower=sum(postRegionData(:).^2);
                    elseif (ypEnd<1 | ypStart>ypStart)
                        postRegionData=zeros(matchPara.winLenPoint, matchPara.winBeamLine);
                        PostDataPower=1;
                    elseif (ypStart<1)
                        tt2=zeros(matchPara.winLenPoint, matchPara.winBeamLine);
                        tt2(matchPara.winLenPoint-ypEnd+1:matchPara.winLenPoint, :)=PostData(1:ypEnd, xpStart:xpEnd);
                        tt2=tt2-mean(mean(PostData(1:ypEnd, xpStart:xpEnd)));
                        postRegionData=tt2.*Win;
                        PostDataPower=sum(sum(tt2(matchPara.winLenPoint-ypEnd+1:matchPara.winLenPoint, :).^2));
                    elseif (ypEnd>nPoint)
                        tt2=zeros(matchPara.winLenPoint, matchPara.winBeamLine);
                        tt2(1:nPoint-ypStart+1, :)=PostData(ypStart:nPoint, xpStart:xpEnd);
                        tt2=tt2-mean(mean(PostData(ypStart:nPoint, xpStart:xpEnd)));
                        postRegionData=tt2.*Win;
                        PostDataPower=sum(sum(tt2(1:nPoint-ypStart+1, :).^2));
                    end

                    regionCC(iy, ix)=sum(preRegionData(:).*postRegionData(:))/sqrt(PreDataPower*PostDataPower);
                    ix=ix+1;
                end
                iy=iy+1;
            end

            [yy, xx]=find(regionCC==max(regionCC(:)));
            yy=fix(median(yy)); xx=fix(median(xx));

            if yy>1 & yy<nCC & xx>1 & xx<mCC
                [deltaX, deltaY] = QuadSurfaceFit (regionCC(yy-1:yy+1, xx-1:xx+1));
            elseif yy>1 & yy<nCC
                deltaX=0;
                deltaY=subSampleFit (regionCC(yy-1:yy+1, xx));
            elseif  xx>1 & xx<mCC
                deltaX=subSampleFit (regionCC(yy, xx-1:xx+1));
                deltaY=0;
            else
                deltaX=0;
                deltaY=0;
            end
            Dp.y(i,j)=yy+deltaY;
            Dp.x(i,j)=xx+deltaX;
        else
            Dp.y(i,j)=NaN;
            Dp.x(i,j)=NaN;
        end
    end
end
Dp.y=Dp.y-matchPara.scatterMovingRangePoint-1;
Dp.x=Dp.x-matchPara.scatterMovingBeamLine-1; 

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
