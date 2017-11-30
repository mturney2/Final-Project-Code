function Dp=GetDp_NormalizedCC_Marker(PreData, PostData, matchPara, marker, Dp)

if nargin<4 & margin>5 % should be nargin??
    disp('number of function input arguments error!\n');
    return;
end
[nPoint, nLine]=size(PreData);

winLenPoint_1=2*matchPara.halfWinLenPoint;
winBeamLine_1=2*matchPara.halfWinBeamLine;
winBeamLine_MoveRange=matchPara.winBeamLine+2*matchPara.scatterMovingBeamLine;

Win = hanning(matchPara.winLenPoint); 
%Win = ones(matchPara.winLenPoint, 1); 
sWin = repmat(Win, 1, matchPara.winBeamLine+2*matchPara.scatterMovingBeamLine);
Win = repmat(Win,1,nLine);            
nSize=2^(ceil(log10(matchPara.winLenPoint)/log10(2)));
nSize_2=nSize/2;
nBoundaryLine=matchPara.halfWinBeamLine;

n4average=10;
tmp1=zeros(nSize, nLine);
tmp2=zeros(nSize, nLine);
if nargin==4
    Dp.yIndex=matchPara.halfWinLenPoint+1:matchPara.shiftPixel:(nPoint-matchPara.halfWinLenPoint);
    nDpPoint=length(Dp.yIndex);
    Dp.x=zeros(nDpPoint, nLine);
    Dp.y=zeros(nDpPoint, nLine);
    Dp.iniX=zeros(nDpPoint, nLine);
    Dp.iniY=zeros(nDpPoint, nLine);

    mCC=2*matchPara.scatterMovingBeamLine+1;
    regionCC=zeros(nSize, mCC);
    nBoundaryLine=matchPara.halfWinBeamLine;
    nBoundary4MovingBeam=2*nBoundaryLine;
    for i=1:nDpPoint %-1
        currentPoint=Dp.yIndex(i);
        yStart=currentPoint-matchPara.halfWinLenPoint; yEnd=currentPoint+matchPara.halfWinLenPoint;
        ttData=PreData(yStart:yEnd,:);
        if matchPara.IsBmode==1
            ttData=ttData-mean(ttData(:));
        end
        preRegionData=ttData.*Win;
        tmp1=fft(preRegionData, nSize);
        PreDataPower=sum(preRegionData.^2); clear preRegionData;

        for j=nBoundaryLine+1:nLine-nBoundaryLine
            xStart=j-matchPara.halfWinBeamLine; xEnd=j+matchPara.halfWinBeamLine;
            if marker(currentPoint, j)==1
                %if i==2 & j==211
                %    i
                %end
                if i>1
                    if marker(Dp.yIndex(i-1), j)==1
                        ind=find(marker(Dp.yIndex(i-1), :)>0);
                        indloc=find(diff(ind)>1);
                        ll=[min(ind), ind(indloc+1)];
                        hh=[ind(indloc), max(ind)];
                        ind=find(j>ll & j<hh);
                        if length(ind)>0
                            lowB=max([ll(ind), j-n4average]);
                            highB=min([hh(ind), j+n4average]);
                            xMovingBeamLine=round(mean(Dp.x(i-1, lowB:highB)));
                            yMovingPoint=round(mean(Dp.y(i-1,  lowB:highB)));
                        else
                            xMovingBeamLine=0;
                            yMovingPoint=0;
                        end
                    else
                        xMovingBeamLine=0;
                        yMovingPoint=0;
                    end
                else
                    xMovingBeamLine=0;
                    yMovingPoint=0;
                end
                Dp.iniX(i,j)=xMovingBeamLine;
                Dp.iniY(i,j)=yMovingPoint;

                uStart=yStart-yMovingPoint;
                uEnd=yEnd-yMovingPoint;
                ttData=PostData(uStart:uEnd,:);
                if matchPara.IsBmode==1
                    ttData=ttData-mean(ttData(:));
                end
                postRegionData=ttData.*Win;
                tmp2=fft(postRegionData, nSize);
                PostDataPower=sum(postRegionData.^2); clear postRegionData;

                tt1=tmp1(:, xStart:xEnd);
                prePower=sum(PreDataPower(xStart:xEnd));
                wStart=xStart+xMovingBeamLine-matchPara.scatterMovingBeamLine;
                wEnd=xStart+xMovingBeamLine+matchPara.scatterMovingBeamLine;
                nCC=1;
                for w=wStart:wEnd
                    if (w>=1 & w+winBeamLine_1<=nLine)
                        tt2=tmp2(:, w:w+winBeamLine_1);
                        postPower=sum(PostDataPower(w:w+winBeamLine_1));
                    elseif (w+winBeamLine_1<1 | w>nLine)
                        tt2=zeros(nSize,matchPara.winBeamLine);
                        postPower=1;
                    elseif (w<1)
                        tt2=zeros(nSize,matchPara.winBeamLine);
                        tt2(:, matchPara.winBeamLine-w-winBeamLine_1+1:matchPara.winBeamLine)=tmp2(:, 1:w+winBeamLine_1);
                        postPower=sum(PostDataPower(1:w+winBeamLine_1));
                    elseif (w+winBeamLine_1>nLine)
                        tt2=zeros(nSize,matchPara.winBeamLine);
                        tt2(:, 1:nLine-w+1)=tmp2(:, w:nLine);
                        postPower=sum(PostDataPower(w:nLine));
                    end
                    regionCC(:, nCC)=sum(abs(ifft(tt1.*conj(tt2))),2)/sqrt(prePower*postPower);
                    nCC=nCC+1;
                end
                regionCC(matchPara.scatterMovingRangePoint+1:end-matchPara.scatterMovingRangePoint, :)=0;
                regionCC=circshift(regionCC, nSize_2);
                [yy, xx]=find(regionCC==max(regionCC(:)));
                yy=fix(median(yy)); xx=fix(median(xx));
                if (yy>=1 & yy<=nSize & xx>=1 & xx<=mCC)
                    Dp.y(i,j)=yy;
                    Dp.x(i,j)=xx;
                    Dp.x(i,j)=Dp.x(i,j)+(xMovingBeamLine-matchPara.scatterMovingBeamLine-1);
                    Dp.y(i,j)=Dp.y(i,j)-nSize_2;
                    Dp.y(i,j)=Dp.y(i,j)+yMovingPoint;
                else
                    ttcc = zeros(3, 3);
                    Dp.y(i,j)=0;
                    Dp.x(i,j)=0;
                end
            end
        end
    end
    Dp.x=round(Dp.x);
    Dp.y=round(Dp.y);
elseif nargin==5
    mCC=2*matchPara.scatterMovingBeamLine+1;
    regionCC=zeros(nSize, mCC);
   
    Dp.yIndex=Dp.yIndex-matchPara.halfWinLenPoint;
    nDpPoint=length(Dp.yIndex);
    for i=1:nDpPoint
        yStart=Dp.yIndex(i); yEnd=Dp.yIndex(i)+winLenPoint_1;
        
        ttData=PreData(yStart:yEnd,:);
        if matchPara.IsBmode==1
            ttData=ttData-mean(ttData(:));
        end
        preRegionData=ttData.*Win;
        tmp1=fft(preRegionData, nSize);
        PreDataPower=sum(preRegionData.^2); clear preRegionData;
        for j=nBoundaryLine+1:nLine-nBoundaryLine
            xStart=j-matchPara.halfWinBeamLine; xEnd=j+matchPara.halfWinBeamLine;
            if sum(sum(marker(yStart:yEnd, xStart:xEnd)))>1
                tt1=tmp1(:, xStart:xEnd);
                prePower=sum(PreDataPower(xStart:xEnd));
                vStart=xStart-matchPara.scatterMovingBeamLine+Dp.x(i,j);
                vEnd=xStart+matchPara.scatterMovingBeamLine+winBeamLine_1+Dp.x(i,j);
                uStart=Dp.yIndex(i)-Dp.y(i, j); %%% special for direct cross-correlation
                uEnd=uStart+winLenPoint_1;
                if vStart>=1 & vEnd<=nLine
                    if uStart>=1 & uEnd<=nPoint
                        tmpData=PostData(uStart:uEnd, vStart:vEnd);
                    elseif uStart>nPoint & uEnd<1
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    elseif uStart<1
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(matchPara.winLenPoint-uEnd+1:matchPara.winLenPoint, :)=PostData(1:uEnd, vStart:vEnd);
                    elseif uEnd>nPoint
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(1:nPoint-uStart+1, :)=PostData(uStart:nPoint, vStart:vEnd);
                    end
                elseif vStart>nLine | vEnd<1
                    tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                elseif vStart<1
                    if uStart>=1 & uEnd<=nPoint
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(:, winBeamLine_MoveRange-vEnd+1:winBeamLine_MoveRange)=PostData(uStart:uEnd, 1:vEnd);
                    elseif uStart>nPoint & uEnd<1
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    elseif uStart<1
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(matchPara.winLenPoint-uEnd+1:matchPara.winLenPoint, ...
                            winBeamLine_MoveRange-vEnd+1:winBeamLine_MoveRange)=PostData(1:uEnd, 1:vEnd);
                    elseif uEnd>nPoint
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(1:nPoint-uStart+1, winBeamLine_MoveRange-vEnd+1:winBeamLine_MoveRange) ...
                            =PostData(uStart:nPoint, 1:vEnd);
                    end
                elseif vEnd>nLine
                    if uStart>=1 & uEnd<=nPoint
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(:, 1:nLine-vStart+1)=PostData(uStart:uEnd, vStart:nLine);
                    elseif uStart>nPoint & uEnd<1
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                    elseif uStart<1
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(matchPara.winLenPoint-uEnd+1:matchPara.winLenPoint, ...
                            1:nLine-vStart+1)=PostData(1:uEnd, vStart:nLine);
                    elseif uEnd>nPoint
                        tmpData=zeros(matchPara.winLenPoint,winBeamLine_MoveRange);
                        tmpData(1:nPoint-uStart+1, 1:nLine-vStart+1) ...
                            =PostData(uStart:nPoint, vStart:nLine);
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
                    Dp.y(i,j)=yy+deltaY;
                    Dp.x(i,j)=xx+deltaX;
                    Dp.y(i,j)=Dp.y(i,j)-nSize_2;
                    Dp.x(i,j)=Dp.x(i,j)-(matchPara.scatterMovingBeamLine+1);
                else
                    Dp.y(i,j)=0;
                    Dp.x(i,j)=0;
                end
            else
                Dp.y(i,j)=0;
                Dp.x(i,j)=0;
            end
        end
    end
    Dp.yIndex=Dp.yIndex+matchPara.halfWinLenPoint;
end


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
