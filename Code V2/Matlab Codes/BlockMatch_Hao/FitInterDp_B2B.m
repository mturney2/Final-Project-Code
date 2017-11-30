function iniDp=FitInterDp_B2B(Dp, firstStepPara, secondStepPara, nBmodePixel, nBeam);

Dp.yIndex=Dp.yIndex*(firstStepPara.BmodeFactor/secondStepPara.BmodeFactor);
Dp.y=Dp.y*(firstStepPara.BmodeFactor/secondStepPara.BmodeFactor);

nBoundaryLine=4; %firstStepPara.halfWinBeamLine;
Dp.x(:,1:nBoundaryLine)=repmat(mean(Dp.x(:, nBoundaryLine+1:nBoundaryLine+7),2), 1, nBoundaryLine);
Dp.x(:, end-nBoundaryLine+1:end)=repmat(mean(Dp.x(:, end-nBoundaryLine-6:end-nBoundaryLine),2), 1, nBoundaryLine);
Dp.y(:,1:nBoundaryLine)=repmat(mean(Dp.y(:, nBoundaryLine+1:nBoundaryLine+7),2), 1, nBoundaryLine);
Dp.y(:, end-nBoundaryLine+1:end)=repmat(mean(Dp.y(:, end-nBoundaryLine-6:end-nBoundaryLine),2), 1, nBoundaryLine);

iniDp.yIndex=secondStepPara.halfWinLenPoint+1:secondStepPara.shiftPixel:(nBmodePixel-secondStepPara.halfWinLenPoint);
nDpPoint=length(iniDp.yIndex);
iniDp.x=NaN(nDpPoint, nBeam);
iniDp.y=NaN(nDpPoint, nBeam);

x=1:nBeam;
[iniX, iniY]=meshgrid(x, iniDp.yIndex);
[X, Y]=meshgrid(x, Dp.yIndex);

iniDp.x = interp2(X, Y, Dp.x, iniX, iniY);
iniDp.y = interp2(X, Y, Dp.y, iniX, iniY);

minY=min(Dp.yIndex);
iii=max(find(iniDp.yIndex<minY));
iniDp.x(1:iii, :)=repmat(iniDp.x(iii+1, :), iii, 1);
iniDp.y(1:iii, :)=repmat(iniDp.y(iii+1, :), iii, 1);
maxY=max(Dp.yIndex);
iii=min(find(iniDp.yIndex>maxY));
iniDp.x(iii:nDpPoint, :)=repmat(iniDp.x(iii-1, :), nDpPoint-iii+1, 1);
iniDp.y(iii:nDpPoint, :)=repmat(iniDp.y(iii-1, :), nDpPoint-iii+1, 1);

iniDp.x=round(iniDp.x);
iniDp.y=round(iniDp.y);
