function        Dp1=Filter2D_Marker(Dp, Region, Marker);

[M, N]=size(Dp.x);
Dp1=Dp;
Marker=Marker(Dp.yIndex, :);

[mlist, nlist]=find(Marker>0); NN=length(mlist);
mlow=max([mlist-Region(1,1), ones(NN,1)], [], 2);
mhigh=min([mlist+Region(1,1), M*ones(NN,1)], [], 2);
nlow=max([nlist-Region(1,2), ones(NN,1)], [], 2);
nhigh=min([nlist+Region(1,2), N*ones(NN,1)], [], 2);
for k=1:NN
    tmpDx=Dp.x(mlow(k):mhigh(k), nlow(k):nhigh(k));
    III=(tmpDx(:)<realmax);
    Dp1.x(mlist(k), nlist(k))=median(tmpDx(III));
end

[mlist, nlist]=find(Marker>0); NN=length(mlist);
mlow=max([mlist-Region(2,1), ones(NN,1)], [], 2);
mhigh=min([mlist+Region(2,1), M*ones(NN,1)], [], 2);
nlow=max([nlist-Region(2,2), ones(NN,1)], [], 2);
nhigh=min([nlist+Region(2,2), N*ones(NN,1)], [], 2);
for k=1:NN
    tmpDy=Dp.y(mlow(k):mhigh(k), nlow(k):nhigh(k));
    III=(tmpDy(:)<realmax);
    Dp1.y(mlist(k), nlist(k))=median(tmpDy(III));
end