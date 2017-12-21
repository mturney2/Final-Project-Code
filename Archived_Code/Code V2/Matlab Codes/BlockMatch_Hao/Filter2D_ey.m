function Dp1=Filter2D_2Block(Dp, Region);

[M, N]=size(Dp);
Dp1=Dp;
N2=N;

tt=Dp(:, 1:N2);
[mlist, nlist]=find(tt<realmax); NN=length(mlist);
mlow=max([mlist-Region(1,1), ones(NN,1)], [], 2);
mhigh=min([mlist+Region(1,1), M*ones(NN,1)], [], 2);
nlow=max([nlist-Region(1,2), ones(NN,1)], [], 2);
nhigh=min([nlist+Region(1,2), N2*ones(NN,1)], [], 2);
for k=1:NN
    tmpDx=tt(mlow(k):mhigh(k), nlow(k):nhigh(k));
    III=(tmpDx(:)<realmax);
    tt(mlist(k), nlist(k))=median(tmpDx(III));
end
Dp1(:, 1:N2)=tt;

