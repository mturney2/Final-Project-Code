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
    y0=subSampleFit(SSD(:,2)');
end