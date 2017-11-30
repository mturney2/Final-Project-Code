function delta=subSampleFit(vector)
delta=0;
if vector(1) ~= vector(3)
    temp= (vector(1)-vector(3))/(2*(vector(1)+vector(3)-2*vector(2)));
    if abs(temp)<1
        delta=temp;
    end
end