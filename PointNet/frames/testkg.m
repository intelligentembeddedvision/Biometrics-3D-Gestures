clear e;
clc
for Kg=1:60
[Error, ConfMatrix] = bbClassify(features, dist_func, Kg);
e(Kg) = Error;
end
plot(e)
e