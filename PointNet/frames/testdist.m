clear e;
clc
for i=1:length(dist_func)
[Error, ConfMatrix] = bbClassify(features, dist_func{i}, Kg);
e(i) = Error;
end
e
[mine indexe]=min(e)
figure
barh(e)
set(gca,'YTickLabel',dist_func)
