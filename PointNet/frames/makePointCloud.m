function pointCloud = makePointCloud(handDist,hwRatio,nrPoints)
disp('Creating hand pointcloud...');
tic

figure(1), imshow(handDist,[]);
dataPos = handDist>0;
coord = [];
count = 0;
pointsToDisplay=zeros(size(dataPos));
while sum(sum(dataPos)) > nrPoints
    addRemove = 'remove';
    coord = getCandidateCoord(dataPos, addRemove);
    pointsToDisplay(coord(1),coord(2)) = 1;
    handDist(coord(1),coord(2))=0;
    dataPos = handDist>0;
    count = count + 1;
end



while (sum(sum(dataPos)) + size(coord,1)) < nrPoints
    addRemove = 'add';
    coord = [coord;getCandidateCoord(dataPos, addRemove)];
    pointsToDisplay(coord(end,1),coord(end,2)) = 1;
    count = count + 1;
end
figure(2), imshow(pointsToDisplay,[]);
text = [num2str(count) ' points to ' addRemove '...'];

% figure(2), imshow(dataPos,[]);
% title(text);
[x y] = find(handDist);



xShifted = x-(max(x)+min(x))/2;
yShifted = y-(max(y)+min(y))/2;
xyScale = max(max(xShifted),max(yShifted));

xScaled = xShifted./xyScale;
yScaled = yShifted./xyScale;


for i = 1:size(x,1)
    pointCloud(:,i) = [xScaled(i) yScaled(i) handDist(x(i),y(i))]';
end

if strcmp(addRemove, 'add')
    originalSize = size(pointCloud,2);
    for i = 1:size(coord,1)
        j = x==coord(i,1);
        k = y==coord(i,2);
        index = find((j+k)==2);
        pointCloud(:, originalSize+i) = [xScaled(index) yScaled(index) handDist(x(index),y(index))]';
    end
end


zDist = pointCloud(3,:);
zShifted = zDist - (max(zDist)+min(zDist))/2;
zScale = max(max(zShifted))/hwRatio;
zScaled = zShifted./zScale;

pointCloud(3,:) = zScaled;
figure(3), scatter3(pointCloud(1,:), pointCloud(2,:), pointCloud(3,:));
axis([-1 1 -1 1 -1 1]);



disp('Pointcloud created');
toc