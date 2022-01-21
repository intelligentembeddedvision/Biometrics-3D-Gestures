clear all
close all
clc

load pointCloudDataset
for i = 1:size(pointCloudSet,1)
    figure(1);
    scatter3(pointCloudSet{i,1}(1,:), pointCloudSet{i,1}(2,:), pointCloudSet{i,1}(3,:));
    axis([-1 1 -1 1 -1 1]);
    title(['Gesture: ' num2str(pointCloudSet{i,2})]);
    input('');
end