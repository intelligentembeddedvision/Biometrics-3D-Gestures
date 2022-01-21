close all
clear all
clc

data = h5read('ply_data_train0.h5','/data');
labels = h5read('ply_data_train0.h5','/label');
object = data(:,:,1);

pobject = data(:,1,:);
pobject = permute(pobject, [1 3 2]);

scatter3(object(1,:), object(2,:), object(3,:),'*');
labels(1)
figure,
plot3(pobject(1,:), pobject(2,:), pobject(3,:),'*');

