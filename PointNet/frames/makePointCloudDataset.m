clear all
clc
close all

load dataset_better

for i = 1:size(dataset,1)
  pointCloudSet{i,1}=makePointCloud(dataset{i,1},0.2,1024);  
  pointCloudSet{i,2} = dataset{i,2};
  i
  input('');
end
save('pointCloudDataset','pointCloudSet');