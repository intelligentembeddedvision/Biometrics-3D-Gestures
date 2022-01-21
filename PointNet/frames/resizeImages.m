close all
clear all
clc

load dataset

datasetResized=dataset;

for i=1:size(dataset,1)
    sizes(i,:)=size(dataset{i,3});
    datasetResized{i,1} = dataset{i,1};
    datasetResized{i,2} = dataset{i,2};
end

targetSize = [mean(sizes(:,1)) mean(sizes(:,2))];
targetSize = round(targetSize);

for i=1:size(dataset,1)
     if i==1084
        x =1; 
     end
     img = dataset{i,3};
     imgSize = size(img);
     scale = targetSize./imgSize;
     minScale = min(scale);
     if find(scale==minScale) == 1
        newSize = [targetSize(1) round(imgSize(2)*scale(1))]; 
     else
        newSize = [round(imgSize(1)*scale(2)) targetSize(2)];
     end
     img_res = imresize(img,minScale);
     img_res = imresize(img, newSize);
     if targetSize(1)==size(img_res,1)
        sizeDif = floor((targetSize(2)-size(img_res,2))/2);
        tempImg = zeros(targetSize(1),sizeDif);
        img_res = [tempImg img_res tempImg];
        if targetSize(2)~=size(img_res,2)
            img_res(:,end+1) = 0;
        end
     end
     if targetSize(2)==size(img_res,2)
        sizeDif = floor((targetSize(1)-size(img_res,1))/2);
        tempImg = zeros(sizeDif,targetSize(2));
        img_res = [tempImg;img_res;tempImg];
        if targetSize(1)~=size(img_res,1)
            img_res(end+1,:) = 0;
        end
     end     
     figure(1), imshow(img,[]);
     figure(2), imshow(img_res,[]);
     datasetResized{i,3} = img_res;
     input('');
end
imageSize = targetSize;
save('datasetResized','datasetResized','imageSize');
