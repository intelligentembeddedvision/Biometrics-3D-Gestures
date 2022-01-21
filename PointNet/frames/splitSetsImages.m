clear all
close all
clc


load datasetResized;

setSize = size(datasetResized,1);
nrGestures = zeros(1,6);
for i = 1:setSize(1)
   nrGestures(datasetResized{i,2}) = nrGestures(datasetResized{i,2})+1;  
end
minNr = min(nrGestures);

split = 0.9;
trainsetSize = round(minNr*split);
testsetSize = minNr-trainsetSize;

indexes = 1:setSize(1);
count = 1;
for i=1:trainsetSize
    for j=1:6
        while 1
            k =ceil(rand()*size(indexes,2));
            index = indexes(k);
            if datasetResized{index,2}==j
                break;
            end
        end
        trainSet(count,:,:) =  datasetResized{index,3};
        trainLabels(count,:) = ones(1,6);
        trainLabels(count,j) = 2;
        indexes(k) = [];
        count = count+1;    
    end
end

count = 1;
for i=1:testsetSize
    for j=1:6
        while 1
            k =ceil(rand()*size(indexes,2));
            index = indexes(k);
            if datasetResized{index,2}==j
                break;
            end
        end
        testSet(count,:,:) =  datasetResized{index,3};
        testLabels(count) = j;
        indexes(k) = [];
        count = count+1;    
    end
end


trainSet = permute(trainSet,[3  2 1]);
testSet = permute(testSet, [3 2 1]);
trainLabels = trainLabels';
% testLabels = testLabels';
name_train = ['X_gestures_' num2str(split*100) '-' num2str(100-split*100) '.h5'];
name_test = ['X_gestures_' num2str(split*100) '-' num2str(100-split*100) '_test.h5'];
h5create(name_train,'/imgs',size(trainSet),'Datatype','single');
h5create(name_train,'/labels',size(trainLabels),'Datatype','uint8');
h5write(name_train,'/imgs',single(trainSet));
h5write(name_train,'/labels',uint8(trainLabels));

h5create(name_test,'/imgs',size(testSet),'Datatype','single');
h5create(name_test,'/labels',size(testLabels),'Datatype','uint8');
h5write(name_test,'/imgs',single(testSet));
h5write(name_test,'/labels',uint8(testLabels));