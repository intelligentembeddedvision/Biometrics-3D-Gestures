clear all
close all
clc

split = 0.9;

train_name = ['ply_data_train_' num2str(split*100) '-' num2str(100-split*100)] ;
test_name = ['ply_data_test_' num2str(split*100) '-' num2str(100-split*100)];


load pointCloudDataset1024

cloudSize = size(pointCloudSet,1);
nrGestures = zeros(1,6);
for i = 1:size(pointCloudSet,1)
   nrGestures(pointCloudSet{i,2}) = nrGestures(pointCloudSet{i,2})+1;  
end
minNr = min(nrGestures);

batchSize = 32;
nrAll = floor(batchSize/6)
nrBatches = floor(minNr*6/batchSize);
indexesSource = 1:size(pointCloudSet,1);


for i = 1:nrBatches
    instances = zeros(1,6);
    j=1;
    flag = 0;
    fill = 0;
    while 1
        a = ceil(size(indexesSource,2)*rand());
        index = indexesSource(a);
        
        if (instances(pointCloudSet{index,2})==fill)
            batch{i}{j,1} = pointCloudSet{index,1};
            batch{i}{j,2} = pointCloudSet{index,2}-1;
            indexesSource(a) = [];
            j = j+1;
            instances(pointCloudSet{index,2}) = instances(pointCloudSet{index,2})+1;
            
        end
        if sum(instances==(fill+1)) == 6
            fill = fill+1;
        end
        

        if sum(instances)==batchSize
            break
        end
    end

end

trainSize = round(nrBatches*split);

trainCount =  1;
testCount = 1;
for i = 1:nrBatches
   if i <= trainSize 
       for j = 1:batchSize
            trainData(trainCount,:,:) = batch{i}{j,1}';
            trainLabels(trainCount,:,:) = batch{i}{j,2};
            
%             figure(1), scatter3(trainData(trainCount,:,1), trainData(trainCount,:,2), trainData(trainCount,:,3),'o');
%             title(num2str(trainLabels(trainCount)));
%             axis([-1 1 -1 1 -1 1]);
%             input('');      
            trainCount = trainCount + 1;
       end
   else
       for j = 1:batchSize
            testData(testCount,:,:) = batch{i}{j,1}';
            testLabels(testCount,:,:) = batch{i}{j,2};
            
%             figure(1), scatter3(testData(testCount,:,1), testData(testCount,:,2), testData(testCount,:,3),'o');
%             title(num2str(testLabels(testCount)));
%             axis([-1 1 -1 1 -1 1]);
%             input('');                 
            
            testCount = testCount +1;
       end       
   end
end


trainData = permute(trainData,[3  2 1]);
testData = permute(testData, [3 2 1]);
trainLabels = trainLabels';
testLabels = testLabels';
h5create([train_name '.h5'],'/data',size(trainData),'Datatype','single');
h5create([train_name '.h5'],'/label',size(trainLabels),'Datatype','uint8');
h5write([train_name '.h5'],'/data',single(trainData));
h5write([train_name '.h5'],'/label',uint8(trainLabels));

h5create([test_name '.h5'],'/data',size(testData),'Datatype','single');
h5create([test_name '.h5'],'/label',size(testLabels),'Datatype','uint8');
h5write([test_name '.h5'],'/data',single(testData));
h5write([test_name '.h5'],'/label',uint8(testLabels));