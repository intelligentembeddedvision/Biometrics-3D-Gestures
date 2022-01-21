%% Init
close all
clear all
clc
%% Set parameters
% See bbConfig script for all parameters
global dataSetCount;
dataSetCount = 1;
dataSetCount2 = 1;
for frame = 1:20
for person =1:11
for gesture = 1:6
%     pentru debug. breakpoint pe un anume set de date
% if (frame==20)&&(person==6)&&(gesture==6)
%     break
% end
if(person==9)
   x =1; 
end
if (frame==1)&&(person==9)&&(gesture==5)
    x = 1;
end
[frame person gesture]
bbConfig
%% Get the distances
[dist img]= bbGetDistances(datatype, file, frameindex);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               


% 2D+3D plot of original data 

bb2D3Dplot(dist,verbose(1))
%% Distance Filtering
filter_dist = bbFilter(dist,filtertype);
%2D+3D plot of filtered data
bb2D3Dplot(filter_dist,verbose(2));
%% Segmemtation
dataSetCount
dataSetCount2

[D, hand, noHandFlag,img] =  bbSegmentation(filter_dist,img{1,1,1});



if ~noHandFlag
    h = figure(1);
%     set(h,'OuterPosition',[900 500 400 400]);   
    h = figure(2);
%     set(h,'OuterPosition',[900 3000 400 400]);  
    figure(1), imshow(hand,[]);
    figure(2), imshow(img,[]);
    key  = input('');
%     key = [];
    if isempty(key)
        dataset{dataSetCount,1} = hand;
        dataset{dataSetCount,2} = gesture;
        dataset{dataSetCount,3} = img/max(max(img));
        dataSetCount = dataSetCount + 1;
    end
end

if rem(dataSetCount,30)==0
    save('dataset','dataset');
end

dataSetCount2 = dataSetCount2 + 1;
    
% [D, hand, pointCloud(:,:,dataSetCount)] =  bbSegmentation(filter_dist);
% 2D+3D plot of segmented data
% bb2D3Dplot(hand,verbose(3));
% %% Delaunay
% [dt3,dt2,FBtri,FBpoints] = bbDelaunay(D,verbose(4));
% %% Feature extraction
% features = bbFeatures(hand, FBtri,FBpoints,featuretype, verbose(2));
% %% Classification
% % [Error, ConfMatrix] = bbClassify(features, sel_dist_func, Kg);

end
end
end
save('dataset','dataset');