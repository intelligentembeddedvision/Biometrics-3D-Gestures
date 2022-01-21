function [D, new_hand, pointCloud] = bbSegmentation(filter_dist)
%Segment the hand

disp('Segmenting ...')
tic

MinArea = 80;
dim = size(filter_dist);



if (dim(1) == 1 && dim(2) == 1)
    dim(3)=1;
end

    hand = cell(dim(1),dim(2), dim(3));
    %hand{1,j,k}=array of distances
    D = cell(dim(1),dim(2), dim(3));
    %pixels coordinates and the distace d for each pixel from hand with d>0
    for i=1:dim(1)
        for j=1:dim(2)
            for k=1:dim(3)
                %fprintf('%d %d %d', i,j,k);
                %fprintf('\r');
                % 2 5 8 ?
                f_dist = filter_dist{i,j,k};
                [handarray, flag] = bbClosestPoint(f_dist);
                if flag > 0
                    fprintf('%d spikes found at %d %d %d %d',flag, i,j,k);
                    fprintf('\r');
                end
                
                pointCloud = makePointCloud(handarray,0.3, 2048);
                hand{i,j,k} = bbRectangle(handarray);
                new_hand = hand;
                %%%%%%%%GEORGI SEGMENTARE%%%%%%%%%%%%%%%%
                dim_mana=size(hand{i,j,k});
%                 for x=1:dim_mana(1,1)
%                     for y=1:dim_mana(1,2)
%                         if(hand{i,j,k}(x,y)>0)
%                             mana{i,j,k}(x,y)=1;
%                         else
%                             mana{i,j,k}(x,y)=0;
%                             
%                         end
%                     end
%                 end
%                 
%                 ImB= im2bw(mana{i,j,k}, 0.9);
%                 %imshow(ImB);
%                 cc = bwconncomp(ImB);
%                 stats = regionprops(cc, 'Area','PixelList','Orientation','centroid');
%                 idx = find([stats.Area] > MinArea);
%                 BW3 = ismember(labelmatrix(cc), idx);
%                 centroids = cat(1, stats.Centroid);
%                 
%                 if size(idx,2)>1
%                     % multiple objectd detected
%                     for m=1:size(idx,2)%nr de regiuni cu mai mult de 80pixels
%                         if (centroids(idx(1,m),2)<((dim_mana(1,1))/2))%verific daca coordonatele centroidului regiunii sunt in jumatatea superioara a imaginii
%                             new_D=stats(idx(1,m)).PixelList;%extrag coordonatele pixelilor din regiune
%                             maxXY=max(new_D);%max pe x respectiv y
%                             minXY=min(new_D);%min pe x respectiv y
%                             %decupez regiunea de mana
%                             oo=1;
%                             for o=minXY(1,1):maxXY(1,1)
%                                 pp=1;
%                                 for p=minXY(1,2):maxXY(1,2)
%                                     new_hand{i,j,k}(pp,oo)=hand{i,j,k}(p,o);%%%%trebuie vazut cum rescriu
%                                     pp=pp+1;
%                                 end
%                                 oo=oo+1;
%                             end
%                               D{i,j,k} = bbTakeOutZero(new_hand{i,j,k});
%                         else
%                             new_D=stats(idx(1,m)).PixelList;
%                         end
%                     end
%                 else
%                     % just one object present in scene
%                     new_hand{i,j,k}(:,:)=hand{i,j,k}(:,:);
%                     D{i,j,k} = bbTakeOutZero(new_hand{i,j,k});
%                 end
%                 %%%%%%%%end%%%%%%
%                % hand=new_hand;
%                
  
               
            end
        
        end
        
         %D{i,j,k} = bbTakeOutZero(new_hand{i,j,k});
    end
disp('Segmentation done!')
toc
disp('=====================================')

end



    function [hand, flag] = bbClosestPoint(filter_dist)
        % Assume that the closest point belongs to hand, get data within 7cm
        dim = size(filter_dist);
%         HAND_DEPTH = 0.07;
        HAND_DEPTH_MIN = 0.03;
        HAND_DEPTH_MAX = 0.07;
        HAND_DEPTH = HAND_DEPTH_MIN;
        MIN_POINTS = 1000;
        flag = 0;
        [sortdist, index] = sort(filter_dist(:));
        y = ceil(index(1)/dim(1));
        x = index(1)-(y-1)*dim(2);
        centroid = [x y];
        for k = 1:length(sortdist)
            adaptDepth = 1;
            while adaptDepth
                for i=1:dim(1)
                    for j=1:dim(2)
                        if filter_dist(i,j) > sortdist(k) && filter_dist(i,j)< sortdist(k)+HAND_DEPTH
                            hand(i,j)=filter_dist(i,j)-sortdist(k);
                        else
                            hand(i,j)=0;
                        end
                    end
                end
                if nnz(hand) > MIN_POINTS
                    return
                end
                if HAND_DEPTH < HAND_DEPTH_MAX
                    HAND_DEPTH = HAND_DEPTH + (HAND_DEPTH_MAX-HAND_DEPTH_MIN)/10;
                else
                    hand = 0;
                    flag = flag + 1;
                    adaptDepth = 0;
                end
            end
        end

    end

% function hand = bbClosestPoint(filter_dist)
% % Assume that the closest point belongs to hand, get data within 7cm
% dim = size(filter_dist);
% HAND_DEPTH=0.07;
% for i=1:dim(1)
%     fl=0;
%     for j=1:dim(2)
%         if filter_dist(i,j) > min( min(filter_dist)) && filter_dist(i,j)< min( min(filter_dist)+HAND_DEPTH)
%             hand(i,j)=filter_dist(i,j)-min( min(filter_dist));
%         else
%             hand(i,j)=0;
%         end
%     end
% end
% end

    function hand = bbRectangle(hand)
        % % takeout zero elements and get the rectangle surrounding the hanf
        hand( :, all(~hand,1) ) = [];
        hand( all(~hand,2), : ) = [];
    end

    function D = bbTakeOutZero(hand)
        % get just the hand eith no zero elements
        size_hand=size(hand);
        z=0;
        for i=1:size_hand(1)
            for j=1:size_hand(2)
                if hand(i,j)>0
                    z=z+1;
                    D(z,:)=[i,j,hand(i,j)];
                end
            end
        end
    end

