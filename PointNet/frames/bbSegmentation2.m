function [D, new_hand] = bbSegmentation2(filter_dist)
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
                hand{i,j,k} = bbRectangle(handarray);
                
                %%%%%%%%GEORGI SEGMENTARE%%%%%%%%%%%%%%%%
                dim_mana=size(hand{i,j,k});
                for x=1:dim_mana(1,1)
                    for y=1:dim_mana(1,2)
                        if(hand{i,j,k}(x,y)>0)
                            mana{i,j,k}(x,y)=1;
                        else
                            mana{i,j,k}(x,y)=0;
                            
                        end
                    end
                end
                
                ImB= im2bw(mana{i,j,k}, 0.9);
                %imshow(ImB);
                cc = bwconncomp(ImB);
                stats = regionprops(cc, 'Area','PixelList','Orientation','centroid');
                idx = find([stats.Area] > MinArea);
                BW3 = ismember(labelmatrix(cc), idx);
                centroids = cat(1, stats.Centroid);
                
                if size(idx,2)>1
                    % multiple objectd detected
                    for m=1:size(idx,2)%nr de regiuni cu mai mult de 80pixels
                        if (centroids(idx(1,m),2)<((dim_mana(1,1))/2))%verific daca coordonatele centroidului regiunii sunt in jumatatea superioara a imaginii
                            new_D=stats(idx(1,m)).PixelList;%extrag coordonatele pixelilor din regiune
                            maxXY=max(new_D);%max pe x respectiv y
                            minXY=min(new_D);%min pe x respectiv y
                            %decupez regiunea de mana
                            oo=1;
                            for o=minXY(1,1):maxXY(1,1)
                                pp=1;
                                for p=minXY(1,2):maxXY(1,2)
                                    new_hand{i,j,k}(pp,oo)=hand{i,j,k}(p,o);%%%%trebuie vazut cum rescriu
                                    pp=pp+1;
                                end
                                oo=oo+1;
                            end
                              D{i,j,k} = bbTakeOutZero(new_hand{i,j,k});
                        else
                            new_D=stats(idx(1,m)).PixelList;
                        end
                    end
                else
                    % just one object present in scene
                    new_hand{i,j,k}(:,:)=hand{i,j,k}(:,:);
                    D{i,j,k} = bbTakeOutZero(new_hand{i,j,k});
                end
                %%%%%%%%end%%%%%%
               % hand=new_hand;
               
  
               
            end
        
        end
        
         %D{i,j,k} = bbTakeOutZero(new_hand{i,j,k});
    end
    
    new_hand = removeForearm(new_hand{1});
    
disp('Segmentation done!')
toc
disp('=====================================')

end

function hand = removeForearm(hand)

    data = bbTakeOutZero(hand);
    data = data(:,1:2);
      
    [x y] = pca(data);
     
    absoluteOffset = data-(x*y')';    
    absoluteOffset = absoluteOffset(1,:);  
    
    
    [a i] = sort(y(:,1))
    y= [a y(i,2)];
    
    ratio =max(abs(y(:,1)))/max(abs(y(:,2)));
    
%     if ratio < 1.5
%         return
%     end
    
    foreArm = zeros(size(hand));
    
    value = max(max(hand))*1.3;
    hand(round(absoluteOffset(1,1)),round(absoluteOffset(1,2))) = value;
    hand(round(absoluteOffset(1,1)+10*x(1,1)),round(absoluteOffset(1,2)+10*x(2,1))) =value;
    hand(round(absoluteOffset(1,1)+20*x(1,1)),round(absoluteOffset(1,2)+20*x(2,1))) =value;
    hand(round(absoluteOffset(1,1)+10*x(1,2)),round(absoluteOffset(1,2)+10*x(2,2))) =value;
    figure, imshow(hand,[]);
    title(num2str(ratio));

    track = [];
    trackIndex = [];
    ftrack = [];
    initialObj1Size = 10;
    forearmWindow = 200;
    
    objContour = [y(1,:) 1];

    for j = 1:size(y,1)-1
            if objContour(end,2) >= 0
               if y(j,2) > objContour(end, 2)
                  objContour(end,:) = [y(j,:) j]; 
               end
               if y(j,2) < 0
                  objContour = [objContour;[y(j,:) j]];
               end
            end
            if objContour(end,2) < 0
               if y(j,2) < objContour(end, 2)
                  objContour(end,:) = [y(j,:) j]; 
               end
               if y(j,2) >= 0
                  objContour = [objContour;[y(j,:) j]];
               end
            end                   
    end  
    dataSize = size(objContour,1);
    
    for i=initialObj1Size:dataSize-(forearmWindow+1)
        obj1 = y(1:i,:);
        obj2 = y(i+1:i+forearmWindow+1,:);

        obj1Contour = [obj1(1,:) 1];
        for j = 1:size(obj1,1)-1
            if obj1Contour(end,2) >= 0
               if obj1(j,2) > obj1Contour(end, 2)
                  obj1Contour(end,:) = [obj1(j,:) j]; 
               end
               if obj1(j,2) < 0
                  obj1Contour = [obj1Contour;[obj1(j,:) j]];
               end
            end
            if obj1Contour(end,2) < 0
               if obj1(j,2) < obj1Contour(end, 2)
                  obj1Contour(end,:) = [obj1(j,:) j]; 
               end
               if obj1(j,2) >= 0
                  obj1Contour = [obj1Contour;[obj1(j,:) j]];
               end
            end            
            
        end
        obj2Contour = [obj2(1,:) 1];
        for j = 1:size(obj2,1)-1
            if obj2Contour(end,2) >= 0
               if obj2(j,2) > obj2Contour(end, 2)
                  obj2Contour(end,:) = [obj2(j,:) j]; 
               end
               if obj2(j,2) < 0
                  obj2Contour = [obj2Contour;[obj2(j,:) j]];
               end
            end
            if obj2Contour(end,2) < 0
               if obj2(j,2) < obj2Contour(end, 2)
                  obj2Contour(end,:) = [obj2(j,:) j]; 
               end
               if obj2(j,2) >= 0
                  obj2Contour = [obj2Contour;[obj2(j,:) j]];
               end
            end            
            
        end       
        
        trackIndex = [trackIndex;i];
        track = [track;mean(abs(obj1Contour(:,2))) max(abs(obj2Contour(:,2))) mean(abs(obj2Contour(:,2)))];
        fsize = 25;
%         if size(track,1)==1
%             ftrack = track;
%         else
%             if size(track,1)<fsize
%                 ftrack = [ftrack;min(track)];
%             else
%                 ftrack = [ftrack;min(track(end-fsize+1:end,:))];
%             end
%         end
        
%         
%     relativeOffset = round(y(i,1));  
%     offset = absoluteOffset+relativeOffset*x(:,1)';
%     c1 = x(1,1);
%     c2 = x(2,1);
%     c3 = -c1*offset(1,1)-c2*offset(1,2);
%     if i==800
%     clc
%     end     
%     
%     temphand = hand;
%     for k=1:size(hand,1)
%         for l=1:size(hand,2)
%             if ([k l]*x(:,1)+c3)>0
%                 temphand(k,l) = 0;
%             end
%         end
%     end
%       if i==800  
%         figure, imshow(temphand,[]); 
%       end   
    end
    
        
%     ftrack = ftrack(:,1).*ftrack(:,2);
%     figure , plot(ftrack);
%     dtrack = abs(ftrack(35+1:end)-ftrack(1:end-35));
%     meanD = dtrack(1);
%     for i = 2:size(dtrack)
%         if dtrack(i)<0.03*meanD
%             break;
%         end
%         meanD = mean(dtrack(1:i));
%     end

    figure, plot(trackIndex,track(:,1));
    figure, plot(trackIndex,track(:,2));   
    figure, plot(trackIndex,track(:,3));
    segIndex = 100;
    relativeOffset = round(y(segIndex,1));  
    offset = absoluteOffset+relativeOffset*x(:,1)';
    c1 = x(1,1);
    c2 = x(2,1);
    c3 = -c1*offset(1,1)-c2*offset(1,2);
    for i=1:size(hand,1)
        for j=1:size(hand,2)
            if ([i j]*x(:,1)+c3)>0
                foreArm(i,j) = hand(i,j);
                hand(i,j) = 0;
            end
        end
    end
  

    ratio1 =max(abs(y(:,1)))/max(abs(y(1:segIndex,2)));
    ratio2 =max(abs(y(:,1)))/max(abs(y(segIndex+1:segIndex+forearmWindow+1,2)));
    
    figure, imshow(hand,[]);
    title([num2str(ratio) ' ' num2str(ratio1)]);
    figure, imshow(foreArm,[]);   
    title([num2str(ratio) ' ' num2str(ratio2)]);    
    close all
end

    function [hand, flag] = bbClosestPoint(filter_dist)
        % Assume that the closest point belongs to hand, get data within 7cm
        dim = size(filter_dist);
        HAND_DEPTH = 0.07;
        MIN_POINTS = 200;
        flag = 0;
        [sortdist, index] = sort(filter_dist(:));
        for k = 1:length(sortdist)
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
            else
                hand = 0;
                flag = flag + 1;
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

