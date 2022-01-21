function [D, hand] = bbSegmentation(filter_dist)
%Segment the hand

disp('Segmenting ...')
tic

dim = size(filter_dist);



if (dim(1) == 1 && dim(2) == 1)
    f_dist = filter_dist{1,1,1};
    handarray = bbClosestPoint(f_dist);
    hand{1,1,1} = bbRectangle(handarray);
    D{1,1,1} = bbTakeOutZero(hand{1,1,1});
else
    hand = cell(dim(1),dim(2), dim(3));
    D = cell(dim(1),dim(2), dim(3));
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
                D{i,j,k} = bbTakeOutZero(hand{i,j,k});
            end
        end
    end
end
disp('Segmentation done!')
toc
disp('=====================================')

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

