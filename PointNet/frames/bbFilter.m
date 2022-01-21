function filter_dist = bbFilter(distance,filtertype)
disp('Filtering ...')
tic
% for 3x3 median filter
m=3;
n=3;

dim = size(distance);

if (dim(1) == 1 && dim(2) == 1)
   dim(3)=1;
    end
    for i=1:dim(1)
        for j=1:dim(2)
            for k=1:dim(3)
                dist = distance{i,j,k};
                if strcmp(filtertype, 'median')
                    filter_dist{i,j,k} = medfilt2(dist, [m n],'symmetric');
                end
            end
        end
    end

disp('Filtering done!')
toc
disp('=====================================')
end

% To Be Evaluated (TBE): smooth3