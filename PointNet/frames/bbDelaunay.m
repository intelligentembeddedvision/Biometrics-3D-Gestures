function [dt3,dt2,FBtri,FBpoints] = bbDelaunay(D,verbose)
% 2D 3D Delaunay and free boundary

disp('Delaunay ...')
tic

dim = size(D);
if (dim(1) == 1 && dim(2) == 1)
    dim(3)=1;
end
    for i=1:dim(1)
        for j=1:dim(2)
            for k=1:dim(3)
                if (~isempty(D{i,j,k}))
                    Darray = D{i,j,k};
                    dt2{i,j,k} = bb2DDelaunay(Darray,verbose);
                    dt3{i,j,k} = bb3DDelaunay(Darray,verbose);
                    [FBtri{i,j,k},FBpoints{i,j,k}]=bbfreeBoundary(dt3{i,j,k},verbose);
                end
            end
        end
    end

disp('Delaunay done!')
toc
disp('=====================================')


%% 2D
    function dt2 = bb2DDelaunay(D,verbose)
        
        dt2 = delaunayTriangulation(D(:,1),D(:,2));
        
        if verbose >=1
            
            figure
            triplot(dt2);
            title ('2D Delaunay')
            % TBE: Calculate freeBoundry/convex hull
        end
    end
%% 3D
    function dt3 = bb3DDelaunay(D,verbose)
        
        dt3 = delaunayTriangulation(D(:,1),D(:,2),D(:,3));
        
        if verbose >=1
            faceColor  = [0.6875 0.8750 0.8984];
            figure
            tetramesh(dt3,'FaceColor',faceColor,'FaceAlpha',0.3);
            title ('3D Delaunay')
        end
    end
%% freeBoundary
    function [FBtri,FBpoints]=bbfreeBoundary(dt3,verbose)
        % TBE: convex hull
        % Seems to be the same, see example 2 in help freeBoundary
        
        [FBtri,FBpoints] = freeBoundary(dt3);
        
        if verbose >=1
            figure
            trisurf(FBtri,FBpoints(:,1),FBpoints(:,2), FBpoints(:,3),'FaceColor','cyan','FaceAlpha', 0.8);
            title ('freeBoundary')
        end
    end
end