function features = bbFeatures( D, FBtri,FBpoints,featuretype, verbose )
% Feature extraction
% featuretype = 'HistOfNormals2Points', 'HistOfNormals2Surfs'; 'Curvatures'

% Two of the most widely used geometric point features are:
% the underlying surface's estimated curvature and
% normal at a query point p.
% Both of them are considered local features, as they characterize a point using the informationprovided by its k closest point neighbors.

% Histogram of oriented gradients (HOG) using surface normals
% TBE: curvature based histograms, spin image signatures, or surflet-pair-relation histograms

% TBD: classify:
% cylinder
% ellipsoidsphere
% sphere

disp('Feature extraction ...')
tic

dim = size(D);
if (dim(1) == 1 && dim(2) == 1)
    dim(3)=1;
end
    %'HistOfNormals2Surfs' not implemented yet

    for i=1:dim(1)
        for j=1:dim(2)
            for k=1:dim(3)
                if (~isempty(D{i,j,k}))
                Darray = D{i,j,k};
                [nx{i,j,k}, ny{i,j,k}, nz{i,j,k}] = bbPointsNorm(Darray, verbose);
                I = bbHist(nx{i,j,k}, ny{i,j,k}, nz{i,j,k}, verbose);
                
                % scaled betwwen 0 and 1
                features{i,j,k} = (I-min(I(:))) ./ (max(I(:)-min(I(:))));
                
                %no scale
                %features{i,j,k} = I;
                end
            end
        end
    end

disp('Feature extraction done!')
toc
disp('=====================================')

%% Normals to points
    function [nx, ny, nz] = bbPointsNorm(D, verbose)
        [nx, ny, nz] = surfnorm(D);
        if verbose >=1
            figure
            surfnorm(D);
        end
    end

%% Normals to surfaces
    function fn = bbSurfsNorm(FBtri, FBpoints, verbose)
        tr = TriRep(FBtri, FBpoints);
        P = incenters(tr);
        fn = faceNormals(tr);
        if verbose >=1
            trisurf(FBtri,FBpoints(:,1),FBpoints(:,2),FBpoints(:,3),'FaceColor', 'cyan', 'faceAlpha', 0.8);
            figure
            axis equal;
            hold on;
            quiver3(P(:,1),P(:,2),P(:,3),fn(:,1),fn(:,2),fn(:,3),0.5, 'color','r');
            % TBE: isonormals isosurface
            hold off;
        end
    end
%% Histogram
    function [N,C] = bbHist(nx,ny,nz, verbose)
        % Coordinates conversion
        [TH,PHI,R] = cart2sph(nx,ny,nz);
        X = [TH(:) PHI(:)];
        bin1 =57;%9
        bin2 = 1;%7
        %Get back counts, but don't make the plot.
        [N,C] = hist3(X, [bin1 bin2]);
        
        %eliminate pi/2 which come mostly from the background
        % X not modified !!! so histo is not good
        N(round(bin1/2),bin2) = 0;
        
        
        if verbose >=1
            % Make a histogram with bars colored according to height
            figure
            hist3(X,[bin1 bin2],'FaceAlpha',.65);
            xlabel('Azimuth'); ylabel('Elevation');
            set(gcf,'renderer','opengl');
            set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
        end
    end
end

