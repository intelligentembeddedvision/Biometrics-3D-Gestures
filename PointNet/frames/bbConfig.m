disp('Configuring ...')
tic
%% Set paths
%addpath 'D:\PMD\PMDMDK\MDK_mexw32'
%%%%%%%%%%%%%%%%%%%%%%%Schimbam calea%%%%%%%%%%%%%%%%%%%%%%%
%%% Verbose
% Verbose level 0 = minimum info display, level 3 = maximum info display
%[Distances Filtered Segmented Delaunay Fetures]

verbose  = [3 3 3 3 3];
%% Set the data source

% Single PMD file
% datatype = 'PMD';
% file = 'E:\MY\My Research\Gesture\Databases\UPT-ToF3D-HGDB\radu.pmd';
% frameindex = 100;

% Single binary file
% gesture = 6, framepergesture = 20, offest = 1...20
% frameindex = (gesture-1)*framepergesture + offset
datatype = 'BIN';
file = ['0' num2str(person) '_dst.bin'];

frameindex = (gesture-1)*20+frame;

% Dataset
%   datatype = 'UPT';
%   file = 'all';
%   frameindex = 'all'
%% Filtering procedure
% using median filter filtertype = 'median';
filtertype = 'median';

%% Feature
% featuretype = 'HistOfNormals2Points', 'HistOfNormals2Surfs'; 'curvatures'
featuretype = 'HistOfNormals2Points';
%% Classification
% dist_func = 'euclidean', 'cosin', 'spearman', 'seuclidean', 'cityblock', 'minkowski', 'chebychev', 'correlation', 'hamming', 'jaccard'  //see pdist2 
% @chi_square_statistics_fast, @kullback_leibler_divergence, etc; //see Schauerte - Histo dist

% TOP
% 'cosine' - best?
% 'correlation' close to the best
% 'cityblock' - v good
% 'chebychev' - medium
% 'minkowski' - medium
% 'hamming' - bad
% 'jaccard' - close to worst
% 'seuclidean' - worstest?

dist_func = {'cosine', 'correlation', 'cityblock', 'chebychev', 'minkowski', 'hamming', 'jaccard', 'seuclidean'};
sel_dist_func = dist_func{1};

% consider the first Kg votes 
% Kg = 56 seems to be optimal value;
Kg = 56;

%%
disp('Configure done!')
toc
disp('=====================================')