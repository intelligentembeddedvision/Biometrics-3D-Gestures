
function [Error, ConfMatrix] = bbClassify(features, dist_func, Kg)

% P = total # of persons = 9 right hand + 3 left hand, p = current person
% G = total # of gestures per person = 6, g = current gesture
% F = total # of selected frames per gesture = 20, f = current frame

% Error evaluation
% Leave one out person method


%nonemptyelements = length(find(~cellfun(@isempty,hand(2,1,:))))

disp('Classifying ...')
tic
 
dim = size(features);
P = dim(1);
G = dim(2);
F = dim(3);

% randomly select a person from P persons
% Generate values from the uniform distribution on the interval [a, b].
% p = a + (b-a).*rand(100,1);
% p = round(1 + (P-1).*rand(1,1))



for i=1:P
    correct = 0;
    Y = bbGetPminus1(i, features, P, G, F);
    for j=1:G
        for k=1:F
            if (~isempty(features{i,j,k}))
                X = (features{i,j,k}(:))';
                Dist = pdist2(X, Y, dist_func);
                [Ysort,I] = sort(Dist,'ascend');
                RecGesture = bbIndex2Gest(I);
                CountGesture = hist(RecGesture(1,1:Kg),G);
                [MaxCG, RecG] = max(CountGesture);
                if RecG==j
                    correct = correct+1;
                end
            end
        end
    end
    % ConfMatrix(i)
    nonemptyframes = length(find(~cellfun(@isempty,features(i,j,:))));
    RecognitionRate(i)=(correct/((P-1)*G*nonemptyframes))*100;
end


RecognitionRate
Error = 100 - (sum(RecognitionRate))/P

%not yet
ConfMatrix = 0;


disp('Classification done!')
toc
disp('=====================================')


function Y = bbGetPminus1(oneout, features, P, G, F)
l=1;
for i=1:P
    if i == oneout
        continue;
    end
    for j=1:G
        for k=1:F
             if (isempty(features{i,j,k}))
                 disp('EMPTY!')
             end
            if (~isempty(features{i,j,k}))
                Y(l,:) = features{i,j,k}(:);
                l = l+1;
            end
        end
    end
end

function gesture = bbIndex2Gest(Index)
% 1...20    p=1, g=1
% 21...40   p=1, g=2
% ...
% 101...120 p=1, g=6
% ---------------------
% 121...140 p=2, g=1
% 141...160 p=2, g=2
% ...
% 221...240 p=2, g=6
% ---------------------
% 241...260 p=3, g=1
% 261...280 p=3, g=2
% ...
% 341...360 p=3, g=6
% ---------------------
% i = (p-1)GF + (g-1)F + f
% p = (i-1)div(GF) + 1
% g = ((i-1)mod(GF))div(F) +1

[q1, r1] = quorem(Index-1,sym(120));
[q2, r2] = quorem(mod(Index-1,120), sym(20));
gesture = double([q2 + 1; q1 + 1]);