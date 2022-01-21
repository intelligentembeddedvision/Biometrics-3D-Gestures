function [D, new_hand, noHandFlag, img] = bbSegmentation(filter_dist,img)
%Segment the hand

disp('Segmenting ...')
tic

MinArea = 80;
dim = size(filter_dist);

noHandFlag = 0;

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
                img = img.*(handarray>0);
                if flag > 0
                    fprintf('%d spikes found at %d %d %d %d',flag, i,j,k);
                    fprintf('\r');
                end
                hand{i,j,k} = bbRectangle(handarray);
                img = bbRectangle(img);
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
                                    new_img(pp,oo) = img(p,o);
                                    pp=pp+1;
                                end
                                oo=oo+1;
                            end
                            
                              D{i,j,k} = bbTakeOutZero(new_hand{i,j,k});
                        else
                            new_D=stats(idx(1,m)).PixelList;
                        end
                    end
                    if exist('new_img','var')
                        img = new_img;
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
    if exist('new_hand','var')
        [new_hand img] = removeForearm(new_hand{1},img);
        img = bbRectangle(img);
    else
        noHandFlag = 1;
        new_hand = 0;
    end
    
disp('Segmentation done!')
toc
disp('=====================================')

end

function [hand img] = removeForearm(hand, img)
% pentru debug. se poate opri executia si vizualiza numarul setului de date 
%     pause(1);
    global dataSetCount;
%         input('');

% se extrag punctele din imagine diferite de zero cu coordonate si valoare
    data = bbTakeOutZero(hand);
%     se pastreaza doar coordinatele
    data = data(:,1:2); 
      
%     se face analiza PCA pe coordonatele punctelor
    [x y] = pca(data);
     
%  se calculeaza pozitia centrului noului sistem de axe fata de sistemul de axe vechi   
    absoluteOffset = data-(x*y')';    
    absoluteOffset = absoluteOffset(1,:);  
    
% se rearanjeaza datele astfel incat sa fie in sens crescator al noilor coordonate x (coordonata x=y(1), coordonata y=y(2))  
    [a i] = sort(y(:,1));
    y= [a y(i,2)];
    
% se calculeaza raportul dintre deviatiile maxime pe axele x si y pentru a cuantifica elongatia formei.
% daca mana contine antebrat elongatia va fi mare, in caza contrar mica
    ratio =max(abs(y(:,1)))/max(abs(y(:,2)));
    
    foreArm = zeros(size(hand));
    
% codul de mai jos comentat (7 linii) este util in faza de debugging pentru a vizualiza modul in care sunt determinate axele in urma analizei PCA    
    
%     value = max(max(hand))*1.3;
%     hand(round(absoluteOffset(1,1)),round(absoluteOffset(1,2))) = value;
%     hand(round(absoluteOffset(1,1)+10*x(1,1)),round(absoluteOffset(1,2)+10*x(2,1))) =value;
%     hand(round(absoluteOffset(1,1)+20*x(1,1)),round(absoluteOffset(1,2)+20*x(2,1))) =value;
%     hand(round(absoluteOffset(1,1)+10*x(1,2)),round(absoluteOffset(1,2)+10*x(2,2))) =value;
%     figure(1), imshow(hand,[]);
%     title(num2str(dataSetCount));
        

% daca factorul de elongatie este sub un prag (determinat prin incercari)se
% renunta la segmentare
    if ratio < 1.5
%         figure(2), imshow(hand,[]);
%         title(num2str(dataSetCount));
        return
    end
    
    
%   se initializeaza dimensiunile ferestrelor de cautare si a functiilor de segmentare      
    
    dataSize = size(y,1);
    track = [];
    trackIndex = [];
    ftrack = [];
    initialObj1Size = 300;
    forearmWindow = 200;
  
%     daca dimensiunea setului de date este mai mica decat dimensiunea minima de pornire se renunta la segmentare 
    
    if dataSize < initialObj1Size + forearmWindow +1
%         figure(2), imshow(hand,[]);
%         title(num2str(dataSetCount));
        return        
    end
    
    
%     se modifica dimensiunea ferestrei variabile
    for i=initialObj1Size:dataSize-(forearmWindow+1)
%    se selecteaza datele aferente mainii (obj1) si incheieturii (obj2)
        obj1 = y(1:i,:);
        obj2 = y(i+1:i+forearmWindow+1,:);
      
% se calculeaza valoarea functiei de segmentare pentru dimensiunea curenta a ferestrei de segmentare
% se salveaza si indexul aferent (variabila "track" da valoarea functiei de
% decizie, iar variabila "trackIndex" da valoarea indexului in cadrul
% setului de date asociat fiecarei valori din "track")

        trackIndex = [trackIndex;i];
        track = [track; mean(abs(obj2(:,2)))/mean(abs(obj1(:,2)))];
    end

%  se filtreaza functia de segmentare
    fsize = 30;
  
    for j=1:size(track)-fsize
        ftrack(j)=min(track(j:j+fsize));
        ftrackIndex(j) = trackIndex(j+fsize);
    end


%     h = figure(3), set(h,'OuterPosition',[900 500 400 400]);
%     plot(ftrackIndex,ftrack,'LineWidth',1.5);    
%     title('No Segmentation');
%     xlabel('Point Index');
%     ylabel('Decision Function');   


% algoritm pentru stabilirea indexului de segmentare. se analizeaza functia
% de segmentare. in acest scop se calculeaza urmatoarele variabile
% "strackDir" 
% daca starckDir(i) = '1' functia de segmentare este in crestere
% daca strackDir(i) = '-1'  functia de segmentare este in scadere
% 
% "strack"
% strack(i) contine pentru cate indexuri consecutive functia de segmenatre
% a fost crescatoare sau descrescatorare conform strackDir(i)
% 
% "strackIndex"
% strackIndex(i) face corespondenta intre informatia din strack(i),
% strackDir(i) si indexul intial din setul de date

    index = 1;
    strack(1) = 1;

    if size(ftrack)<10  
%         figure(2), imshow(hand,[]);
%         title(num2str(dataSetCount));
        return 
    end
    if ftrack(2)>=ftrack(1)
        strackDir(1) = 1;
    else
        strackDir(1) = -1;
    end
    strackIndex = ftrackIndex(1);
      
    for j = 3:size(ftrack,2)
        if ftrack(j)==ftrack(j-1)

        else
            if ftrack(j)>ftrack(j-1)
               if strackDir(index)==1
                    strack(index) = strack(index) + 1; 
               else
                    index = index + 1;
                    strack(index) = 1;
                    strackDir(index) = 1;
                end 
            else
               if strackDir(index)==-1
                    strack(index) = strack(index) + 1; 
               else
                    index = index + 1;
                    strack(index) = 1;
                    strackDir(index) = -1;
               end               
            end
        end
        strackIndex(index) = ftrackIndex(j);
    end    
   
    
%     se compacteaza informatia referitoare la functia de segmentare
%     disponibila in "strack", "strackDir" si strackIndex"
%    se elimina din analiza zonele in care functia de segmentare isi schimba monotonia pentru un scurt timp (numar redus de indexuri) 
% astfel de ex: daca numarul maxim de indexuri pentru care functia de
% segmentare nu isi schimba monotonia este 500, se va elimina din analiza
% orice schimbare de monotonie care nu dureaza mai mult de 500*0.03 = 15
% indexuri in cadrul functiei de segmentare
%
    threshold = max(strack)*0.03;
    
    flag2 = 1;

       j = 1;
       while flag2
          if strack(j) < threshold

             strack(j) = [];
             strackDir(j) = [];
             strackIndex(j) = [];
             if size(strack,2)>1
                flag3 = 1;
             else
                flag3 = 0;
             end
             k = 2;
             while flag3
                if strackDir(k)==strackDir(k-1)
                   strack(k) = strack(k)+strack(k-1);
                   strack(k-1) = [];
                   strackDir(k-1) = [];
                   strackIndex(k-1) = [];
                   if k > size(strack,2)
                       flag3 = 0;
                   end
                else
                   k =k+1;
                   if k > size(strack,2)
                      flag3 = 0; 
                   end
                end
             end
             if j>1
                j = j-1;
             end
             if j > size(strack,2)
                 flag2 = 0;
             end
          else
             j = j+1;
             if j > size(strack,2)
                 flag2 = 0;
             end
          end
       end   
%     sfarsit algoritm de compactare

% daca nu exista schimbari de monotonie segmentarea este abandonata
    if size(strack,2)==1  
        figure(2), imshow(hand,[]); 
        title(num2str(dataSetCount));
        return
    end      
    
%     se implementeaza algoritmul din lucrare fig. 9 pag 7.
    flag = 0;
    for j = 1:size(strack,2)
        if strackDir(j) == -1
            if flag == 0
               flag = 1;
               minimum = ftrack(find(ftrackIndex==strackIndex(j)));
               k = j;
               minimum1 = minimum;
               k1 = k;
            else
               if ftrack(find(ftrackIndex==strackIndex(j))) < minimum
                  minimum =  ftrack(ftrackIndex==strackIndex(j));
                  k = j;
               end
            end

        end
    end
    
    if minimum/minimum1 > 0.9
       minimum = minimum1;
       k = k1; 
    end
    
    if k == size(strack,2)
%                 figure(2), imshow(hand,[]);
%                 title(num2str(dataSetCount));
                return
    end
    segIndex = strackIndex(k);
    
%     se vizualizeaza functia de segmentare si decizia de segmentare luata
%     h = figure(3), set(h,'OuterPosition',[900 500 400 400]);
%     plot(ftrackIndex,ftrack,[segIndex segIndex],[0.95*minimum 1.05*minimum],'LineWidth',1.5);
%     title(['Segmentation Index:' num2str(segIndex)]);
%     xlabel('Point Index');
%     ylabel('Decision Function');
    
% se realizeaza segmentarea in sistemul de coordonate initial 
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
                img(i,j) = 0;
            end
        end
    end
  
    
%     figure(2), imshow(hand,[]);
%     title(num2str(dataSetCount));

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

