function [dist img] = bbGetDistances(fromwhere, file, frameindex)
disp('Getting the distances ...')
tic
if strcmp(fromwhere, 'PMD')
    
    % test for file existence to be add
    
    
    %% Load plugins
    
    % From camera
    % hnd = pmdOpen ('camcube3.W32.pap', '', 'camcubeproc.W32.ppp', '')
    
    % From file
    hnd = pmdOpen ('D:\PMD\PMDMDK\MDK_mexw32\pmdfile.W32.pcp', file, 'D:\PMD\PMDMDK\MDK_mexw32\camcubeproc.W32.ppp', '');
    %%
    
    pmdSourceCommand (hnd, 'GetNumberOfFrames');
    % pmdSourceCommand (hnd, 0, 0, “Open recorded.pmd”);
    % pmdSourceCommand (hnd, 0, 0, “Pause”);
    % pmdSourceCommand (hnd, 0, 0, “Reset”);
    
    
    % use frameindex instead of 1000!
    pmdSourceCommand (hnd, 'SetFrame 1000');
    % pmdSourceCommand (hnd, 0, 0, “Start”);
    % pmdSourceCommand (hnd, 0, 0, “Stop”);
    
    % for i=1:1085
    %     disp(i);
    %     pmdUpdate (hnd);
    % end
    % (Calculate and) return distances
    
    pmdUpdate (hnd);
    pmddist = pmdGetDistances (hnd);
    
    % Disconnect and close plugins
    
    %Got source data and data desc:
    %pmdUpdate(hnd);
    pmdsrc = pmdGetSourceData(hnd);
    ddesc = pmdGetSourceDataDescription(hnd);
    
    %Got time of frame:
    ddesctime = ddesc.std.reserved1 + ddesc.std.reserved2/1000000;
    
    %Calculated images:
    %  pmdamp = rot90(pmdCalcAmplitudes(pmdhdl,ddesc,pmdsrc),1);
    %  pmdflg = rot90(pmdCalcFlags(pmdhdl,ddesc,pmdsrc),1);
    dist{1,1,1} = rot90(pmdCalcDistances(hnd,ddesc,pmdsrc),-1);
    %  pmdxyz = pmdCalc3DCoordinates(pmdhdl,ddesc,pmdsrc);
    %  PMDX = rot90(pmdxyz(:,:,1),1);
    %  PMDY = rot90(pmdxyz(:,:,2),1);
    %  PMDZ = rot90(pmdxyz(:,:,3),1);
    
    pmdClose (hnd)
    
elseif strcmp(fromwhere, 'BIN')
    fid = fopen(file, 'r');
    if strcmp(file(3),'_')
        fid_image = fopen([file(1:3) 'int.bin'], 'r');
    else
        fid_image = fopen([file(1:4) 'int.bin'], 'r');
    end
    % 6 gestures x 20 frames x 200x200 of doubles
    alldists = fread(fid, 'double');
    allpixels = fread(fid_image,'double');
    fclose(fid);
    fclose(fid_image);
    s = size(alldists);
    ralldists = reshape(alldists,200,s(1)/200);
    rallpixels = reshape(allpixels,200,s(1)/200);
    % f1: 1 ... val
    % f2: val+1 ... 2 x val
    % f3: (2 x val)+1 ... 3 x val
    % ...
    % fn: ((n-1) x val) + 1 ... n x val
    val = 200*200;
    dist{1,1,1} = reshape(ralldists(((frameindex-1)*val)+1:frameindex*val),200,200);
    img{1,1,1} = reshape(rallpixels(((frameindex-1)*val)+1:frameindex*val),200,200);
    
elseif strcmp(fromwhere, 'UPT')
    % 01 02 03 04 07r 08 09 10 11 right hand
    % 05 06 07l left hand
    
%    all = {'01_dst.bin', '02_dst.bin', '03_dst.bin', '04_dst.bin', '07r_dst.bin', '08_dst.bin', '09_dst.bin', '10_dst.bin', '11_dst.bin'};
   
   %RecognitionRate =

   % 6.7708    4.4792    5.0000    4.1667    3.5417    7.0833    3.9583    5.3125    4.4792
%    \\  all = {'11_dst.bin', '10_dst.bin'}
%    all = {'01_dst.bin','11_dst.bin', '10_dst.bin'}
   %all = {'01_dst.bin','11_dst.bin', '10_dst.bin'}
    all = {'01_dst.bin', '10_dst.bin'};
   % all = {'02_dst.bin'};
    % get the number of persons
    dim = size(all);
    val = 200*200;
    gestures = 6;
    frames = 20;
    for i=1:dim(2)
        fid = fopen(all{i}, 'r');
        alldists = fread(fid, 'double');
        fclose(fid);
        s = size(alldists);
        ralldist = reshape(alldists,200,s(1)/200);
        frameindex = 1;
        for j = 1:gestures
            for k = 1:frames
            % dist {person, gest, frame}
            dist{i,j,k} = reshape(ralldist(((frameindex-1)*val)+1:frameindex*val),200,200);
            frameindex = frameindex + 1;
            end
        end
    end
    
end
disp('Getting the distances done!')
toc
disp('=====================================')
