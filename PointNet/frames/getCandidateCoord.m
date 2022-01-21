function coord = getCandidateCoord(dataPos, op)

[x y] = find(dataPos);
neighbors = zeros(size(dataPos));
for i = 1:size(x,1)
    kernelSize = 1;
    while 1
    if ((x(i)-kernelSize)>=1)& ((x(i)+kernelSize)<= size(dataPos,1))
        if ((y(i)-kernelSize)>=1)& ((y(i)+kernelSize)<= size(dataPos,2))
            window = dataPos((x(i)-kernelSize):(x(i)+kernelSize),(y(i)-kernelSize):(y(i)+kernelSize));
            if sum(sum(window)) < (kernelSize*2+1)^2
                break;
            end
        else 
            window = 0;
            break;
        end
    else 
        window = 0;
        break;
    end
    kernelSize=kernelSize+1;

    end
    neighbors(x(i),y(i)) = sum(sum(window))-1;
end

if strcmp(op, 'remove')
    flags = (neighbors==max(max(neighbors)));
end

if strcmp(op, 'add')
    neighborsData = reshape(neighbors, [size(neighbors,1)*size(neighbors,2) 1]);
    neighborsData(neighborsData==0) = [];
    [a,b] = hist(neighborsData,100);
    histRes = b(2)-b(1);
    c = find(cumsum(a)>(0.5*sum(a)));
    threshold = round(b(c(1)));
    flags = (neighbors>=threshold-ceil(histRes/2))&(neighbors<=threshold+ceil(histRes/2));
end
[flagsx, flagsy]= find(flags);
index = ceil(rand()*size(flagsx,1));
coord = [flagsx(index) flagsy(index)];