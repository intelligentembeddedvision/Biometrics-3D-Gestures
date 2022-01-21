function bb2D3Dplot(distance,verbose)  

dim = size(distance);

if (dim(1) == 1 && dim(2) == 1)
    dim(3)=1;
end
    for i=1:dim(1)
        for j=1:dim(2)
            for k=1:dim(3)
                dist = distance{i,j,k};
                bb2D3DplotCore(dist,verbose)
            end
        end
    end
end



function bb2D3DplotCore(dist,verbose)

            %%
            if verbose >=1
                figure
                imagesc(dist);
            end
            %%
            if verbose >= 2
                size_data=size(dist);
                vx = 0 : (size_data(1)-1);
                vy = 0 : (size_data(2)-1);
                [x,y] = meshgrid(vx,vy);
                
                figure
                meshc(x,y,dist');
                
                % xx=x(:);
                % yy=y(:);
                % zz=dist(:);
                % figure
                % scatter3(xx,yy,zz,1);
            end
            %%
            if verbose ==3
                figure
                scatter3(x(:),y(:),dist(:)',1);
                
                %surf?
            end
end

 % grid on
