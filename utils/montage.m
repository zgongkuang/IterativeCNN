function imgout = montage(v, format )

% function img = montage(v)
%
% mirror function of montage in image toolbox
% display a 3D volume 

M = size(squeeze(v));

if M(3)==1   % change from 4D to 3D
  M(3) = M(4);
end
v = reshape(v,[M(1) M(2) M(3)]);

if nargin < 2 
  n1=round(sqrt(prod(M))/M(2));
  n2 = ceil(M(3)/n1);
else
  n1 = format(2);
  n2 = format(1);
end


img = ones(n2*M(1), n1*M(2)) * min(v(:));

for i=1:n2-1
  img(1+(i-1)*M(1):i*M(1),:) = reshape( v(:,:,1+(i-1)*n1:i*n1),[M(1) M(2)*n1]);
end
i=n2;
coln =( M(3) - (i-1)*n1)*M(2); 
img(1+(i-1)*M(1):i*M(1),1:coln) =  reshape( v(:,:,1+(i-1)*n1:M(3)),[M(1) coln]);

imshow( img );
axis off; hold on; 
cm = colormap; c = [120,95,112]/255.0;%[0.8,0.0,0.5];%cm(end,:);%[0.8,0.0,0.5];
% draw separate lines
line([0.5, size(img,2)+0.5], [0.5, 0.5], 'color', c, 'linew', 3.0);
line([0.5, size(img,2)+0.5], [0.5, 0.5]+size(img,1), 'color', c, 'linew', 3.0);
line([0.5, 0.5], [0.5, size(img,1)+0.5], 'color', c, 'linew', 3.0);
line([0.5, 0.5]+size(img,2), [0.5, size(img,1)+0.5], 'color', c, 'linew', 3.0);
nh = size(img, 1) / size(v, 1);
for n=1:nh-1
line([0.5, size(img,2)+0.5], [0.5, 0.5]+n*size(v,1), 'color', c);
end
nv = size(img, 2) / size(v, 2);
for n=1:nv-1
line([0.5, 0.5]+n*size(v,2), [0.5, size(img,1)+0.5], 'color', c);
end
%
c=1;
for n=1:nh
	for m=1:nv
	if (c <= size(v,3))
		text((m-1)*size(v,2)+size(v,2)*0.1, (n-1)*size(v,1)+size(v,1)*0.2, num2str(c), 'color', 'y', 'fontsize',12);
		
%         if (c == 1)
%         text((m-1)*size(v,2)+size(v,2)*0.05, (n-1)*size(v,1)+size(v,1)*0.1, 'a', 'color', 'y', 'fontsize',18);
%         elseif (c==2)
%                     text((m-1)*size(v,2)+size(v,2)*0.05, (n-1)*size(v,1)+size(v,1)*0.1, 'b', 'color', 'y', 'fontsize',18);
%           elseif (c==3)
%                     text((m-1)*size(v,2)+size(v,2)*0.05, (n-1)*size(v,1)+size(v,1)*0.1, 'c', 'color', 'y', 'fontsize',18);
%                                 
%         end
        c = c + 1;
		end
	end
end
%
hold off;
axis on;
%title(sprintf('patch size: [%d %d]', size(v,1), size(v,2)));

if nargout == 1
  imgout = img; 
end
