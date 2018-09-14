function imshow(varargin)
% imshow -- Image display of object assuming arbitrary values
%  Usage
%    same as imagesc, but set square pixel and show colorbar
%  
%  To show without colorbar, use imshow(...,'nc');
%


level = 256;

if strcmp(varargin(end),'nc')
  cb = 0;
	  varargin(end) = [];
	  narg = nargin - 1; 
	else
	  cb = 1;
	  narg = nargin;
	end
	
	imagesc(varargin{:});
	colormap(pink(level));
    %colorbar('southoutside');

	axis image;

	if narg < 3 
	dumy = deal(varargin(end));
	M = [cellfun('size',dumy,1) cellfun('size',dumy,2)];
	if cb
	  if M(1) > M(2) 
	    colorbar('vert','FontSize',12);
	  else
	    colorbar('vert','FontSize',12);
	  end
	end
	
	% set picture size
	while max(M) > 768
	  M = M/2;
	end
	while max(M) <= 256
	  M = M*2;
	end

	  h = gcf; 
	  
	  axesHandle = gca;
	  p = get(axesHandle,'position');
	  p2= get(h, 'position');
	  p3 = ceil(M(2:-1:1)./p(3:4));
	  set(h,'position', [p2(1) p2(2)+p2(4)-p3(2) p3]);
	  set(axesHandle, 'position', [p(1) p(2) M(2:-1:1)./p3]);
	
	  end 
	

    
zoom on;
