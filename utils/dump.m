function dump(data, filename, type)
%function dump(data, filename, type)

if nargin < 3
	type = 'float32';
end

fid = fopen(filename, 'wb');
fwrite(fid, data, type);
fclose(fid);
% 
% 
% clear;
% clc;
% close all
% fid = fopen('all_direct_geoCombine79.data','wb');
% for filenum = 1:4000
%     filenum
%     str = sprintf('Direct_plane79_%d.mat',filenum);
%     load (str);
%     fwrite(fid, yy_Measure, 'float32');
%     clear yy_Measure;
% end
% fclose(fid);    
    
