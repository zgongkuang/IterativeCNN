function dump(data, filename, type)
%function dump(data, filename, type)

if nargin < 3
	type = 'float32';
end

fid = fopen(filename, 'wb');
fwrite(fid, data, type);
fclose(fid);
