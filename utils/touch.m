function data = touch(fname, format, length)
if nargin < 2
	format = 'float32';
	length = inf;
end

if nargin < 3
	length = inf;
end

fid = fopen(fname, 'rb');
data = fread(fid, length, format);
fclose(fid);
