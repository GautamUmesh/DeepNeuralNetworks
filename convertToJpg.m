function [] = convertToJpg(path, train)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% input parameters
% path - the path to the mnist image file 'mnist/train-images-idx3-ubyte'
%        or 'mnist/t10k-images-idx3-ubyte'
% train - boolean; true if converting training set, false if test set.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fp = fopen(path, 'rb');
assert(fp ~= -1, ['Could not open ', path, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', path, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);

if (train == true)
    directory = 'trainImages';
else
    directory = 'testImages';
end

mkdir(directory);

for i = 1:numImages
    imwrite(images(:, :, i), [directory '/image' int2str(i) '.jpg'], 'jpg');
end

fclose(fp);
end