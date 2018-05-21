% define useful variables
myFolder = './g_negatives';
filePattern = fullfile(myFolder, '*.png');
pngFiles = dir(filePattern);
X = [];

% check if the specified path exists
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

% roll all images into an MxN matrix, where:
% M is the number of images and N the number of pixels in each image.
for k = 1:length(pngFiles)
  baseFileName = pngFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  imageMatrix = imread(fullFileName);
  imageVector = imageMatrix(:)'; % turn matrix into row vector
  if max(imageVector) == 1 % convert B&W to Grayscale
    imageVector = imageVector*255;
  end
  imageVector = double(imageVector); % cast imageVector as double
  X = [X; imageVector];
end

% make labels vector
Y = zeros((size(X)(1)), 1);
% save matrix to .mat file along with labels
save('model/negative_dataset.mat', 'X', 'Y');
