% define useful variables
myFolder = '/home/kipman/Documents/NEXT/Demo/positives';
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
  X = [X; imageVector];
end

% make labels vector
Y = ones((size(X)(1)), 1);
% save matrix to .mat file along with labels
save('dataset.mat', 'X', 'Y');