% Initialization
clear; close all; clc;

% Setup architecture of neural Network
input_layer_size = 576; % 24x24 input images
num_hidden1      = 100; % size of first hidden layer
num_hidden2      = 25;  % size of second hidden layer
num_labels       = 2;   % size of output layer
dimensions       = [input_layer_size, ... % put all dimensions into
                    num_hidden1, ...      % vector for easier param
                    num_hidden2, ...      % handling
                    num_labels];
num_iter         = 1000; % number of learning iterations
learning_rate    = 0.01; % a.k.a. alpha
lambda           = 0.03; % regularization coeficient

% Load and vizualise dataset
fprintf('Loading and Visualizing Data ...\n')

load('dataset.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

% Initialize weight matrices values between 0 and 1
params = initializeDeep(dimensions);

% for i=1:num_iter
  cache = {};
  % Run forward propagation
  [cache, AL] = forwardProp(X, params);

  % Run backward propagation

  % Update parameters

% end
