% reset variable space, close all open visualization windows, Clear screen
clear; close all; clc;
pkg load optim % load package for computing derivatives

% Setup hyperparameters and variables of Neural Network
input_layer_size = 576;                   % 24x24 input images
num_hidden1      = 100;                   % size of first hidden layer
num_hidden2      = 25;                    % size of second hidden layer
num_labels       = 1;                     % size of output layer
dimensions       = [input_layer_size, ... %
                    num_hidden1, ...      % put all dimensions into vector for easier paramhandling
                    num_hidden2, ...      %
                    num_labels];          %
num_iter         = 100;                   % number of learning iterations
learning_rate    = 0.001;                 %
lambda           = 0;                     % regularization coeficient
params = initializeDeep(dimensions);      % Initialize weight matrices with values between 0 and 1
grads = {};                               %
costs = [];                               % to keep track of the cost

% Load and vizualise dataset
fprintf('Loading and Visualizing Data ...\n')

load('dataset.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

X_train = X(1:306, :);
X_test = X(307:409, :);
X_valid = X(410:511, :);

Y_train = Y(1:306, :);
Y_test = Y(307:409, :);
Y_valid = Y(410:511, :);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

% Training loop
for i=1:num_iter

  % Run forward propagation
  [cache, AL] = forwardProp(X, params);

  % compute cost
  cost = computeCost(AL, Y, lambda);

  % Run backward propagation
  grads = backwardProp(cache, AL, Y);

  % Update parameters
  params = updateParams(params, grads, alpha);

  % Print the cost every 100 training example
  % if mod(i, 100) == 0
    fprintf('Cost after iteration %d: %f\n', i, cost);
    costs = [costs ; cost];
  % end

end
