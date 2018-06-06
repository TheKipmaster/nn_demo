% reset variable space, close all open visualization windows, Clear screen
clear; close all; clc;

% Setup hyperparameters and variables of Neural Network
input_layer_size = 576;                   % 24x24 input images
num_hidden1      = 288;                   % size of first hidden layer
% num_hidden2      = 25;                 % size of second hidden layer
% num_hidden3      = 3;                   % size of third hidden layer
num_labels       = 1;                     % size of output layer
dimensions       = [input_layer_size, ... %
                    num_hidden1, ...      % put all dimensions into vector for easier param handling
                    % num_hidden2, ...    %
                    % num_hidden3, ...    %
                    num_labels];          %
num_iter         = 500;                   % number of learning iterations
learning_rate    = 0.05;                  %
lambda           = 25;                    % regularization coeficient
rounds           = 1;                     % number of training rounds over which to compute average precision

% Load and vizualise dataset
fprintf('Loading and Visualizing Data ...\n')

load('dataset.mat');
X = (X-128)/255; % normalize inputs

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

X_train = X(1:409, :);
X_valid = X(410:511, :);

Y_train = Y(1:409, :);
Y_valid = Y(410:511, :);

displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;

train_accuracy = [];
valid_accuracy = [];
for i=1:rounds
  % Training loop
  params = initializeDeep(dimensions);      % Initialize weight matrices with values between 0 and 1
  [params, costs] = training(params, num_iter, X_train, Y_train, learning_rate, lambda);
  train_predictions = predict(X_train, params);
  valid_predictions = predict(X_valid, params);

  train_accuracy = [train_accuracy; accuracy(train_predictions, Y_train)];
  valid_accuracy = [valid_accuracy; accuracy(valid_predictions, Y_valid)];
end
[train_accuracy, valid_accuracy; sum(train_accuracy)/10, sum(valid_accuracy)/10]

%  Randomly permute examples
m = size(X_valid, 1);
rp = randperm(m);

%  Display random example and corresponding (ground truth) label as well as prediction
for i = 1:m
    % Display
    fprintf('\nDisplaying Example Image\n');
    displayData(X_valid(rp(i), :));

    pred = valid_predictions(rp(i),:);
    % translate binary output of network to concrete label
    if pred == 1
      answer = 'plane';
    else
      answer = 'not plane';
    end
    fprintf('\nNeural Network Prediction: %s \n', answer);

    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end
