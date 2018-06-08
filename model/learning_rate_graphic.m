function [lr_costs] = learning_rate_graphic(learning_rate, num_iter)
  % static hyperparameters
  hidden_sizes1    = 288;
%   hidden_sizes2    = 25;
  lambda           = 25;
  dimensions = [576, hidden_sizes1, 1];

  % load dataset
  load('dataset.mat');
  X = (X-128)/255; % normalize inputs
  m = size(X, 1);

  X_train = X(1:779, :);
  Y_train = Y(1:779, :);

  X_valid = X(780:end, :);
  Y_valid = Y(780:end, :);

  lr_costs = [];

  for i=1:length(learning_rate)
    params = initializeDeep(dimensions);      % Initialize weight matrices with values between 0 and 1
    [params, costs] = training(params, num_iter, X_train, Y_train, learning_rate(i), lambda); % Optimize weights

    % Compute predictions based on optimized weights
    train_predictions = predict(X_train, params);
    valid_predictions = predict(X_valid, params);

    lr_costs = [lr_costs, costs];
    fprintf('Training epoch #%d done. %d epochs left\n', i, length(learning_rate)-i);
    fflush(stdout);
  end

  lr_costs = [lr_costs; learning_rate]
  plot(1:num_iter, lr_costs);
  xlabel('t(iterations)');
  ylabel('cost');
  title('cost over time for different learning rates');

end
