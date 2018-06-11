% Recieves a vector of values between 100 and 779 used to measure accuracy of
% neural network with various dataset sizes.
function accuracy = dataset_size_graphic(sizes)

  % check if there is a number greater than 779 in sizes
  if [sizes > 779]
    errorMessage = sprintf('Error: input cannot contain numbers bigger than 779 :\n');
    uiwait(warndlg(errorMessage));
    return;
  end

  % Setup hyperparameters and variables of Neural Network
  [dimensions, num_iter, learning_rate, lambda] = set_hyperparams();
  rounds = 10; % number of training rounds over which to compute average accuracy

  load('dataset.mat');
  X = (X-128)/255; % normalize inputs


  t_accuracy = v_accuracy = train_accuracy = valid_accuracy = [];
  for s=1:length(sizes)
    % size(s) = 90% of total dataset
    X_train = X(1:sizes(s), :);
    Y_train = Y(1:sizes(s), :);

    % breakoff = 10% of total dataset
    breakoff = idivide((0.1*sizes(s)), 0.9, 'ceil');
    X_valid = X(879-breakoff:end, :);
    Y_valid = Y(879-breakoff:end, :);

    % Training loop
    for i=1:rounds
      params = initializeDeep(dimensions); % Initialize weight matrices with values between 0 and 1
      [params, costs] = training(params, num_iter, X_train, Y_train, learning_rate, lambda); % Optimize weights

      % Compute predictions based on optimized weights
      train_predictions = predict(X_train, params);
      valid_predictions = predict(X_valid, params);

      % Save accuracy metrics
      t_accuracy(i) = evaluate_model(train_predictions, Y_train);
      v_accuracy(i) = evaluate_model(valid_predictions, Y_valid);
    end
    % Saves average accuracy in a vector
    train_accuracy = [train_accuracy; sum(t_accuracy)/rounds];
    valid_accuracy = [valid_accuracy; sum(v_accuracy)/rounds];

    % Gives feedback to user on how far the program is from finishing
    fprintf('Training epoch #%d done. %d epochs left\n', s, length(sizes)-s);
    fflush(stdout);
  end
  accuracy = [train_accuracy, valid_accuracy];

  % Plot graphic
  plot(sizes, accuracy(:, 1), ';training;', 'linewidth', 4, sizes, accuracy(:, 2), ';validation;', 'linewidth', 4);
  xlabel('number of training examples');
  ylabel('accuracy');
  title('accuracy across training set sizes');
end
