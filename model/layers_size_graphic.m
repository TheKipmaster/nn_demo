% Recieves two vectors os same length containing the values for the number of
% neurons in the first two layers of a neural network and computes corresponding
% accuracy for each combination of the values in each vector.
function [train_accuracy, valid_accuracy] = layers_size_graphic(hidden_sizes1, hidden_sizes2)
  % static hyperparameters
  [_, num_iter, learning_rate, lambda] = set_hyperparams();

  % load dataset
  load('dataset.mat');
  X = (X-128)/255; % normalize inputs
  m = size(X, 1);

  % Divide dataset into training and validation subsets
  X_train = X(1:779, :);
  Y_train = Y(1:779, :);

  X_valid = X(780:end, :);
  Y_valid = Y(780:end, :);

  I = length(hidden_sizes1);
  J = length(hidden_sizes2);
  for i=1:I
    for j=1:J
      t_accuracy = [];
      v_accuracy = [];
      for t=1:10
        if hidden_sizes2(j) == 0
          dimensions = [576, hidden_sizes1(i), 1];
        else
          dimensions = [576, hidden_sizes1(i), hidden_sizes2(j), 1];
        end
        params = initializeDeep(dimensions);      % Initialize weight matrices with values between 0 and 1
        [params, costs] = training(params, num_iter, X_train, Y_train, learning_rate, lambda); % Optimize weights

        % Compute predictions based on optimized weights
        train_predictions = predict(X_train, params);
        valid_predictions = predict(X_valid, params);

        % Save accuracy metrics
        t_accuracy(i) = evaluate_model(train_predictions, Y_train);
        v_accuracy(i) = evaluate_model(valid_predictions, Y_valid);
      end
      % Saves average accuracy in a matrix
      train_accuracy(i,j) = sum(t_accuracy)/10;
      valid_accuracy(i,j) = sum(v_accuracy)/10;
      % Gives feedback to user on how far the program is from finishing
      fprintf('Training epoch #%d done. %d epochs left\n', j+(i-1)*I, I*J - (j+(i-1)*I));
      fflush(stdout);
    end
  end

  % Plot meshgrid of accuracy vectorspace
  figure(1)
  mesh(hidden_sizes1, hidden_sizes2, valid_accuracy);
  xlabel('size of first layer');
  ylabel('size of second layer');
  zlabel('accuracy');
  title('validation set accuracy across hidden layer sizes, averaged over 10 training rounds');

  % Plot contour lines of accuracy vectorspace
  figure(2)
  contour(hidden_sizes1, hidden_sizes2, valid_accuracy);
  xlabel('size of first layer');
  ylabel('size of second layer');
  title('contour plot of validation set accuracy across hidden layer sizes, averaged over 10 training rounds');

end
