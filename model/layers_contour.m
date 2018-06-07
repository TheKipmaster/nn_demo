function [train_accuracy, valid_accuracy] = layers_contour(hidden_sizes1, hidden_sizes2)
  % static hyperparameters
  num_iter         = 500;
  learning_rate    = 0.05;
  lambda           = 0;
  % load dataset
  load('dataset.mat');
  X = (X-128)/255; % normalize inputs
  m = size(X, 1);

  X_train = X(1:409, :);
  X_valid = X(410:511, :);

  Y_train = Y(1:409, :);
  Y_valid = Y(410:511, :);

  for i=1:length(hidden_sizes1)
    for j=1:length(hidden_sizes2)
      t_accuracy = [];
      v_accuracy = [];
      for t=1:10
        if hidden_sizes2(j) == 0
          dimensions = [576, hidden_sizes1(i), 1];
        else
          dimensions = [576, hidden_sizes1(i), hidden_sizes2(j), 1];
        end
        % Training loop
        params = initializeDeep(dimensions);      % Initialize weight matrices with values between 0 and 1
        [params, costs] = training(params, num_iter, X_train, Y_train, learning_rate, lambda);
        train_prediction = predict(X_train, params);
        valid_prediction = predict(X_valid, params);

        t_accuracy = [t_accuracy; accuracy(train_prediction, Y_train)];
        v_accuracy = [v_accuracy; accuracy(valid_prediction, Y_valid)];
      end
      train_accuracy(i,j) = sum(t_accuracy)/10;
      valid_accuracy(i,j) = sum(v_accuracy)/10;
      fprintf('Training epoch #%d done.', i+(j-1)*length(hidden_sizes1));
    end
  end

  % mesh(layers1, layers2, z_valid);
  % xlabel('layer one size');
  % ylabel('layer two size');
  % zlabel('precision');
  % title('validation set precision across hidden layer sizes');

end
