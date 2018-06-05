function [params, costs] = training(params, num_iter, X_train, Y_train, learning_rate, lambda)
    grads = {};                               %
    costs = [];                               % to keep track of the cost
    cost = [];


    % Training loop
    for i=1:num_iter

        % Run forward propagation
        [cache, AL] = forwardProp(X_train, params);

        % compute cost
        cost = computeCost(AL, Y_train, params, lambda);

        % Run backward propagation
        grads = backwardProp(cache, AL, Y_train, lambda);

        % Update parameters
        params = updateParams(params, grads, learning_rate);

        % Print the cost every 100 training example
        % if mod(i, 100) == 0
        %     fprintf('Cost after iteration %d: %f\n', i, cost);
        % end
        costs = [costs ; cost];
    end
end
