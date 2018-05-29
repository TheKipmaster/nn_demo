function predict = predict(X, params)
  [cache, A2] = forwardProp(X, params);
  predict = [A2 > 0.5];
end