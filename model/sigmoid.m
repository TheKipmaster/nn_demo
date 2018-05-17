% Compute sigmoid of z
function g = sigmoid(z)
  g = 1 ./ (1+exp(-z));
end
