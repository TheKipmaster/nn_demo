% Compute cost based on the final output of the network
function J = computeCost(AL, Y, lambda)
  m = size(Y, 1);
  J = (Y'*log(AL)+(1-Y')*(log(1-AL)))/-m;
end
