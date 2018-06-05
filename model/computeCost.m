% Compute cost based on the final output of the network
function J = computeCost(AL, Y, params, lambda)
  m = size(Y, 1);
  L = length(params);
  reg = 0;

  for l=1:L
    reg += sum(sum(params{l}{1}.^2));
  end
  unregJ = (Y'*log(AL)+(1-Y')*(log(1-AL)))/-m;
  J = unregJ + (lambda/2*m)*reg;
end
