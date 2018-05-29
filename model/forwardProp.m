% Compute final output for a neural network of depth L
%% AL is the activation of the final layer
%% cache saves values computed useful for backwardProp()
function [cache AL] = forwardProp(X, params)
  cache = {};
  A = X;
  L = length(params);

  for l=1:(L-1)
    A_prev = A;
    wl = params{l}{1};
    bl = params{l}{2};
    [cache{l}, A] = forwardActivation(wl, bl, A_prev, 'relu');
  end
  wL = params{L}{1};
  bL = params{L}{2};
  [cache{L}, AL] = forwardActivation(wL, bL, A, 'sigmoid');
end
