% computes partial derivatives of w, b, and A with respect to the cost
function [dA_prev, dw, db] = backwardActivation(dA, cache, non_liearity)
  [w, b, A_prev, z] = cache{:};
  m = size(A_prev, 1);

  % dz = dA.*arrayfun(@(z)(deriv(non_liearity, z)), z);
  if strcmp(non_liearity, 'sigmoid')
    dz = sigmoidPrime(z);
  elseif strcmp(non_liearity, 'relu')
    dz = reluPrime(z);
  end

  dw = (dz'*A_prev)/m;
  db = sum(dz, 1)/m;
  dA_prev = dz*w;
end
