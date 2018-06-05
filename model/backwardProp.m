% compute gradients for a neural network of size L, L-1 relu activation layers and a final sigmoid layer
function grads = backwardProp(cache, AL, Y, lambda)
  grads = {};
  L = length(cache);
  m = size(AL(:,1));

  dAL = -(Y./AL - (1-Y)./(1-AL));
  current_cache = cache{L};
  [grads{L}{1}, grads{L}{2}, grads{L}{3}] = backwardActivation(dAL, current_cache, lambda, 'sigmoid');

  for l=L-1:-1:1
    current_cache = cache{l};
    dA = grads{l+1}{1};
    [grads{l}{1}, grads{l}{2}, grads{l}{3}] = backwardActivation(dA, current_cache, lambda, 'relu');
  end
end
