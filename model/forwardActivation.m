% computes activation of a given layer
function [cache A] = forwardActivation(w, b, A_prev, non_liearity)
  z = A_prev*w'+b;
  if strcmp(non_liearity, 'relu')
    A = relu(z);
  elseif strcmp(non_liearity, 'sigmoid')    
    A = sigmoid(z);
  end
  cache = {w, b, A_prev, z};
end
