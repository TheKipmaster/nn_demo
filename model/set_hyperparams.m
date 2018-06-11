% Returns optimized hyperparams for network
function [dimensions, num_iter, alpha, lambda] = set_hyperparams()
  dimensions = [576, 288, 1];
  num_iter = 70;
  alpha = 1;
  lambda = 0;
end
