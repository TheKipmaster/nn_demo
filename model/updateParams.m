function new_params = updateParams(old_params, grads, learning_rate)
  L = length(old_params);
  new_params = {};

  for l=1:L
    new_params{l}{1} = old_params{l}{1} - learning_rate*grads{l}{2};
    new_params{l}{2} = old_params{l}{2} - learning_rate*grads{l}{3};
  end
end
