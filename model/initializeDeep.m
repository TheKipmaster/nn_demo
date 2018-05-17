function params = initializeDeep(ThetaDims)
  L = length(ThetaDims);
  params = [];
  for l=2:L
    wl = randInitializeWeights(ThetaDims(l-1), ThetaDims(l));
    bl = zeros(ThetaDims(l-1), 1);
    params = [params; wl(:); bl(:)];
  end
end
