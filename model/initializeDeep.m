function params = initializeDeep(ThetaDims)
  L = length(ThetaDims);
  params = {};
  for l=2:L
    wl = randInitializeWeights(ThetaDims(l-1), ThetaDims(l));
    bl = zeros(1, ThetaDims(l));
    params{l-1}{1} = wl;
    params{l-1}{2} = bl;
  end
end
