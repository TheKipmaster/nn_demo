function g = sigmoidPrime(z)
  g = (exp(-z)./(1+exp(-z)).^2);
end
