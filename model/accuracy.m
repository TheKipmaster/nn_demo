function accuracy = accuracy(Y_hat, Y)
    accuracy = length(find([Y_hat == Y]))/length(Y_hat);
end
