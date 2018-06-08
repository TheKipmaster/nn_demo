function [acc, rec, pre, f1s] = evaluate_model(Y_hat, Y)

  % accuracy = length(find([Y_hat == Y]))/length(Y_hat);

  TP = length(find([Y_hat == 1 & Y == 1]));
  FP = length(find([Y_hat == 1 & Y == 0]));
  TN = length(find([Y_hat == 0 & Y == 0]));
  FN = length(find([Y_hat == 0 & Y == 1]));

  acc = (TP+TN)/(FP+FN+TP+TN);
  rec = TP/(TP+FN);
  pre = TP/(TP+FP);
  f1s = 2*(rec * pre) / (rec + pre);

end
