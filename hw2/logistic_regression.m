function [new_accu, train_accu] = logistic_regression(train_data, train_label, new_data, new_label)
% calculating training accuracy
row = size(train_data,1);
coeff = mnrfit(train_data, train_label);
% predicted = mnrval(coeff, train_data, 'model', 'nominal');
predicted = mnrval(coeff, train_data);
[~,result] = max(predicted,[],2);
train_accu = sum(result==train_label)/row;

% calculating validation/test accuracy
row = size(new_data,1);
% predicted = mnrval(coeff, new_data, 'model', 'nominal');
predicted = mnrval(coeff, new_data);
[~,result] = max(predicted,[],2);
new_accu = sum(result==new_label)/row;

end

