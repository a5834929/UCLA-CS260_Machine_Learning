function [new_accu, train_accu] = decision_tree(train_data, train_label, new_data, new_label, splitCriterion, minLeaves)
% calculating training accuracy
row = size(train_data,1);
tree = fitctree(train_data,train_label,'Prune','off','SplitCriterion',splitCriterion,'MinLeaf',minLeaves);
result = predict(tree, train_data);
train_accu = sum(result==train_label)/row;

% calculating validation/test accuracy
row = size(new_data,1);
result = predict(tree, new_data);
new_accu = sum(result==new_label)/row;

% view(tree, 'Mode', 'graph');
end

