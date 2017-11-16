function [new_accu, train_accu] = knn_classify(train_data, train_label, new_data, new_label, k)

% calculating training accuracy
row = size(train_data,1);
result = zeros(row,1);
for i=1:row
    point = train_data(i,:);
    point = repmat(point,row,1); 
    dist = sum((train_data-point).^2,2);
    [~, sortedInd] = sort(dist, 'ascend');
    result(i) = mode(train_label(sortedInd(2:1+k)));
end
train_accu = sum(result==train_label)/row;

% calculating validation/test accuracy
row = size(new_data,1);
result = zeros(row,1);
for i=1:row
    point = new_data(i,:);
    point = repmat(point,size(train_data,1),1); 
    dist = sum((train_data-point).^2,2);
    [~, sortedInd] = sort(dist, 'ascend');
    result(i) = mode(train_label(sortedInd(1:k)));
end
new_accu = sum(result==new_label)/row;

end