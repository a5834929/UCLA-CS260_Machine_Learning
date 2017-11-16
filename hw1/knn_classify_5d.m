function [ points, labels ] = knn_classify_5d(train_data, train_label, k)

p = 0:0.005:1;
[x,y] = meshgrid(p,p);
points = [x(:) y(:)];
row = size(points,1);
result = zeros(row,1);

for i=1:row
    point = repmat(points(i,:),size(train_data,1),1); 
    dist = sum((train_data-point).^2,2);
    [~, sortedInd] = sort(dist, 'ascend');
    result(i) = mode(train_label(sortedInd(1:k)));
end

labels = result;

end