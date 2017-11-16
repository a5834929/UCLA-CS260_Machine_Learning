function [ w, b, entropy, nm_w, nm_b] = batch_gradient( train_data, train_label, step, lambda )

entropy = zeros(1,50);
w = zeros(size(train_data,2),1);
b = 0.1;
for iter=1:50
    sig = sigmoid(train_data*w+b);
    entropy(iter) = cross_entropy(train_label, sig, w, lambda);
    
    tmp = sig - train_label;
    b = b - step*sum(tmp);
    tmp = repmat(tmp,1,size(w,1));
    grad = tmp.*train_data;
    w = w - (step/size(train_data,1)).*(sum(grad,1)'+2*lambda*w);
    if iter==5
        nm_w = w;
        nm_b = b;
    end
end

end

