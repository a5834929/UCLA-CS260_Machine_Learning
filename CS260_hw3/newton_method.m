function [ w, entropy ] = newton_method( train_data, train_label, init_w, init_b, lambda )

train_data(:,end+1)=1;
entropy = zeros(1,50);
w = [init_w;init_b];
for iter=1:50
    H = hessian(train_data, w, lambda);

    sig = sigmoid(train_data*w);
    entropy(iter) = cross_entropy(train_label, sig, w, lambda);
    
    tmp = sig - train_label;
    tmp = repmat(tmp,1,size(w,1));
    grad = tmp.*train_data;
    w = w - pinv(H)*(sum(grad,1)'+2*lambda*w);
    
end

end