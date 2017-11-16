function H = hessian( train_data, w, lambda )

tmp = sigmoid(train_data*w).*(1-sigmoid(train_data*w));
tmp = diag(tmp);
H = train_data'*tmp*train_data;
reg = eye(length(w))*2*lambda;
reg(end)=0;
H = H+reg;

end

