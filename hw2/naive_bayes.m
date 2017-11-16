function [new_accu, train_accu] = naive_bayes(train_data, train_label, new_data, new_label)

attribute = size(train_data,2);
class = unique(train_label);
prior = zeros(size(class,1),1);
prob = zeros(size(class,1), attribute);
for c=1:size(class,1)
    observC = train_data(train_label==c,:);
    prior(c) = size(observC,1)/size(train_data,1);
    for a=1:attribute
        prob(c,a) = sum(observC(:,a))/sum(train_label==c);
    end
end
prob(prob==0)=1e-7;

% calculating training accuracy
result = zeros(size(train_label));
for i=1:size(train_data,1)
    likelihood = zeros(size(class,1),1);
    for c=1:size(class,1)
        likelihood(c) = log(prior(c));
        for a=1:attribute
            if train_data(i,a)==1
                likelihood(c) = likelihood(c) + log(prob(c,a));
            else
                likelihood(c) = likelihood(c) + log(1-prob(c,a));
            end
        end
    end
    [~, ind] = max(likelihood);
    result(i) = ind;
end
train_accu = sum(result==train_label)/size(result,1);

% calculating validation/test accuracy
result = zeros(size(new_label));
for i=1:size(new_data,1)
    likelihood = zeros(size(class,1),1);
    for c=1:size(class,1)
        likelihood(c) = log(prior(c));
        for a=1:attribute
            if new_data(i,a)==1
                likelihood(c) = likelihood(c) + log(prob(c,a));
            else
                likelihood(c) = likelihood(c) + log(1-prob(c,a));
            end
        end
    end
    [~, ind] = max(likelihood);
    result(i) = ind;
end
new_accu = sum(result==new_label)/size(result,1);

end