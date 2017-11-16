close all
clear all

% one hot encoding
[data, labels] = one_hot_encoder('car_train.data');
[valid_data, valid_label] = one_hot_encoder('car_valid.data');
[test_data, test_label] = one_hot_encoder('car_test.data');
% train and classify
for k = 1:2:23
    fprintf('k = %d\n', k);
    [valid_accu, train_accu] = knn_classify(data, labels, valid_data, valid_label, k);
    fprintf('\ttrain_accu: %f%%\tvalid_accu: %f%%\t', train_accu*100, valid_accu*100);
    [test_accu, train_accu] = knn_classify(data, labels, test_data, test_label, k);
    fprintf('test_accu:  %f%%\n', test_accu*100);
end

% boundary.mat
load('boundary.mat');
train_data = features;
train_label = labels;
% draws plots with different k
for k=[1 5 15 20]
    [test_data, labels] = knn_classify_5d(train_data, train_label, k);
    
    figure;
    hold on;
    scatter(test_data(labels==1,1),test_data(labels==1,2),[], [0.5 0.8 1], '.');
    scatter(test_data(labels==-1,1),test_data(labels==-1,2),[], [1 0.7 0.8], '.');
    scatter(train_data(train_label==1,1),train_data(train_label==1,2),'b^', 'filled');
    scatter(train_data(train_label==-1,1),train_data(train_label==-1,2),'ro', 'filled');
    title(['k = ' num2str(k)]);
    legend('Classified 1','Classified -1', 'Class 1', 'Class -1');
end
