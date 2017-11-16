close all
clear all

% one hot encoding
[data, labels] = one_hot_encoder('car_train.data');
[valid_data, valid_label] = one_hot_encoder('car_valid.data');
[test_data, test_label] = one_hot_encoder('car_test.data');

% decision tree
% splitCriterion = cell(1,2);
% splitCriterion{1} = 'gdi';
% splitCriterion{2} = 'deviance';
% fprintf('Decision Tree:\n');
% for i=1:2
%     for minLeaves=1:10
%         [valid_accu, ~] = decision_tree(data, labels, valid_data,...
%                                    valid_label, splitCriterion{i}, minLeaves);
%         [test_accu, train_accu] = decision_tree(data, labels, test_data,...
%                                    test_label, splitCriterion{i}, minLeaves);
%         fprintf('%s, %d\n', splitCriterion{i}, minLeaves);
%         fprintf('\ttrain_accu: %f%%\tvalid_accu: %f%%\t\ttest_accu: %f%%\n',...
%                 train_accu*100, valid_accu*100, test_accu*100);
%     end
% end
% 
% % naive bayes
% [valid_accu, ~] = naive_bayes(data, labels, valid_data, valid_label);
% [test_accu, train_accu] = naive_bayes(data, labels, test_data, test_label);
% fprintf('Bernoulli Naive Bayes:\n');
% fprintf('\ttrain_accu: %f%%\tvalid_accu: %f%%\t\ttest_accu: %f%%\n',...
%                 train_accu*100, valid_accu*100, test_accu*100);

% logistic regression
[valid_accu, ~] = logistic_regression(data, labels, valid_data, valid_label);
[test_accu, train_accu] = logistic_regression(data, labels, test_data, test_label);
fprintf('Logistic Regression:\n');
fprintf('\ttrain_accu: %f%%\tvalid_accu: %f%%\t\ttest_accu: %f%%\n',...
                train_accu*100, valid_accu*100, test_accu*100);
