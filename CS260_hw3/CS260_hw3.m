close all
clear all
%% settings
step = [0.001, 0.01, 0.05, 0.1, 0.5];
lambda = linspace(0.05,0.5,10);
color = 'rbgmc';
marker = 'd*>xo'; 
% ionosphere dataset
[ion_train_data, ion_train_label] = ion_preprocess('ionosphere_train.dat');
[ion_test_data, ion_test_label] = ion_preprocess('ionosphere_test.dat');
% spam/ham dataset
dict = importdata('spam/dic.dat');
[mail_train_data, mail_train_label] = mail_preprocess('train/', dict);
[mail_test_data, mail_test_label] = mail_preprocess('test/', dict);
%% Q1------------------------------------------------------------------------
wordcnt = sum(mail_train_data);
[occur, idx] = sort(wordcnt, 'descend');
fprintf('Most frequent 3 words:\n');
fprintf('{(%s: %d),(%s: %d),(%s: %d)}\n', dict{idx(1)}, occur(1),...
                                          dict{idx(2)}, occur(2),...
                                          dict{idx(3)}, occur(3));                             
for dataset=0:1
%% Q3------------------------------------------------------------------------
    figure;
    hold on;
    if dataset==0
          fprintf('Ionosphere:\n');
    else  fprintf('EmailSpam:\n');
    end
    for i=1:5
        if dataset==0
              [w, ~, entropy, ~, ~] = batch_gradient(ion_train_data, ion_train_label, step(i), 0);
        else  [w, ~, entropy, ~, ~] = batch_gradient(mail_train_data, mail_train_label, step(i), 0);
        end
        fprintf('\tL2 norm of w (step size=%f): %f\n', step(i), norm(w));
        plot(entropy, [color(i),'-',marker(i)]);
    end
    if dataset==0
          title('Cross-Entropy for each Iteration (Ionosphere)');
    else  title('Cross-Entropy for each Iteration (EmailSpam)');
    end
    legend('\eta=0.001','\eta=0.01', '\eta=0.05', '\eta=0.1', '\eta=0.5');
    xlabel('Iteration');
    ylabel('Cross-Entropy');
    hold off;
%% Q4(a)---------------------------------------------------------------------
    figure;
    hold on;
    for i=1:5
        if dataset==0
                [~, ~, entropy, ~, ~] = batch_gradient(ion_train_data, ion_train_label, step(i), lambda(2));
        else    [~, ~, entropy, ~, ~] = batch_gradient(mail_train_data, mail_train_label, step(i), lambda(2));
        end
        plot(entropy, [color(i),'-',marker(i)]);
    end
    if dataset==0
            title('Cross-Entropy for each Iteration with \lambda=0.1 (Ionosphere)');
    else    title('Cross-Entropy for each Iteration with \lambda=0.1 (EmailSpam)');
    end
    legend('\eta=0.001','\eta=0.01', '\eta=0.05', '\eta=0.1', '\eta=0.5');
    xlabel('Iteration');
    ylabel('Cross-Entropy');
    hold off;
%% Q4(b)---------------------------------------------------------------------
    if dataset==0
          fprintf('Ionosphere:\n');
    else  fprintf('EmailSpam:\n');
    end
    for i=1:10
        if dataset==0
                [w, ~, ~, nm_w, nm_b] = batch_gradient(ion_train_data, ion_train_label, step(2), lambda(i));
                if i==1
                    newton_ion_w = nm_w;
                    newton_ion_b = nm_b;
                end
        else    [w, ~, ~, nm_w, nm_b] = batch_gradient(mail_train_data, mail_train_label, step(2), lambda(i));
                if i==1
                    newton_mail_w = nm_w;
                    newton_mail_b = nm_b;
                end
        end
        fprintf('\tL2 norm of w (lambda=%f): %f\n', lambda(i), norm(w));
    end
%% Q4(c)---------------------------------------------------------------------
    training_entropy = zeros(10,1);
    test_entropy = zeros(10,1);
    for i=1:5
        for j=1:10
            if dataset==0
                    [w, b, entropy, ~, ~] = batch_gradient(ion_train_data, ion_train_label, step(i), lambda(j));
            else    [w, b, entropy, ~, ~] = batch_gradient(mail_train_data, mail_train_label, step(i), lambda(j));
            end
            training_entropy(j) = entropy(50);

            if dataset==0   sig = sigmoid(ion_test_data*w+b);
                            test_entropy(j) = cross_entropy(ion_test_label, sig, w, lambda(j));
            else            sig = sigmoid(mail_test_data*w+b);  
                            test_entropy(j) = cross_entropy(mail_test_label, sig, w, lambda(j));
            end
        end
        figure;
        if dataset==0
                title(['Cross-Entropy for each \lambda at T=50, \eta=', num2str(step(i)), ' (Ionosphere)']);
        else    title(['Cross-Entropy for each \lambda at T=50, \eta=', num2str(step(i)), ' (EmailSpam)']);
        end
        hold on;
        plot(lambda, training_entropy, '-or');
        plot(lambda, test_entropy, '-xb');
        legend('training data','test data');
        xlabel('Regularization Coefficient \lambda');
        ylabel('Cross-Entropy');
        hold off;
    end
%% Q6---------------------------------------------------------------------
    figure;
    hold on;
    if dataset==0
          fprintf('Ionosphere:\n');
          [w, entropy] = newton_method(ion_train_data, ion_train_label, newton_ion_w, newton_ion_b, 0);
          sig = sigmoid([ion_test_data ones(size(ion_test_data,1),1)]*w);
          fprintf('\tCross-Entropy of Test Data %f\n', cross_entropy(ion_test_label, sig, w, 0));
          title('Cross-Entropy for each Iteration (Ionosphere)');
    else  fprintf('EmailSpam:\n');
          [w, entropy] = newton_method(mail_train_data, mail_train_label, newton_mail_w, newton_mail_b, 0);
          sig = sigmoid([mail_test_data ones(size(mail_test_data,1),1)]*w);
          fprintf('\tCross-Entropy of Test Data %f\n', cross_entropy(mail_test_label, sig, w, 0));
          title('Cross-Entropy for each Iteration (EmailSpam)');
    end
    fprintf('\tL2 norm of w: %f\n', norm(w));
    plot(entropy, [color(1),'-',marker(1)]);
    xlabel('Iteration');
    ylabel('Cross-Entropy');
    hold off;
%% Q7---------------------------------------------------------------------
    figure;
    for i=1:5
        hold on;
        if dataset==0
              fprintf('Ionosphere:\n');
              [w, entropy] = newton_method(ion_train_data, ion_train_label, newton_ion_w, newton_ion_b, lambda(i));
              sig = sigmoid([ion_test_data ones(size(ion_test_data,1),1)]*w);
              fprintf('\tCross-Entropy of Test Data %f\n', cross_entropy(ion_test_label, sig, w, lambda(i)));
              title('Cross-Entropy for each Iteration for Ionosphere');
        else  fprintf('EmailSpam:\n');
              [w, entropy] = newton_method(mail_train_data, mail_train_label, newton_mail_w, newton_mail_b, lambda(i));
              sig = sigmoid([mail_test_data ones(size(mail_test_data,1),1)]*w);
              fprintf('\tCross-Entropy of Test Data %f\n', cross_entropy(mail_test_label, sig, w, lambda(i)));
              title('Cross-Entropy for each Iteration for EmailSpam');
        end
        fprintf('\tL2 norm of w (lambda=%f): %f\n', lambda(i), norm(w));
        plot(entropy, [color(i),'-',marker(i)]);
    end
    legend('\lambda=0.05','\lambda=0.1', '\lambda=0.15', '\lambda=0.2', '\lambda=0.25');
    xlabel('Iteration');
    ylabel('Cross-Entropy');
    hold off;
    figure;
    for i=6:10
        hold on;
        if dataset==0
              fprintf('Ionosphere:\n');
              [w, entropy] = newton_method(ion_train_data, ion_train_label, newton_ion_w, newton_ion_b, lambda(i));
              sig = sigmoid([ion_test_data ones(size(ion_test_data,1),1)]*w);
              fprintf('\tCross-Entropy of Test Data %f\n', cross_entropy(ion_test_label, sig, w, lambda(i)));
              title('Cross-Entropy for each Iteration for Ionosphere');
        else  fprintf('EmailSpam:\n');
              [w, entropy] = newton_method(mail_train_data, mail_train_label, newton_mail_w, newton_mail_b, lambda(i));
              sig = sigmoid([mail_test_data ones(size(mail_test_data,1),1)]*w);
              fprintf('\tCross-Entropy of Test Data %f\n', cross_entropy(mail_test_label, sig, w, lambda(i)));
              title('Cross-Entropy for each Iteration for EmailSpam');
        end
        fprintf('\tL2 norm of w (lambda=%f): %f\n', lambda(i), norm(w));
        plot(entropy, [color(i-5),'-',marker(i-5)]);
    end
    legend('\lambda=0.3','\lambda=0.35', '\lambda=0.4', '\lambda=0.45', '\lambda=0.5');
    xlabel('Iteration');
    ylabel('Cross-Entropy');
    hold off;
end
