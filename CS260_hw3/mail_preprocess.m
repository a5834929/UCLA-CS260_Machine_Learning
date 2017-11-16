function [ train_data, train_label ] = mail_preprocess(dataset, dict)

path = {'spam/','ham/'};
train_data = [];
train_label = [];

for i=1:2
    folder = ['hw3_data/spam/' dataset path{i}];
    fprintf('%s:\n', folder);
    files = dir(folder);
    filenum = size(files,1);
    data = zeros(filenum-2, size(dict,1));
    label = repmat(strcmp(path{i},'spam/'),filenum-2,1);
    for j=3:filenum
        file = [folder files(j).name];
        fileID = fopen(file,'r');
        content = textscan(fileID,'%s','Delimiter',' ,.?');
        content = content{1};
        for k=1:size(content,1)
            idx = find(ismember(dict,lower(content{k})));
            if size(idx,1)~=0
                data(j-2, idx) = data(j-2, idx)+1;
            end
        end
        if mod(j-2, 100)==0
            fprintf('\t%d files done\n', round((j-2)/100)*100);
        end
        fclose(fileID);
    end
    train_data = [train_data;data];
    train_label = [train_label;label];
end


end

