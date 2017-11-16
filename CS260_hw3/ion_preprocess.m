function [ data, label ] = ion_preprocess(filename)
rawdata = importdata(['hw3_data/ionosphere/' filename]);
row = size(rawdata,1);
data = zeros(row,34);
label = zeros(row ,1);
for i=1:row
    d = strsplit(rawdata{i}, ',');
    for j=1:34
        data(i,j) = str2double(d{j});
    end
    label(i) = d{end}=='b';
end

end

