function [ data, labels ] = one_hot_encoder( filename )
% Data Format:
%     Features:
%         -buying:   vhigh, high, med, low (1,2,3,4)
%         -maint:    vhigh, high, med, low (5,6,7,8)
%         -doors:    2, 3, 4, 5more        (9,10,11,12)
%         -persons:  2, 4, more            (13,14,15)
%         -lug_boot: small, med, big       (16,17,18)
%         -safety:   low, med, high        (19,20,21)
%     Labels:
%         unacc, acc, good, vgood          (1,2,3,4)

keySet = {'1vhigh', '1high', '1med', '1low',...
          '2vhigh', '2high', '2med', '2low',...
          '32', '33', '34', '35more',...
          '42', '44', '4more',...
          '5small', '5med', '5big',...
          '6low', '6med', '6high',...
          '7unacc', '7acc', '7good', '7vgood'};

valueSet = [1:21 1:4];
mapObj = containers.Map(keySet,valueSet);

rawData = importdata(filename);
row = size(rawData,1);
oneHotFeatures = zeros(row, 20);
oneHotLabels = zeros(row, 1);

for i=1:row
    d = strsplit(rawData{i}, ',');
    for j=1:7
        ind = mapObj(strcat(int2str(j), d{j}));
        if j<7
            oneHotFeatures(i, ind) = 1;
        else
            oneHotLabels(i) = ind;
        end
    end
end

data = oneHotFeatures;
labels = oneHotLabels;

end

