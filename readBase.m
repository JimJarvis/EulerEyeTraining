%% Persistent databases
TOTAL = 184462;
TrBase = cell(TOTAL, 1);
TrSizes = zeros(TOTAL, 2);
TrCodes = zeros(TOTAL, 1);
% Map Code => Symbol, Category
CodeTable = containers.Map('KeyType','uint32', 'ValueType','any');
% Number of training examples for each symbol
% Code => indices of the training samples with this code ()
TrIndex = containers.Map('KeyType','uint32', 'ValueType','any');

%% Reading images from training database
baseId = fopen('CDB/OcrBase.txt');
paramId = fopen('CDB/OcrParams.txt');
for i = 1:TOTAL
    bline = fgets(baseId);
    pline = fgets(paramId);
    
    TrBase{i} = logical(strread(bline, '%d', 'delimiter',','));
    [h, w, TrCodes(i)] = strread(pline, '%d %d %d', 'delimiter', ',');
    % Must change h and w because reshape works through columns
    TrSizes(i, :) = [h w];
    TrBase{i} = reshape(TrBase{i}, [w h])';

    if mod(i, 1000) == 0
        fprintf('Progress: %6d\n', i)
    end
end
fprintf('COMPLETE!!! Total should be 184462: %d\n', i);

fclose(baseId);
fclose(paramId);

%% CodeTable
ocrCodeId = fopen('CDB/OcrCodeTable.txt');
cline = fgets(ocrCodeId);
i = 1;
while ischar(cline)
    entry = strsplit(strtrim(cline), ',');
    CodeTable(str2num(entry{1})) = {entry{2:3}};
    cline = fgets(ocrCodeId);
    i = i + 1;
end
fclose(ocrCodeId);

%% Training images index
indices = 1:numel(TrCodes);
for cod = CodeTable.keys
    cod = cod{1};
    TrIndex(cod) = indices(TrCodes == cod);
end
