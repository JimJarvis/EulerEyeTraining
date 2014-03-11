%% Analyze the confusion matrix
function [] = errorAnalyzer(pred, y)

% generate report
fid = fopen('Confusion.txt', 'w');

confTable = containers.Map('KeyType', 'uint32', 'ValueType', 'any');
CodeTable = evalin('base', 'CodeTable');

% Accuracy can be expressed in one line:
% mean(double(pred == y)) * 100
n = numel(y);
acc = 0;
% Records the adjusted accuracy that assumes x==X or 1==I are correct
acc_conf = 0;

for i = 1:n
    predc = pred(i);
    yc = y(i);
    if predc == yc  
        acc = acc + 1;
        acc_conf = acc_conf + 1;
        continue
    end

    % We consider the adjusted accuracy
    sym = CodeTable(predc); predsym = sym{1}; predsym_ = lower(sym{1});
    sym = CodeTable(yc); ysym = sym{1}; ysym_ = lower(sym{1});
    symcells = {predsym, ysym};
    if ~isempty(findstr(predsym_, 'cosvwxz')) && strcmp(predsym_, ysym_) ...
            || cellContains({'one', 'l'}, symcells) ...
            || cellContains({'zero', 'o'}, symcells) ...
            || cellContains({'zero', 'O'}, symcells) ...
            || cellContains({'I', 'l'}, symcells)
        acc_conf = acc_conf + 1;
    end

    % We record the times each error pair occurs in the confusion matrix
    % The first one is the correct symbol, the second is our prediction
    hash = encode(yc, predc);
    if ~confTable.isKey(hash)
        confTable(hash) = 1;
    else
        confTable(hash) = confTable(hash) + 1;
    end
end

% Sort and create the confusion matrix
confMat = zeros(confTable.size(1), 2);
i = 1;
for hash = confTable.keys
    confMat(i, 1) = hash{1};
    confMat(i, 2) = confTable(hash{1});
    i = i + 1;
end
confMat = sortChain(confMat, 2);

% Display the confused pairs
for entry = confMat'
    [yc predc] = decode(entry(1));
    sym_y = CodeTable(yc);
    sym_pred = CodeTable(predc);
    fprintf(fid, '%s %s == %s %s: %d\n', sym_y{:}, sym_pred{:}, entry(2));
end

fprintf(fid, '\nRaw Accuracy: %d/%d == %f\n', acc, n, acc/n);
fprintf('\nRaw Accuracy: %d/%d == %f\n', acc, n, acc/n);
fprintf(fid, '\nAdjusted Accuracy: %d/%d == %f\n', acc_conf, n, acc_conf/n);
fprintf('\nAdjusted Accuracy: %d/%d == %f\n', acc_conf, n, acc_conf/n);

fclose(fid);

end

function hash = encode(c1, c2)
    hash = c1 * 1000 + c2;
end
function [c1 c2] = decode(hash)
    c1 = floor(hash/1000);
    c2 = hash - c1 * 1000;
end
