%% List the sample sizes of each symbols in the training base
%

sampleSizes = zeros(CodeTable.size(1), 2);
i = 1;
for cod = TrIndex.keys
    cod = cod{1};
    sampleSizes(i, 1) = cod;
    sampleSizes(i, 2) = numel(TrIndex(cod));
    i = i + 1;
end

fid = fopen('CDB/SampleSizes.txt', 'w');
% sortedSizes: number of training samples
% sortedCodes: corresponding symbol codes
sampleSizes = sortChain(sampleSizes, 2);
sortedCodes = sampleSizes(:, 1);
sortedSizes = sampleSizes(:, 2);
for cod = 1:CodeTable.size(1)
    entry = CodeTable(sortedCodes(cod));
    fprintf(fid, '%4d = %s:%s\r\n', sortedSizes(cod), entry{:});
end
fclose(fid);
