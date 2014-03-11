%% Split the base into training and testing samples
% set how much percentage to be training (VS testing), default 80%
% thresh = this symbol must have at least this many training
% samples to be considered.
% set_train/Test contains the indices of the samples
% set_codes are all possible code tags
function [set_train set_test set_codes] = splitBase(percent, thresh)

if ~exist('percent', 'var') || isempty(percent)
    percent = 0.8;
end
if ~exist('thresh', 'var') || isempty(thresh)
    thresh = 60;
end


TrBaseRz = evalin('base', 'TrBaseRz');
TrIndex = evalin('base', 'TrIndex');

set_train = [];
set_test = [];
set_codes = [];

for cod = TrIndex.keys
    cod = cod{1};
    indices = TrIndex(cod);
    n = numel(indices);
    % We skip the ones with too few samples or class 'Calligraphic'
    if n < thresh || (500 <= cod && cod < 600) continue; end
    set_codes = [set_codes cod];
    split = ceil(n * percent);
    indices = indices(randperm(n));
    set_train = [set_train indices(1:split)];
    set_test = [set_test indices(split+1:end)];
end
set_train = randpermA(set_train);
set_test = randpermA(set_test);

end
