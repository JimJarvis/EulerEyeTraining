%% Sorting with satellite data
% sort w.r.t one column and all the other reorders with that sorted column
% 'a' = 'ascending', 'd' = 'descending'. Default: descending
function sorted = sortChain(mat, col, direction)
    if ~exist('direction', 'var') || isempty(direction)
        direction = 'd';
    end
    if direction == 'a' direction = 'ascend'; else direction = 'descend'; end
    [sorted idx] = sort(mat(:, col), direction);
    sorted = mat(idx, :);
end
