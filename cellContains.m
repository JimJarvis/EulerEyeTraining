%% Whether the cell array contains the strings
% if the str is another cell array of strings, return 1 only if all 
% elements in str is contained in cellarr
function bool = cellContains(cellarr, str)
    found = @(str)~isempty(find(ismember(cellarr, str))); 
    if iscell(str)
        if numel(str) > numel(cellarr) bool = false; return; end
        for i = 1:numel(str)
            if ~found(str{i})
                bool = false;
                return;
            end
        end
        bool = true;
    else
        bool = found(str);
    end
end
