%% Randomly permutate an array
function perm = randpermA(arr)
    perm = arr(randperm(numel(arr)));
end
