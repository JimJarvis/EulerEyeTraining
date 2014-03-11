%% Visualize selected images from a database
%
rnd = randperm(size(TrBase, 1));
% rnd = randperm(3000);
for i = rnd
    subplot(1,2,1);
    img = TrBase{i};
    imshow(img);
    title(sprintf('Old: %d-x-%d', size(img, 1), size(img, 2)));
    subplot(1,2,2);
    imshow(reshape(TrBaseRz(i, :), RESIZE_DIM));
    title(sprintf('New: %d-x-%d', RESIZE_DIM(1), RESIZE_DIM(2)));
    entry = CodeTable(TrCodes(i));
    fprintf('%s %s\n', entry{:});
    pause
end
