%% Resize all binary images in the training base to a certain dimension
% Here's our reshaping scheme:
% We preserve the original width/height ratio and resize to our prescribed dimension as large as possible, with the rest padded with white space.

RESIZE_DIM = [33 25];

%{
% exporting variables to the workspace
assignin('base', 'RESIZE_DIM', RESIZE_DIM);
% Loading variables from the workspace
TrBase = evalin('base', 'TrBase');
%}

H = RESIZE_DIM(1);
W = RESIZE_DIM(2);
flatdim = prod(RESIZE_DIM);
TrBaseRz = zeros(TOTAL, flatdim);
for i = 1:TOTAL
    img = TrBase{i};
    h = size(img, 1);
    w = size(img, 2);

    % First resize h to H
    new_h = H;
    new_w = ceil(w * H/h);
    rz_flag = 'w';
    % Then check if the new width exceeds the designated width
    if new_w > W
        new_w = W;
        new_h = ceil(h * W/w);
        rz_flag = 'h';
    end
    img = imresize(img, [new_h, new_w]);
    img = imgPad(img, [H W], rz_flag);

    TrBaseRz(i, :) = ...
        reshape(img, [1 flatdim]);

    if mod(i, 1000) == 0
        fprintf('Progress: %6d\n', i)
    end
end

TrBaseRz = logical(TrBaseRz);
fprintf('COMPLETE!!!\n');
