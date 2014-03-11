%% Helper: padding 1's to a specific size along a dimension
% Used in training image resizing
% dim = 'h' for padding along height, 'w' for padding along width
function padimg = imgPad(img, siz, dim)
    if dim == 'h' % along height
        h = size(img, 1);
        siz = siz(1);
        up = floor((siz - h)/2);
        down = siz - h - up;
        padimg = padarray(img, [up 0], 1, 'pre');
        padimg = padarray(padimg, [down 0], 1, 'post');
    else % along width
        w = size(img, 2);
        siz = siz(2);
        left = floor((siz - w)/2);
        right = siz - w - left;
        padimg = padarray(img, [0 left], 1, 'pre');
        padimg = padarray(padimg, [0 right], 1, 'post');
    end
end
