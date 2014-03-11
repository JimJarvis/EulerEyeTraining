folder = 'E:/Dropbox/Programming/Java/JarvisJava/EulerEye/EulerBase/';

dlmwrite([folder 'TestImageSizes.txt'],TrSizes(set_test, :));
dlmwrite([folder 'TestImageCodes.txt'],TrCodes(set_test, :));

j = 0;
for i=1:TOTAL
    j = j + 1;
    if mod(j, 1000) == 0 fprintf('%5d ...\n',j); end
    dlmwrite([folder 'TestImages.txt'], TrBase{i}, '-append');
end

clear folder i j
