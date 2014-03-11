%% Load data and run a quick error analyzer
% 1-layer NN only
%

if ~exist('CodeTable', 'var')
    load EulerBase
end
if ~exist('Theta1', 'var')
    load EulerNN
end

errorAnalyzer(predict(Theta1, Theta2, set_codes, ...
                    TrBaseRz(set_test, :), ...
                    pca_mu, pca_sigma, pca_U), ...
            TrCodes(set_test, :));

fprintf('\nConfusion.txt generated.\n');
