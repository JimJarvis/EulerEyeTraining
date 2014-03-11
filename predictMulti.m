function p = predictMulti(Theta, labels, X)
%PREDICT Predict the label of an input given a trained multilayer neural network
%   Outputs the predicted label of X given the trained weights of a neural network (Theta: cell array)
%
%   If pca_* are not given, we do not perform PCA transformation
if exist('pca_mu', 'var')
    X = applyPCA(X, pca_mu, pca_sigma, pca_U);
end

m = size(X, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

h = sigmoid([ones(m, 1) X] * Theta{1}');

for i = 2:numel(Theta)
    h = sigmoid([ones(m, 1) h] * Theta{i}');
end

[~, p] = max(h, [], 2);
p = labels(p);

% =========================================================================


end
