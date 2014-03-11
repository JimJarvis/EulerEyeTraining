function p = predict(Theta1, Theta2, labels, X, pca_mu, pca_sigma, pca_U)
%PREDICT Predict the label of an input given a trained 1-layer neural network
%   Outputs the predicted label of X given the trained weights of a neural network (Theta1, Theta2)
%
%   If pca_* are not given, we do not perform PCA transformation
if exist('pca_mu', 'var')
    X = applyPCA(X, pca_mu, pca_sigma, pca_U);
end

m = size(X, 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[~, p] = max(h2, [], 2);
p = labels(p);

% =========================================================================


end
