%% Apply PCA to the training images
% X: rows are observations while columns are feature variables
% var_thresh: the amount of variance we would like to retain
%
% X_pca = transformed training data
% U = linear transformer to the PCA space
% k = the number of components after compression
% mu, sigma = normalization parameters
%
function [X_pca, U, k, mu, sigma] = trainPCA(X, var_thresh)
    if ~exist('var_thresh', 'var')
        var_thresh = 0.98;
    end

    [X, mu, sigma] = normalizeFeature(X);
    [U, X_pca, latent] = pca(X);
    k = max(find( cumsum(latent)/sum(latent) < var_thresh ));
    U = U(:, 1:k);
    X_pca = X_pca(:, 1:k);

end
