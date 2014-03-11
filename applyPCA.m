%% Apply PCA compression with a coefficient matrix
% on new/test images
% Note: mu and sigma are obtained from the training
function X_pca = applyPCA(X, mu, sigma, U, k)
    if exist('k', 'var')
        U = U(:, 1:k);
    end
    X = normalizeFeature(X, mu, sigma);
    X_pca = X * U;
end
