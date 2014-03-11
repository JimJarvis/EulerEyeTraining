%% Normalize the training/test set by mu and sigma
% If mu and sigma are not provided, we calc it from X
function [X_norm mu sigma] = normalizeFeature(X, mu_, sigma_)

if ~exist('mu_', 'var') mu = mean(X); else mu = mu_; end
if ~exist('sigma_', 'var') sigma = std(X); else sigma = sigma_; end
sigma(sigma == 0) = 1; % avoid division by zero

X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
