%%%% EulerEye Multilayer Neural Network training (and testing)
% Jim Fan 2014
% Jarvis Initiative
%
%% Settable parameters
hidden_layer_size_array = [50 30]; % roughly 1.3 * label_size?
iter = 100;  % gradient descent iterations
lambda = 1;  % regularization coeff
pca_variance_thresh = 0.95; % how much variance we'd like to retain 
                    % when compressing training data by PCA
preprocess = 0; % set to true to perform data set splitting and PCA, 
                % otherwise directly call pca_* and set_* from workspace


%% ================== Training/Testing data =================
fprintf('\nPreparing training and testing data ...\n')

% Split the data into training and testing sets
% set_codes is the set of all possible choices/symbols
% splitBase 60: to exclude the symbols with too few training
%
if preprocess
    disp('Splitting the database into training/test.');
    [set_train set_test set_codes] = splitBase(0.85, 2000); end

label_size = numel(set_codes);          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
fprintf('Training set size: %d\n', numel(set_train));
fprintf('Testing set size: %d\n', numel(set_test));
fprintf('Possible labels: %d\n', label_size);

X_train = TrBaseRz(set_train, :);
y_train = TrCodes(set_train, :);
    % unroll y_train
    % Expand y to m-by-10 row vectors with only 0 and 1
    % then convert logicals to double 0/1
    % y = double(repmat(y, 1, label_size) == repmat(1:label_size, m, 1));
    % Better: y = eye(label_size)(y, :); //only works in Octave
y_train = double(repmat(y_train, 1, label_size) ...
            == repmat(set_codes, size(X_train, 1), 1));

% Apply PCA compression to X_train
if preprocess
    fprintf('Apply PCA to training data');
    [X_train pca_U pca_k pca_mu pca_sigma] = trainPCA(X_train, pca_variance_thresh); 
else
    X_train = applyPCA(X_train, pca_mu, pca_sigma, pca_U); end

input_layer_size = pca_k;
fprintf('Dimension reduction: %d/%d',...
    pca_k, prod(RESIZE_DIM));


X_test = TrBaseRz(set_test, :);
y_test = TrCodes(set_test, :);

% We also apply the same PCA parameters to the test set
X_test = applyPCA(X_test, pca_mu, pca_sigma, pca_U);


%% ================= Initialization ===============
fprintf('\n\nInitializing Neural Network Parameters ...\n')

Layer_sizes = [input_layer_size hidden_layer_size_array label_size];

% here Theta is a cell array that contains (number-of-hidden-layer + 1) Thetas.
Theta = cell(numel(Layer_sizes)-1, 1);

% Unroll parameters
initial_nn_params = [];
for i = 1:numel(Theta)
    thetai = randInitializeWeights(Layer_sizes(i), Layer_sizes(i+1));
    initial_nn_params = [initial_nn_params; thetai(:)];
end

fprintf('Theta parameter size = %d', numel(initial_nn_params));

%% ============= Training Neural Network ===================
%
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', iter, 'GradObj', 'on');

% curried function to be minimized: takes only 1 input
costFunction = @(p) nnCostFunctionMulti(p, Layer_sizes, ...
                        X_train, y_train, lambda);

% Advanced minimizer
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain the Theta{} by unrolling the flattened nn_params
startn = 1; lengn = 0;
for i = 1:numel(Theta)
    lengn = Layer_sizes(i+1) * (Layer_sizes(i)+1);
    Theta{i} = reshape(nn_params(startn:startn+lengn-1), ...
        Layer_sizes(i+1), Layer_sizes(i)+1);
    startn = startn + lengn;
end

%% ================= Testing =================
%
fprintf('\nTesting: saving error report to Confusion.txt\n\n');

% The X_test is already transformed by PCA. 
% Otherwise we have to pass pca_* to predict()
pred = predictMulti(Theta, set_codes, X_test);

errorAnalyzer(pred, y_test);

save EulerNN_test.mat Theta1 Theta2 set_train set_test set_codes RESIZE_DIM Layer_sizes pca_U pca_k pca_mu pca_sigma;
