% nnCostFunctionMulti implements the neural network cost function
% for a multilayered neural architecture which performs classification
%   y needs to be an unrolled matrix where each row is a vector that has 
%   only one '1' and all other '0'
%    [J grad] = NNCOSTFUNCTON(nn_params, Layer_sizes, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%   Layer_sizes should be an array of sizes for input, hidden and output layers% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
function [J grad] = nnCostFunctionMulti(nn_params, Layer_sizes, ...
                                   X, y, lambda)

% Reshape the flattened nn_params back into a cell array Theta
%
layers = numel(Layer_sizes);
Theta = cell(layers-1, 1);
% Obtain the Theta{} by unrolling the flattened nn_params
startn = 1; lengn = 0;
for i = 1:numel(Theta)
    lengn = Layer_sizes(i+1) * (Layer_sizes(i)+1);
    Theta{i} = reshape(nn_params(startn:startn+lengn-1), ...
        Layer_sizes(i+1), Layer_sizes(i)+1);
    startn = startn + lengn;
end

m = size(X, 1);
         
% Forward propagation
A = cell(layers, 1); % activations after sigmoid
Z = cell(layers, 1); % activations before sigmoid
A{1} = [ones(m, 1) X]; % add 1 to training data
% Let the first cell of Z be empty because we don't do anything to the input layer
for i = 2:layers
    Z{i} = A{i-1} * Theta{i-1}';
    if i ~= layers % not the output layer
        A{i} = [ones(m, 1) sigmoid(Z{i})];
    else
        A{i} = sigmoid(Z{i}); % output h(X)
    end
end

%% Cost function
% WARNING: Theta{i}(1) are NOT regularizied!
reg = 0;
for i = 1:numel(Theta)
    reg = reg + sum(sum(Theta{i}(:, 2:end).^2));
end
reg = 0.5 * lambda/m * reg;

J = 1/m * sum(sum(-y.*log(A{end}) - (1-y).*log(1-A{end}))) + reg;

%% Gradient
% accumulator
Delta = cell(layers, 1);
% Compute the error in the output layer
Delta{end} = A{end} - y;

% Back propagation
for i = layers-1:-1:2
    % When back propagating, always exclude the '1' unit
    Delta{i} = Delta{i+1} * Theta{i}(:, 2:end) .* sigmoidGradient(Z{i});
end

% Gradient with regularization
Theta_grad = cell(layers-1, 1);

for i = 1:layers-1
    Theta_grad{i} = 1/m * Delta{i+1}' * A{i} + ... % regularized gradient
        lambda/m * [zeros(size(Theta{i}, 1), 1) Theta{i}(:, 2:end)];
end

%% Unroll gradients
grad = [];
for i = 1:numel(Theta)
    grad = [grad; Theta_grad{i}(:)];
end

end
