% nnCostFunction implements the neural network cost function for a two layer
% neural network which performs classification
%   y needs to be an unrolled matrix where each row is a vector that has 
%   only one '1' and all other '0'
%   [J grad] = NNCOSTFUNCTON(nn_params, input_layer_size, hidden_layer_size,...
%   label_size, X, y, lambda)
%   computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   label_size, ...
                                   X, y, lambda)

% Reshape the flattened nn_params back into the parameters Theta1 and Theta2
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 label_size, (hidden_layer_size + 1));

m = size(X, 1);
         
% Forward propagation
A1 = [ones(m, 1) X]; % add 1 to training
Z2 = A1 * Theta1';
A2 = [ones(m, 1) sigmoid(Z2)]; % add 1
Z3 = A2 * Theta2'; 
A3 = sigmoid(Z3); % h(x)

%% Cost function
% WARNING: Theta1(1) and Theta2(1) are NOT regularizied!
reg = 0.5 * lambda/m * (sum(sum(Theta1(:, 2:end).^2)) + ...
    sum(sum(Theta2(:, 2:end).^2)));

J = 1/m * sum(sum(-y.*log(A3) - (1-y).*log(1-A3))) + reg;

%% Gradient
% Compute the error in the output layer
Delta3 = A3 - y;

% Back propagation
Delta2 = Delta3 * Theta2(:, 2:end) .* sigmoidGradient(Z2);

% Gradient with regularization
Theta1_grad = 1/m * (Delta2' * A1) + ...
    lambda/m * [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
Theta2_grad = 1/m * (Delta3' * A2) + ...
    lambda/m * [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

%% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

%{
Old non-vectorized implementation:

for i = 1:m
    % forward propagation
    a1 = X(i, :)';
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    % compute the error in the output layer
    delta3 = a3 - y(i, :)';
    % Back propagation
    delta2 = Theta2' * delta3 .* [1; sigmoidGradient(z2)];
    % Accumulate the gradient, skip the delta(0) part
    Theta1_grad = Theta1_grad + delta2(2:end) * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';
end
%}

end
