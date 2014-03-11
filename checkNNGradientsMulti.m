function checkNNGradientsMulti(lambda)
% CHECKNNGRADIENTS Creates a small multilayered neural network to check the
% backpropagation gradients
%   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
%   backpropagation gradients, it will output the analytical gradients
%   produced by your backprop code and the numerical gradients (computed
%   using computeNumericalGradient). These two gradient computations should
%   result in very similar values.
%

if ~exist('lambda', 'var') || isempty(lambda)
    lambda = 0;
end

Layer_sizes = [3 5 7 9 11]; % include input_layer and output_layer
Layer_sizes = randi(7, [1 12]) + 2; % random layers
m = 30;

layers = numel(Layer_sizes);
% We generate some 'random' test data
%
Theta = cell(layers-1, 1);
for i = 1:layers-1
    Theta{i} = debugInitializeWeights(Layer_sizes(i+1), Layer_sizes(i));
end

% Reusing debugInitializeWeights to generate X
X  = debugInitializeWeights(m, Layer_sizes(1) - 1); % input_layer - 1
label_size = Layer_sizes(end);
y  = 1 + mod(1:m, label_size)';
% unrolling y into matrix of row vectors of 1 and 0
ytmp = eye(label_size);
y = ytmp(y, :);

% Unroll parameters
nn_params = [];
for i = 1:numel(Theta)
    nn_params = [nn_params; Theta{i}(:)];
end

% Short hand for cost function
costFunc = @(p) nnCostFunctionMulti(p, Layer_sizes, X, y, lambda);

[cost, grad] = costFunc(nn_params);

disp('Back prop complete. Start numerical gradient ...');
numgrad = computeNumericalGradient(costFunc, nn_params);

% Visually examine the two gradient computations.  The two columns
% you get should be very similar. 
disp([numgrad grad]);
fprintf('(Left- Numerical Gradient, Right- Backprop Gradient)\n\n');

% Evaluate the norm of the difference between two solutions.  
% If you have a correct implementation, and assuming you used EPSILON = 0.0001 
% in computeNumericalGradient.m, then diff below should be less than 1e-9
diff = norm(numgrad-grad)/norm(numgrad+grad);

if diff < 1e-9
    fprintf( 'Relative difference = %g\tCORRECT\n', diff);
else
    fprintf( 'Relative difference = %g < 1e-9? ERROR\n', diff);
end

end
