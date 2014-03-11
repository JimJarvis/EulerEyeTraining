function g = sigmoidGradient(z)

% g = sigmoid(z) .* (1 - sigmoid(z));
ez = exp(z);
g = ez ./ (1 + ez).^2;

end
