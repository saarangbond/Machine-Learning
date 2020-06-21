function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    delta = zeros(2,1);
    for i = 1:m
      hypothesis = theta(1)*X(i,1) + theta(2) * X(i, 2);
      delta(1) = delta(1) + (hypothesis - y(i));
      delta(2) = delta(2) + ((hypothesis - y(i)) * X(i, 2));
      end;
    theta = theta - (alpha / m) * delta;
    % ============================================================
    % Save the cost J in every iteration    
    cost = computeCost(X, y, theta);
    J_history(iter) = cost;
    
end

end
