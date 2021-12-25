function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

xcolumns = size(X)(1,2); 

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    s = zeros(xcolumns, 1);

    for i = 1 : m,
        h = 0;
        for c = 1 : xcolumns,
            h = h + theta(c, 1) * X(i, c);
        end;

        drv = h - y(i, 1);
        for si = 1 : xcolumns,
            s(si, 1) = s(si, 1) + drv * X(i, si);
        end;
    end;

    tmp = zeros(xcolumns, 1);
    for t = 1 : length(theta),
        tmp(t, 1) = theta(t, 1) - alpha * s(t, 1) / m;
    end;

    for t = 1 : length(theta),
        theta(t, 1) = tmp(t, 1);
    end;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
