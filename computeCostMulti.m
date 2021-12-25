function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

s = 0;
for i = 1 : m,
    h = 0;
    for c = 1 : size(X)(1,2),
        h = h + theta(c, 1) * X(i, c);
    end;
    drv = h - y(i, 1);
    s = s + (drv ^ 2);
end;

J = s / (2 * m);


% =========================================================================

end
