function [ w, e_in ] = logistic_reg( X, Y, max_its,w,eta,epsilon )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix
%       Y : data labels (plus or minus 1)
%	max_its : max iteration number
%	w : initial weight vector
%	eta : learning rate
%	epsilon : algorithm terminate tolerance
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)



% Note, this script assumes the w vector has not been augmented to include a
% parameter for the intercept term. It also assumes that X has not been
% augemented.


% Augment X matrix and get needed dimensions
num_examples = size(X,1);
newCol = ones(num_examples, 1);
X = [newCol X];
num_features = size(X,2);
g = zeros(num_features, 1);

% Switch non1's in y to -1's
Y(Y~=1) = -1;

% Initialize to avoid early termination
e_in = 1;

for i = 1:max_its
    
    % Terminate if the gradient descent is decreasing at a slow rate
    % and e_in is tolerably low
    if and(max(g < .001), e_in < epsilon)
        break;
    end
    
    % Set gradient to 0
    g = zeros(num_features, 1);
    
    % Sum contributions of each example to get the gradient
    for j = 1:num_examples
        g = g + transpose(Y(j)*X(j,:)/(1 + exp(Y(j)*transpose(w)*transpose(X(j,:)))));
    end
    
    % Standardize and get direction
    g = -g/num_examples;
    
    % Flip direction to get descent direction
    Vt = -g;
    
    % Reset weights
    w = w + Vt*eta;
    
    % Find the e_in produced by the new weights
    
    % Intialize to 0
    e_in = 0;
    num_examples = size(X, 1);
    
    % Add contributions of each example to error
    for j = 1:num_examples
        e_in = e_in + log(1 + exp(-Y(j)*transpose(w)*transpose(X(j,:))));
    end
    
    % Standardize
    e_in = e_in/num_examples;
    
    
end


