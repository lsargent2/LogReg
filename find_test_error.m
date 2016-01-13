function [ test_error ] = find_test_error( w, X, Y )
%FIND_TEST_ERROR Find the test error of a linear separator
%   This function takes as inputs the weight vector representing a linear
%   separator (w), the test examples in matrix form with each row
%   representing an example (X), and the labels for the test data as a
%   column vector (y). The labels are assumed to be plus or minus one. The
%   function returns the error on the test examples as a fraction. The
%   hypothesis is assumed to be of the form (sign ( [1 x(n,:)] * w )

num_examples = size(X,1);
threshold = .5;

% If w has been augmented, but X has not, augment X
if (size(w,1) == size(X,2)+1)
newCol = ones(num_examples, 1);
X = [newCol X]; 
end

classifications = zeros(num_examples, 1);
sucesses = 0.0;

% Get number of correctly classified examples
for k = 1:num_examples
    if and( (exp(transpose(w)*transpose(X(k,:)))/(1+exp(transpose(w)*transpose(X(k,:))))) > threshold, Y(k)==1);
        sucesses = sucesses+1;
        classifications(k) = 1;
    end

%     Y can equal 0 or -1; accounts for differences in data sets
    if and( (exp(transpose(w)*transpose(X(k,:)))/(1+exp(transpose(w)*transpose(X(k,:))))) < threshold, or(Y(k)==-1,Y(k)==0));
        sucesses = sucesses+1;
        classifications(k) = -1;
    end
end

% Divide succesful classifications by total observations.
test_error = 1 - (sucesses/num_examples);

end

