function [C, sigma,sorted_results] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_all = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_all = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_all = eye(size(C_all,2)*size(sigma_all,2),3);
row = 0;

for i=1:size(C_all,2)
    for j=1:size(sigma_all,2)
        model= svmTrain(X, y, C_all(i), @(x1, x2) gaussianKernel(x1, x2, sigma_all(j)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        row = row+1;
        
        error_all(row,:) = [C_all(i) sigma_all(j) error];
    end
end


sorted_results = sortrows(error_all, 3); % sort matrix by column #3, the error, ascending

C = sorted_results(1,1);
sigma = sorted_results(1,2);



% =========================================================================

end
