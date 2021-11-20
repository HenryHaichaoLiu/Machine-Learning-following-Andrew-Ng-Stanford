function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

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
test_mat = [0.01 0.03 0.1 0.3 1 3 10 30];
num_test_mat = size(test_mat,2);
accuracy = zeros(num_test_mat, num_test_mat);
for i = 1:num_test_mat
    C = test_mat(i);
    for j = 1:num_test_mat
        sigma = test_mat(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        accuracy(i,j) = mean(double(predictions ~= yval));
    end
end
Max = min(min(accuracy));
[row_max, cal_max] = find(accuracy==Max,1);
C = test_mat(row_max);
sigma = test_mat(cal_max);






% =========================================================================

end
