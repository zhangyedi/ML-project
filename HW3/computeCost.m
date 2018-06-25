function [J, grad, new_weight]=computeCost(weight,X,y,option)
% ----------------- used for GD and Newton ------------------ %
% input:
%      weight:  parameters in logistic regression weight = [b, w]
%      X:       data matrix with homogeneous form
%      y:       label, a vector
%      option:  1-->GD, 2-->Newton, 3-->BFGS, 4-->modified BFGS 
% output:
%      J:       the loss in one iteration, a value
%      grad:    the norm of gradient in one iteration, a value
%      new_weight:  parameters in logistic regression weight = [b, w]

    [m,n] = size(X);
    h = sigmoid( X * weight );
    g = X'* (h-y);
    J = - sum(y .* log(h) + (1-y) .* log(1- h) );
    D=diag(h .*(1-h),0);
    H = X'* D * X;
    if (option==1)
        gamma=1; % learning rate : gamma %
        new_weight = weight - gamma * g/m;
        grad = norm(g,2);
    elseif(option==2)
        new_weight = weight-H\g;
        grad = norm(g/m,2);
    end