function [Error_rms Norm_w] = linear_regression_regular(X,y,lambda)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
theta = [ones(size(X,1),1)];

% to design matrix
for i = 1:10
    
    theta = [theta X.^i];
    
end
% added regularization parameter
temp = pinv((lambda*eye(11)) + (theta'*theta));
%Parameters computed 
w = temp * theta' * y;
%output
Y = w' * theta';

%% Error and norm for different lambda values

Error_rms = [];
Norm_w = [];

for j = linspace(0,lambda,10)
    
    temp = pinv((j*eye(11)) + (theta'*theta));
    
    w = temp * theta' * y;
    
    Y = w' * theta';
    
    E = (Y - y').^2;
    E = sum(E)/2;
    Error_rms = [Error_rms sqrt(2*E/length(X))];
    Norm_w = [Norm_w norm(w)^2];
    
end

%% plot the outputs after regularization
figure(2)
scatter(X,y)
hold on;
plot(X,Y)
xlabel('Input');
ylabel('Output value');
title('After regularization');
end