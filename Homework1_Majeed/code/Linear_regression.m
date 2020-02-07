function error_rms = Linear_regression(X,y,M)
% Bias 
phi = [ones(size(X,1),1)];

% Matrix
for i = 1:M
    
    phi = [phi X.^i];
    
end

temp = pinv(phi'*phi);
%Regression parameters
w = temp * phi' * y; 
%output
y_estimate = w' * phi';

%% Compute the RMS error
difference = (y_estimate - y').^2;
difference = sum(difference)/2;
error_rms = sqrt(2*difference/length(X));

%% to plot the Output
figure(1)
scatter(X,y)
hold on;
plot(X,y_estimate)
xlabel('Input ');
ylabel('output value');
title('Before applying regularization');