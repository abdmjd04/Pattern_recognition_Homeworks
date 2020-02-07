%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%% Linear Regression with Regularization                               %%%
%%%                                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load training  data
Train_data = load('hw1training.txt');
X_train = Train_data(:,1);
y_train = Train_data(:,2);
M =10; 

% Load test data
Test_data = load('hw1test.txt');
X_test = Test_data(:,1);
y_test = Test_data(:,2); 

%% Computing linear regression for M = 10 and plot the outputs

Linear_regression(X_train,y_train,M)
Linear_regression(X_test,y_test,M)


%% Computing train and test errors for different values of M without regularization

error_train = [];
error_test = [];


for i=1:M
    
 error_train = [error_train Linear_regression(X_train,y_train,i)];
 error_test = [error_test Linear_regression(X_test,y_test,i)];

end
%% using lambda = 0.001 and plot the outputs

lambda = 0.001;
linear_regression_regular(X_train,y_train,lambda);

% Training and test error for different values of lambda with regularization

lambda = 0.001;
[Error_rms_train Norm_w_train] = linear_regression_regular(X_train,y_train,lambda);
[Error_rms_test Norm_w_test] = linear_regression_regular(X_test,y_test,lambda);

% %% plot train errors
figure(3)
%x_ax = 1:10; % for without regularization
x_ax = linspace(0,lambda,10); % for regularization
plot(x_ax,Error_rms_train,'p-',x_ax,Error_rms_test,'p-');
axis([0,lambda,0,1])
%xlabel('M');
title('To find best lambda')
xlabel('lambda')
ylabel('RMS Error');
legend('train','test');
 
% plot test errors
figure(4)
%x_ax = 1:M;
x_ax = linspace(0,lambda,10); % for regularization
plot(x_ax,Norm_w_train,'p-',x_ax,Norm_w_test,'p-');
xlabel('lambda');
ylabel('Norm');
legend('train','test');



  

