%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Homework II                                                       %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc;
close all;
clear all;

%Load Given Data
load spamData;

%Max and mean value of the average length of uninterrrupted sequences of
%capital letters.
A = Xtrain(:,55);
A_mean = mean(A);
A_max = max(A);

%Max and mean value of the longest length of uninterrrupted sequences of
%capital letters.
B=Xtrain(:,56);
B_mean = mean(B);
B_max = max(B);
disp('*******************************************************************************************')
fprintf('Mean of the average length of uninterrupted sequences of capital letters %f\n',A_mean);
fprintf('Max of the average length of uninterrupted sequences of capital letters %f\n',A_max);
disp('*******************************************************************************************')
fprintf('Mean of the longest length of uninterrupted sequences of capital letters %f\n',B_mean);
fprintf('Max of the longest length of uninterrupted sequences of capital letters %f\n',B_max);
%% Logistic Regression using First Preprocessing Technique.

%preprocess using first technique to the train data
[ z ] = preprocessing(Xtrain,1);

%train labels
data = ytrain; 

%calculating Percentage for Cross Validation.
y=size(data)/100*80;

%80 perecentage of Trainfeatures
X=z(1:y(1),:);

%80 percent of  trainlabels
y1=data(1:2452,1);


%20 percent of trainfeatures
X1=z(2453:end,:);

%Adding a column of ones to the Features
X=[ones(size(X,1),1) X];

%Initializing w with Zeros.
w = zeros(size(X, 2), 1);

i=1;
X1=[ones(size(X1,1),1) X1];%Adding a column of ones to the Features
lambda1=[0.5 0.05 0.005 0.0005 0.00005];
for lambda=[0.5 0.05 0.005 0.0005 0.00005];
    
    
    
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    
    [W, cost] = fminunc(@(w)costFunction_regu(X,y1,w,lambda), w, options);%Optimizing the cost Function
    
    
    y2=data(2453:end,1);%20 percent of train test labels
    [Y1]=predict(W,X1);%Predicting the matcheddata to find accuracy
    accuracy1(i)=mean(double(Y1' == y2)) * 100
    
    
    [Y2]=predict(W,X);
    Accuracy1(i)=mean(double(Y2' == y1)) * 100
    disp('****************************************************************')
    disp('Train Results using First preprocessing technique' )
    disp('****************************************************************')
    %Displaying both the accuracy
    fprintf('Train Accuracy: %f\n', mean(double(Y2' == y1)) * 100);
    fprintf('Validation Accuracy: %f\n', mean(double(Y1' == y2)) * 100);
    
    i=i+1;
    
    
end
%Plot lambda values against accuracy
figure;
plot(lambda1,accuracy1);
hold on
plot(lambda1, Accuracy1);
title('Logistic regression using First Preprocessing ');
xlabel('Lambda values');
ylabel('Accuracy');
legend('Test data', 'Training data');


%% Testing using First Preprocessing Technique

[ z ] = preprocessing(Xtrain,1);%preprocessing the data

data = ytrain;%test the train label

y=size(data)/100*80;

X=z;%new Train features
y1=data;% Train labels

%preprocessing test with test data set
[ Test] = preprocessing(Xtest,1);
X1=Test;

X=[ones(size(X,1),1) X]; %Adding ones to the features
w = zeros(size(X, 2), 1); %intializing w with zeros


i=1;
X1=[ones(size(X1,1),1) X1];%Adding ones to the  Test features

for lambda=0.005
    
    
    
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    
    [W, cost] = fminunc(@(w)costFunction_regu(X,y1,w,lambda), w, options);
    
    
    y2=(ytest);
    
    [Y1]=predict(W,X1);
    accuracy1(i)=mean(double(Y1' == y2)) * 100;
    [Y2]=predict(W,X);
    
    Accuracy1(i)=mean(double(Y2' == y1)) * 100;
    disp('****************************************************************')
    disp('Test Results using First preprocessing technique' )
    disp('****************************************************************')
   %Displaying both the accuracy
    fprintf('Test Accuracy using first preprocessing technique: %f\n', mean(double(Y1' == y2)) * 100);
    fprintf('Train Accuracy using first preprocessing technique: %f\n', mean(double(Y2' == y1)) * 100);
    i=i+1;
    
end
%% Logistic Regression using Second Preprocessing Technique.

[ z ] = preprocessing(Xtrain,2);%Calling the preprocessing function to preprocess the train data
% for j=1:3
data = ytrain; %data is train labels initialized

y=size(data)/100*80;%calculating Percentage for Cross Validation,

X=z(1:y(1),:);%80 perecentage of Trainfeatures

y1=data(1:2452,1);%80 percent of  trainlabels

X1=z(2453:end,:);%20 percent of train features


X=[ones(size(X,1),1) X];%Adding a column of ones to the Features
w = zeros(size(X, 2), 1);%Initializing w with Zeros.


i=1;
X1=[ones(size(X1,1),1) X1];%Adding a column of ones to the Features
lambda1=[0.5 0.05 0.005 0.0005 0.00005];
for lambda=[0.5 0.05 0.005 0.0005 0.00005];
    

    
    
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    
    [W, cost] = fminunc(@(w)costFunction_regu(X,y1,w,lambda), w, options);%Optimizing the cost Function
    
    
    y2=data(2453:end,1);%20 percent of train test labels
    [Y1]=predict(W,X1);%Predicting the matcheddata to find accuracy
    accuracy2(i)=mean(double(Y1' == y2)) * 100
    
    
    [Y2]=predict(W,X);
    Accuracy2(i)=mean(double(Y2' == y1)) * 100
    disp('****************************************************************')
    disp('Train Results using Second preprocessing technique' )
    disp('****************************************************************')
    %Displaying both the accuracy
    fprintf('Train Accuracy using second preprocessing technique: %f\n', mean(double(Y2' == y1)) * 100);
    fprintf('Validation Accuracy using second preprocessing technique: %f\n', mean(double(Y1' == y2)) * 100);
    
    i=i+1;
    
    
end
figure;
%Plot lambda values against accuracy
plot(lambda1,accuracy2);
hold on
plot(lambda1, Accuracy2);
title('Logistic regression using Second Preprocessing');
xlabel('Lambda values');
ylabel('Accuracy');
legend('Test data', 'Training data');



%% Testing using Second Preprocessing Technique

[ z ] = preprocessing(Xtrain,2);%preprocessing the data


data = ytrain;%test the train label

y=size(data)/100*80;

X=z;% Train features
y1=data;% Train labels

%preprocessing test with test data set
[ Test] = preprocessing(Xtest,2);
X1=Test;

X=[ones(size(X,1),1) X]; %Adding ones to the features
w = zeros(size(X, 2), 1); %intializing w with zeros


i=1;
X1=[ones(size(X1,1),1) X1];%Adding ones to the  Test features


for lambda=0.005
    
    
    
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    
    [W, cost] = fminunc(@(w)costFunction_regu(X,y1,w,lambda), w, options);%optimizing Cost Function.
    
    %Calculating error for both test and train data
    y2=(ytest);
    
    [Y1]=predict(W,X1);
    accuracy2(i)=mean(double(Y1' == y2)) * 100%Predicting the matcheddata to find accuracy
    
    [Y2]=predict(W,X);
    Accuracy2(i)=mean(double(Y2' == y1)) * 100
    disp('****************************************************************')
    disp('Test Results using Second preprocessing technique' )
    disp('****************************************************************')
    %Displaying both the accuracy
    fprintf('Test Accuracy using second preprocessing technique: %f\n', mean(double(Y1' == y2)) * 100);
    fprintf('Train Accuracy using second preprocessing technique: %f\n', mean(double(Y2' == y1)) * 100);
    i=i+1;
    
end

%% Logistic Regression using Third Preprocessing Technique.

[ z ] = preprocessing(Xtrain,3);%preprocess the train data

data = ytrain; %train labels 

y=size(data)/100*80;%calculating Percentage for Cross Validation,

X=z(1:y(1),:);%80 perecentage of Trainfeatures

y1=data(1:2452,1);%80 percent of  trainlabels

X1=z(2453:end,:);%20 percent of train features


X=[ones(size(X,1),1) X];%Adding a column of ones to the Features
w = zeros(size(X, 2), 1);%Initializing w with Zeros.


i=1;
X1=[ones(size(X1,1),1) X1];%Adding a column of ones to the Features
lambda1=[0.5 0.05 0.005 0.0005 0.00005];
for lambda=[0.5 0.05 0.005 0.0005 0.00005];
    
    
    
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    
    [W, cost] = fminunc(@(w)costFunction_regu(X,y1,w,lambda), w, options);%Optimizing the cost Function
    
    
    y2=data(2453:end,1);%20 percent of train test labels
    [Y1]=predict(W,X1);%Predicting the matcheddata to find accuracy
    accuracy3(i)=mean(double(Y1' == y2)) * 100
    
    
    [Y2]=predict(W,X);
    Accuracy3(i)=mean(double(Y2' == y1)) * 100
    disp('****************************************************************')
    disp('Train Results using Third preprocessing technique' )
    disp('****************************************************************')
    %Displaying both the accuracy
    fprintf('Train Accuracy using third preprocessing technique: %f\n', mean(double(Y2' == y1)) * 100);
    fprintf('Validation Accuracy using third preprocessing technique: %f\n', mean(double(Y1' == y2)) * 100);
    
    i=i+1;
    
    
end
%Plot lambda values against accuracy
figure;
plot(lambda1,accuracy3);
hold on
plot(lambda1, Accuracy3);
title('Logistic regression using Third Preprocessing');
xlabel('Lambda values');
ylabel('Accuracy');
legend('Test data', 'Training data');

%% Testing using Third Preprocessing Technique

[ z ] = preprocessing(Xtrain,3);%preprocessing the data
% for j=1:3
data = ytrain;%test the train label

y=size(data)/100*80;

X=z;%new Train features
y1=data;% Train labels

%preprocessing test with test data set
[ Test] = preprocessing(Xtest,3);
X1=Test;

X=[ones(size(X,1),1) X]; %Adding ones to the features
w = zeros(size(X, 2), 1); %intializing w with zeros


i=1;
X1=[ones(size(X1,1),1) X1];%Adding ones to the  Test features

for lambda=0.005
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    
    [W, cost] = fminunc(@(w)costFunction_regu(X,y1,w,lambda), w, options);
    
    
    y2=(ytest);
    %
    [Y1]=predict(W,X1);
    accuracy3(i)=mean(double(Y1' == y2)) * 100%Predicting the matcheddata to find accuracy
    
    [Y2]=predict(W,X);
    Accuracy3(i)=mean(double(Y2' == y1)) * 100
    %Displaying both train and test accuracy
    disp('****************************************************************')
    disp('Test Results using Third preprocessing technique' )
    disp('****************************************************************')
    fprintf('Test Accuracy using third preprocessing technique: %f\n', mean(double(Y1' == y2)) * 100);
    fprintf('Train Accuracy using third preprocessing technique: %f\n', mean(double(Y2' == y1)) * 100);
    i=i+1;
    
end
