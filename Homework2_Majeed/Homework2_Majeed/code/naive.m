%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Naive Bayes                                                        %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
clc;
close all;
clear all;

%load data
load spamData

[Xtrain_prepro, fmeans, fsigmas] = preprocessing(Xtrain,1); %preprocess using first preprocessing technique
%[Xtrain_prepro, fmeans, fsigmas] = preprocessing(Xtrain,2);  %preprocess using Second preprocessing technique
%[Xtrain_prepro, fmeans, fsigmas] = preprocessing(Xtrain,3); %preprocess using third. preprocessing technique

%Calculating mean and variance of spam and non spam
mu_spam = mean(Xtrain_prepro(1:1218,:));
sigma_spam = var(Xtrain_prepro(1:1218,:));

%not spam
mu_nspam = mean(Xtrain_prepro(1219:3065,:));
sigma_nspam =var(Xtrain_prepro(1219:3065,:));

%%

[ Xtest_prepro] = (Xtest - repmat(fmeans, size(Xtest, 1), 1)) ./ repmat(fsigmas, size(Xtest, 1), 1);


% finding probabilities of a mail being spam and not spam in train data
% Find the indices for the spam and nonspam labels
spam = find(ytrain == 1);
nonspam = find(ytrain == 0);

% Calculate probability of spam & not spam
prob_spam = length(spam)/ length(ytrain);
prob_notspam=length(nonspam)/ length(ytrain);

%%
predict=[];
for i=1:size(Xtest,1)
    
    prob_mail_givenspam = normpdf(Xtest_prepro(i,:), mu_spam, sigma_spam); %calculating the normal distribution
    likelihood_spam = prod(prob_mail_givenspam);%naive bayes assumption
    
    posterior_spam_mail = likelihood_spam*prob_spam;
    
    prob_mail_given_nspam = normpdf(Xtest_prepro(i,:),mu_nspam,sigma_nspam);%calculating the normal distribution
    likelihood_nspam = prod(prob_mail_given_nspam); %naive bayes assumption
    
    posterior_nonspam_mail = likelihood_nspam*prob_notspam;
    
    if posterior_spam_mail > posterior_nonspam_mail
        
        predict(i) =1;
        
    else predict(i)=0;
        
    end
end
%calculating accuracy

Accuracy = sum(predict' == ytest)/length(ytest)


