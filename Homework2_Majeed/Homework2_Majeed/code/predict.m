function [y]=predict(W,X )
q=size(X,1);

% X=[ones(size(X,1),1) X];

y = zeros(q, 1);

size(X)
size(W)

y=sigmoid(W'*X')

y=y>=0.5;

end

