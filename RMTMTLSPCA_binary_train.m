function [error,error_opt,error_opt_th,score_test,score_test_opt,error_st] = RMTMTLSPCA_binary_train(X,X_test_target,y_true,MM,ns,k,m)
% RMTMTLSPCA_binary_train train the Random Matrix Improved Supervised PCA
% for binary classification
%   Input: Data X (size p*n) containing all the training data
%   (Source+Target), X_test_target (size p*n_test) containing the test
%   data,y_true (n_test*1) containing the test ground truth for error
%   computation, MM=M'*M (2k*2k) scalar product of means of different tasks
%   and classes,ns (2k*1) containing the number of samples per task and per
%   class, number of tasks k and number of class m,
%   Output: Classification error for Naive MTL error, Classification error
%   for MTL SPCA Optimized error_opt, Classification error for single task
%   error_st, test score for optimized and non optimized score_test and
%   score_test_opt
p=size(X_test_target,1);
n=sum(ns);
co=k*p/n;
e3=zeros(m*k,1);e3(m*(k-1)+1)=1;
e4=zeros(m*k,1);e4(m*k)=1;
c=ns./sum(ns);
% Optimal Label
tilde_y=(diag(c)*MM*diag(c)/co+diag(c))\((e3-e4)'*MM*diag(c))';
J=zeros(n,m*k);
J(1:ns(1),1)=ones(ns(1),1);
for d=1:m*k
    J(sum(ns(1:d-1))+1:sum(ns(1:d)),d)=ones(ns(d),1);
end
% Eigenvector extraction of MTL with optimized label
 [A1_i,V1_i] = eigs(X*J*(tilde_y*tilde_y')*J'*X'/n,1);[lj_i,ind1_i]=sort(diag(V1_i),'descend');A1_i=A1_i(:,ind1_i);A1_i=A1_i(:,1);
% Eigenvector extraction of MTL with classical label
 [A1_no,V1_no] = eigs(X*J*J'*X',1);[lj_no,ind1_no]=sort(diag(V1_no),'descend');A1_no=A1_no(:,ind1_no);A1_no=A1_no(:,1);
 % Eigenvector extraction of SPCA single task
 y_st=[zeros(m*(k-1),1);1;-1];
 [A1_no_st,V1_no_st] = eigs(X*J*y_st*y_st'*J'*X',1);[lj_no_st,ind1_no_st]=sort(diag(V1_no_st),'descend');A1_no_st=A1_no_st(:,ind1_no_st);A1_no_st=A1_no_st(:,1);

%%%%%%% Theoretical and empirical errors %%%%%%%
pred=zeros(size(X_test_target,2),1);
pred_opt=zeros(size(X_test_target,2),1);
score_test_opt=A1_i(:,1)'*X_test_target;
score_test=A1_no(:,1)'*X_test_target;
score_test_st=A1_no_st(:,1)'*X_test_target;
pred(score_test>0)=1;pred(score_test<0)=-1;
pred_st(score_test_st>0)=1;pred_st(score_test_st<0)=-1;
pred_opt(score_test_opt>0)=1;pred_opt(score_test_opt<0)=-1;
error=sum(pred~=y_true)./(length(y_true));
error_st=sum(pred_st'~=y_true)./(length(y_true));
if error>0.5
    error=1-error;
end
if error_st>0.5
    error_st=1-error_st;
end
error_opt=sum(pred_opt~=y_true)./(length(y_true));
if error_opt>0.5
    error_opt=1-error_opt;
end
error_opt_th=erfc(sqrt(e3'*MM*diag(c)*inv((diag(c)*MM*diag(c)+diag(co*c/k)))*diag(c)*MM*e3)/(sqrt(2)))/2;
end

