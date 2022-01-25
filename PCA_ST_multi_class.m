function [error,error_lssvm] = PCA_ST_multi_class(X,X_test_target,y_test,m,ns)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
p=size(X,1);
n=sum(ns);
J=zeros(n,m);
J(1:ns(1),1)=ones(ns(1),1);
for d=1:m
    J(sum(ns(1:d-1))+1:sum(ns(1:d)),d)=ones(ns(d),1);
end
for class=1:m
    tilde_y0=[1;-1];
    tilde_y_ext=[circshift([tilde_y0(1);tilde_y0(2)*ones(m-1,1)],class-1)];
    score_test(:,class)=(tilde_y_ext'*J'*X'./norm(tilde_y_ext'*J'*X'))*X_test_target/sqrt(p);
    trnY=(J*tilde_y_ext);
    H=(X'*X+eye(n));
    eta = H \ ones(n,1); 
    nu = H \trnY; 

    S = ones(1,n)*eta;
    b = (S\eta')*trnY;
    alpha2 = nu - eta*b;
    score_test_lssvm(:,class)=(1/sqrt(p))*(X_test_target'*X*alpha2)+b;
end
figure
subplot(2,1,1)
for j=1:m
    hold on
    plot(score_test(:,j))
end
subplot(2,1,2)
for j=1:m
    hold on
    plot(score_test_lssvm(:,j))
end
 [~,predi]=max(score_test,[],2);
 [~,predi_lssvm]=max(score_test_lssvm,[],2);
error=sum(predi~=y_test)./length(y_test);
error_lssvm=sum(predi_lssvm~=y_test)./length(y_test);
end

