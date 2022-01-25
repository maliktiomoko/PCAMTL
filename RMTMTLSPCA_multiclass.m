function [error,error_opt] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M)
% RMTMTLSPCA_binary_train train the Random Matrix Improved Supervised PCA
% for multi-class classification in one-versus-all scheme.
%   Input: Data X (size p*n) containing all the training data
%   (Source+Target), X_test_target (size p*n_test) containing the test
%   data,y_test (n_test*1) containing the test ground truth for error
%   computation, M (p*2k) means of different tasks
%   and classes,ns (2k*1) containing the number of samples per task and per
%   class, number of tasks k and number of class m,
%   Output: Classification error for Naive MTL error, Classification error
%   for MTL SPCA Optimized error_opt
p=size(X,1);
n=sum(ns);
sort1=1:m;
J=zeros(n,m*k);
J(1:ns(1),1)=ones(ns(1),1);
for d=1:m*k
    J(sum(ns(1:d-1))+1:sum(ns(1:d)),d)=ones(ns(d),1);
end
for class=1:m
    %Preparing data for one-versus-all classification.
    sort2=circshift(sort1,-class+1);
    ns2(1)=ns(sort2(1));ns2(2)=sum(ns(sort2(2:m)));
    M1(:,1)=M(:,sort2(1));
    moy=zeros(p,1);
    for j=1:m-1
        moy=moy+(ns(sort2(j+1))./ns2(2))*M(:,sort2(j+1));
    end
    M1(:,2)=moy;
    for sd=1:k-1
        sort3{sd}=sort2+m*sd;
        ns2(1+2*sd)=ns(sort3{sd}(1));ns2(2+2*sd)=sum(ns(sort3{sd}(2:m)));
        M1(:,2*sd+1)=M(:,sort3{sd}(1));
        moy2=zeros(p,1);
        for j=1:m-1
            moy2=moy2+(ns(sort3{sd}(j+1))./ns2(2+2*sd))*M(:,sort3{sd}(j+1));
        end
        M1(:,2*sd+2)=moy2;
    end
    MM1=M1'*M1;
    n=sum(ns2);
    co2=k*p/n;
    e3=zeros(2*k,1);e3(2*(k-1)+1)=1;
    e4=zeros(2*k,1);e4(2*k)=1;
    c2=ns2./sum(ns2);
    %Optimal Label
    tilde_y1=(diag(c2)*MM1*diag(c2)/co2+diag(c2))\((e3-e4)'*MM1*diag(c2))';
    tilde_y=tilde_y1;
    %Alignment of scores
    er=[zeros(k-1,1);1];
    K=co2*(diag(c2)*MM1*diag(c2)/co2+diag(c2/k));
    m1=kron(er,ones(2,1));
    Delta=4*(m1'*(diag(c2)*MM1*(e3*e3')*MM1*diag(c2)-K)*tilde_y)^2-4*(tilde_y'*(diag(c2)*MM1*(e3*e3')*MM1*diag(c2)-K)*tilde_y)*(m1'*(diag(c2)*MM1*(e3*e3')*MM1*diag(c2)-K)*m1);
    asolve=(sqrt(Delta)-2*(m1'*(diag(c2)*MM1*(e3*e3')*MM1*diag(c2)-K)*tilde_y))./(2*(m1'*(diag(c2)*MM1*(e3*e3')*MM1*diag(c2)-K)*m1));
    tilde_y=tilde_y+kron(er,asolve*ones(2,1));
    tilde_y0=kron(ones(k,1),[1;-1]);
    tilde_y_ext=[];y_opt_ext=[];
    for task=1:k
        tilde_y_ext=[tilde_y_ext;circshift([tilde_y0(1+2*(task-1));tilde_y0(2+2*(task-1))*ones(m-1,1)],class-1)];
        y_opt_ext=[y_opt_ext;circshift([tilde_y(1+2*(task-1));tilde_y(2+2*(task-1))*ones(m-1,1)],class-1)];
    end
    %Score for each one versus all scheme
    score_test_opt(:,class)=(y_opt_ext'*J'*X'./norm(y_opt_ext'*J'*X'))*X_test_target;
    score_test(:,class)=(tilde_y_ext'*J'*X'./norm(tilde_y_ext'*J'*X'))*X_test_target/sqrt(k*p);
end
%Prediction and scores
 [~,predi]=max(score_test,[],2);
 [~,predi_opt]=max(score_test_opt,[],2);
error=sum(predi~=y_test)./length(y_test);
error_opt=sum(predi_opt~=y_test)./length(y_test);
end

