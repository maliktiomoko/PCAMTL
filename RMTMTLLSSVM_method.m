function [accuracy,accuracy_opt] = RMTMTLLSSVM_method(X,X_test,y_test,ns,m,k)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
addpath('/Users/tiomokomalik/Documents/RMTMTLLSSVM_matlab')
init_order=1:k*m;
gamma=[1;1];lambda=1;
nsi=reshape([zeros(1,k);reshape(ns,m,k)],1,k*m+k);
gx_t=zeros(size(X_test,2),m);gx_t_opt=zeros(size(X_test,2),m);
for i=1:m
    i
        if i~=1
                init_order(m*(1-1)+1)=m*(1-1)+i;init_order(m*(1-1)+i)=m*(1-1)+i-1;
        end
        if i~=1
                init_order(m*(k-1)+1)=m*(k-1)+i;
                init_order(m*(k-1)+i)=m*(k-1)+i-1;
        end
        nsn=ns(init_order);
        nsa=zeros(2*k,1);
        nsa(1:2:end)=ns(1:m:end);
        nsa(2:2:end)=sum(reshape(ns,m,k))'-ns(1:m:end);
        Xt2=X(:,sum(nsa(1:2*(k-1)))+1:end);
        orderm=1:sum(ns(m*(k-1)+1:k*m));
        orderm1=1+sum(nsi((m+1)*(k-1)+1:(m+1)*(k-1)+i)):sum(nsi((m+1)*(k-1)+1:(m+1)*(k-1)+1+i));
        orderm2=orderm;
        orderm2(orderm1)=[];
        order2=[orderm1';orderm2'];
        X2=Xt2(:,order2);
        Xt1=X(:,1:nsa(2*(1-1)+1)+nsa(2*(1-1)+2));
        ordert=1:sum(ns(m*(1-1)+1:m));
        ordert1=1+sum(nsi(1+m*(1-1):i+m*(1-1))):sum(nsi(1+m*(1-1):m*(1-1)+i+1));
        ordert2=ordert;
        ordert2(ordert1)=[];
        order1=[ordert1';ordert2'];
        X1p=Xt1(:,order1);
        n=sum(ns);
        J=zeros(n,m*k);
        for h=1:m*k
                J(sum(ns(1:h-1))+1:sum(ns(1:h)),h)=ones(ns(h),1);
        end
        tildey=-ones(m*k,1);tildey(1:m:end)=1;
        yc=J*tildey;
        n1=sum(nsn(1:m));
        Slab=yc(1:n1);Tlab=yc(n1+1:end);
        ne=[nsn(1);sum(nsn(2:m));nsn(m+1);sum(nsn(m+2:end))];
        nst=10*ones(4,1);
        Jk=zeros(n,2*k);
        for g=1:2*k
            Jk(sum(ne(1:g-1))+1:sum(ne(1:g)),g)=ones(ne(g),1);
        end
        [~,gx_t1,~,~,~,~,~,~,~,~,...
        ~,gx_t_opt1,~,~,~,~,~,~,~,~]=...
        RMTMTLLSSVM(X1p,Slab,X2,Tlab,lambda,gamma,X(:,1:sum(ns(1:m))),X_test,ne,nst,i);
        gx_t(:,i)=gx_t1; gx_t_opt(:,i)=gx_t_opt1;
end
 [~,pred]=max(real(gx_t),[],2);
 [~,pred_opt]=max(real(gx_t),[],2);
accuracy=1-sum(pred~=y_test)./length(y_test);
accuracy_opt=1-sum(pred_opt~=y_test)./length(y_test);
end

