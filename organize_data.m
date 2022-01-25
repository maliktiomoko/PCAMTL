function [X,X_test_target,y_test,M,ns,nst,ys,yt]=organize_data(data_source,data_target,label_source,label_target,m,ns,nst,k,data)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here
Xs=[];ys=[];
p=size(data_source{1},2);
for task=1:k-1
    Xs{task}=[];ys{task}=[];
for j=1:m
    X_tot{j+m*(task-1)}=data_source{task}(label_source{task}==j,:)';
    if strcmp(data,'all')
        ns(j+m*(task-1))=size(X_tot{j+m*(task-1)},2);
    end
    X1{j+m*(task-1)}=datasample(X_tot{j+m*(task-1)},ns(j+m*(task-1)),2);
    X_mean{j+m*(task-1)}=X1{j+m*(task-1)};
    Xs{task}=[Xs{task} X1{j+m*(task-1)}];
    ys{task}=[ys{task};j*ones(ns(j+m*(task-1)),1)];
end
end
Xt=[];X_test_target=[];yt=[];y_test=[];
for j=1:m
    X_tot{j+m*(k-1)}=data_target(label_target==j,:)';
    if strcmp(data,'all')
        ns(j+m*(k-1))=floor(size(X_tot{j+m*(k-1)},2)/4);
        nst(j)=size(X_tot{j+m*(k-1)},2)-ns(j+m*(k-1));
    end
    [X2{j},ind]=datasample(X_tot{j+m*(k-1)},ns(j+m*(k-1)),2);
    X_int=X_tot{j+m*(k-1)};
    X_int(:,ind)=[];
    X_mean{j+m*(k-1)}=X2{j};
    X_test_target=[X_test_target datasample(X_int,nst(j),2)];
    Xt=[Xt X2{j}];
    yt=[yt;j*ones(ns(j+m*(k-1)),1)];
    y_test=[y_test;j*ones(nst(j),1)];
end
for task=1:k-1
    n1{task}=size(Xs{task},2);
end
    n2=size(Xt,2);
[M]=compute_statistic(X_mean);
Moy1=[];moy2=zeros(p,1);
for task=1:k-1
    moy1=zeros(p,1);
    for j=1:m
        moy1=moy1+(ns(j+m*(task-1))./sum(ns(1+m*(task-1):m*task)))*M(:,j+m*(task-1));
    end
    Moy1=[Moy1 moy1*ones(1,m)];
end
for j=1:m
    moy2=moy2+(ns(j+m*(k-1))./sum(ns(m*(k-1)+1:end)))*M(:,j+m*(k-1));
end
M=M-[Moy1 moy2*ones(1,m)];
X_test_target=X_test_target-moy2;
X=[];
for task=1:k-1
    X=[X Xs{task}*(eye(n1{task})-(1/n1{task})*ones(n1{task}))];
end
X=[X Xt*(eye(n2)-(1/n2)*ones(n2))]/sqrt(k*p);
end

