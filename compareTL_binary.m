%% Script Comparing MTL-SPCA, N-SPCA, ST-SPCA, MTL-LSSVM and CDLS for binary transfer learning gaussian setting %%
clear all
clc
close all
addpath('utils/')
p=100;
beta_vec=linspace(0,1,10);
perf_lssvm=zeros(length(beta_vec),1); perf_cdls=zeros(length(beta_vec),1); perf_emp=zeros(length(beta_vec),1);
perf_emp_opt=zeros(length(beta_vec),1);perf_th=zeros(length(beta_vec),1);perf_th_opt=zeros(length(beta_vec),1);
 for ut=1:length(beta_vec)
    n_trial=1;
    error=zeros(length(n_trial),1);error_opt=zeros(length(n_trial),1);
    accuracy_CDLS_vec=zeros(length(n_trial),1);
    for rt=1:n_trial
    m=2;k=2;
    ns=[1000 1000 50 50]';
    nst=10000*ones(m*k,1);
    tot_vec=1;
    param=0;
    [S,T,X_test_target,y_test,M,Ct,X_test_source] = generate_mvr(ns,nst,p,m,k,beta_vec(ut),tot_vec,'binary',param);
    n=sum(ns);
    co=k*p/n;
    MM=M'*M;
    c=ns./sum(ns);
    e3=zeros(m*k,1);e3(m*(k-1)+1)=1;
    e4=zeros(m*k,1);e4(m*k)=1;
    tilde_y0=kron(ones(k,1),[1;-1]);
    tilde_y=(diag(c)+diag(c)*MM*diag(c/co))\(diag(c/co)*MM*(e3-e4));
    score_th_opt=(tilde_y'*diag(c)*MM/sqrt(co))./sqrt((tilde_y'*(diag(c/k)+diag(c)*MM*diag(c/co))*tilde_y));
    score_th_opt=abs(score_th_opt(3));
    score_th=(tilde_y0'*diag(c)*MM/sqrt(co))./sqrt((tilde_y0'*(diag(c/k)+diag(c)*MM*diag(c/co))*tilde_y0));
    score_th=(score_th(4));


    J=zeros(n,2*2);
    for i=1:2*2
    J(sum(ns(1:i-1))+1:sum(ns(1:i)),i)=ones(ns(i),1);
    end
    X=[S.fts{1} T.fts']./sqrt(k*p);
    [Vstar,Vapstar] = eigs(X*J*(tilde_y*tilde_y')*J'*X'/n,1);[~,ind1_i]=sort(diag(Vapstar),'descend');Vstar=Vstar(:,ind1_i);Vstar=Vstar(:,1);
    [V,Vap] = eigs(X*(J*J')*X',1);[~,ind1_no]=sort(diag(Vap),'descend');V=V(:,ind1_no);V=V(:,1);
    %%%%%%%Error théorique et empirique %%%%%%%
    pred=zeros(size(X_test_target,2),1);
    pred_opt=zeros(size(X_test_target,2),1);
    score_test_opt=Vstar(:,1)'*X_test_target;
    score_test=V(:,1)'*X_test_target;
    pred(score_test>0)=1;pred(score_test<0)=-1;
    pred_opt(score_test_opt>0)=1;pred_opt(score_test_opt<0)=-1;
    error(rt)=sum(pred~=[-ones(nst(3),1);ones(nst(3),1)])./(2*nst(3));
    if error(rt)>0.5
    error(rt)=1-error(rt);
    end
    error_opt(rt)=sum(pred_opt~=[-ones(nst(1),1);ones(nst(1),1)])./(2*nst(1));
    if error_opt(rt)>0.5
    error_opt(rt)=1-error_opt(rt);
    end
    gamma=ones(2,1);lambda=1;
    trnY(1:ns(1))=ones(ns(1),1);trnY(1+ns(1):ns(1)+ns(2))=-ones(ns(2),1);
    trnY(1+ns(1)+ns(2):ns(1)+ns(2)+ns(3))=ones(ns(3),1);
    trnY(1+ns(1)+ns(2)+ns(3):ns(1)+ns(2)+ns(3)+ns(4))=-ones(ns(4),1);
    S.fts{1}=S.fts{1}/sqrt(k*p);T.fts=T.fts./sqrt(k*p);
    [accuracy_CDLS_vec(rt)] = CDLS_method(S.fts{1}',S.labels{1}',T.fts,T.labels,X_test_target',y_test);

    end
    [~,~,~,~,~, ~,~,~,~,~,y_opt,~,~] = MTLLSSVMTrain_binary(S.fts,T.fts',trnY, gamma, lambda,M,Ct,X_test_target,ns,'task',k,nst(3:4));
    trnY_opt=J*y_opt;
    [~,~,error_th,~,~, ~,~,~,~,~,~,~,~] = MTLLSSVMTrain_binary(S.fts,T.fts',trnY_opt, gamma, lambda,M,Ct,X_test_target,ns,'task',k,nst(3:4));
    perf_lssvm(ut)=mean(error_th(3:4));
    perf_cdls(ut)=1-mean(accuracy_CDLS_vec)/100;
    perf_emp(ut)=mean(error);
    perf_emp_opt(ut)=mean(error_opt);
    perf_th(ut)=0.5*erfc(((abs(score_th(1))/sqrt(2))));
    perf_th_opt(ut)=0.5*erfc(((abs(score_th_opt(1))/sqrt(2))));
 end
 figure
 hold on
plot(beta_vec,perf_emp,'r*')
plot(beta_vec,perf_th,'r')
plot(beta_vec,perf_emp_opt,'go')
plot(beta_vec,perf_th_opt,'g')
plot(beta_vec,perf_lssvm,'k')
plot(beta_vec,perf_cdls,'c')
vec=zeros(2*length(beta_vec),1);
vec(1:2:end)=beta_vec;
vec(2:2:end)=perf_emp;
sprintf('(%d,%d)',vec)
vec2=zeros(2*length(beta_vec),1);
vec2(1:2:end)=beta_vec;
vec2(2:2:end)=perf_th;
sprintf('(%d,%d)',vec2)
vec3=zeros(2*length(beta_vec),1);
vec3(1:2:end)=beta_vec;
vec3(2:2:end)=perf_emp_opt;
sprintf('(%d,%d)',vec3)
vec4=zeros(2*length(beta_vec),1);
vec4(1:2:end)=beta_vec;
vec4(2:2:end)=perf_th_opt;
sprintf('(%d,%d)',vec4)
vec5=zeros(2*length(beta_vec),1);
vec5(1:2:end)=beta_vec;
vec5(2:2:end)=perf_lssvm;
sprintf('(%d,%d)',vec5)
vec6=zeros(2*length(beta_vec),1);
vec6(1:2:end)=beta_vec;
vec6(2:2:end)=perf_cdls;
sprintf('(%d,%d)',vec6)