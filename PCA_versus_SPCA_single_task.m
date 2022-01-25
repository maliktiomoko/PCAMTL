%% Script Comparing PCA versus SPCA for binary single task gaussian setting %%


p_vec=200:100:1000;
error_pca_vec=zeros(length(p_vec),1);perf_th_pca=zeros(length(p_vec),1);
error_spca_vec=zeros(length(p_vec),1);perf_th_spca=zeros(length(p_vec),1);
for lf=1:length(p_vec)
    %%%%%%%%%% Setting %%%%%%%%%%%%%%
    p=p_vec(lf);
    c=[0.5;0.5];
    n_test=1000;
    n=1000;
    means{1}=[-1;0;0;0;zeros(p-4,1)];
    means{2}=[1;0;0;0;zeros(p-4,1)];
    M=[means{1} means{2}];
    J=zeros(n,2);J_test=zeros(n_test,2);
    J(1:c(1)*n,1)=ones(c(1)*n,1);
    J(c(1)*n+1:end,2)=ones(c(2)*n,1);
    J_test(1:c(1)*n_test,1)=ones(c(1)*n_test,1);
    J_test(c(1)*n_test+1:end,2)=ones(c(2)*n_test,1);
    n_trial=10;
    param=0.0;
    Ct=toeplitz(param.^(0:p-1));
    error_pca=zeros(n_trial,1);error_spca=zeros(n_trial,1);
    for ut=1:n_trial
        X=(Ct^(1/2)*randn(p,n)+M*J');
        X_test=(Ct^(1/2)*randn(p,n_test)+M*J_test');
        [U,~]=eigs(X*X'/p,2);
        [V,~]=eigs(X*(J*J')*X'/(n*p),2);
        co=p/n;
        
        %%%%%%%%%%% SPCA/PCA Asymptotic %%%%%%%%%%%%%%%%
        Mgot=diag(c)^(1/2)*(M'*M)*diag(c)^(1/2)/co;
        [U1,VP]=eig(diag(c)+diag(c)^(1/2)*Mgot*diag(c)^(1/2));barlambda=diag(VP);
        barv=cell(2,1);vec=cell(2,1);
        for i=1:2
            barv{i}=U1(:,i);
            vec{i}=(co/barlambda(i))*diag(c)^(-1/2)*Mgot*diag(c)^(1/2)*barv{i}*barv{i}'*diag(c)^(1/2)*Mgot*diag(c)^(-1/2);
        end
        Mat=sqrt(vec{2});
        score_th2=(Mat(1,1));
        score_th_spca(1)=score_th2;
        score_th_spca(2)=-score_th2;
        [~,V_src]=eig(Mgot);[ell,ind]=sort(diag(V_src),'descend');ell=ell(1:2);
        maxis=1/sqrt(co);
        ell(ell<maxis)=[];
        [baru,~]=eig(Mgot);
         c=[0.5;0.5];
         score_th_pca=sqrt(diag(((co*ell^2-1)./(ell*(ell+1)))*diag(c)^(-1/2)*Mgot*diag(c)^(-1/2)*baru(:,2)*baru(:,2)')).*[-1;1];





        %%%%%%%%%%%%% Prediction on test data %%%%%%%%%%%%%
        pred_pca=zeros(size(X_test,2),1);
        pred_spca=zeros(size(X_test,2),1);
        score_test=U(:,1)'*X_test;
        score_test_spca=V(:,1)'*X_test;
        pred_pca(score_test>0)=1;pred_pca(score_test<0)=-1;
        pred_spca(score_test_spca>0)=1;pred_spca(score_test_spca<0)=-1;
        error_pca(ut)=sum(pred_pca~=[-ones(c(1)*n_test,1);ones(c(2)*n_test,1)])./(n_test);
        if error_pca(ut)>0.5
            error_pca(ut)=1-error_pca(ut);
        end
        error_spca(ut)=sum(pred_spca~=[-ones(c(1)*n_test,1);ones(c(2)*n_test,1)])./(n_test);
        if error_spca(ut)>0.5
            error_spca(ut)=1-error_spca(ut);
        end
    end
    error_pca_vec(lf)=mean(error_pca);
    error_spca_vec(lf)=mean(error_spca);
    perf_th_pca(lf)=0.5*erfc(((abs(score_th_pca(1))/sqrt(2))));
    perf_th_spca(lf)=0.5*erfc(((abs(score_th_spca(1))/sqrt(2))));
end
plot(p_vec,error_pca_vec,'r*-')
hold on
plot(p_vec,error_spca_vec,'go-')
plot(p_vec,perf_th_pca,'r-')
plot(p_vec,perf_th_spca,'g-')
vec=zeros(2*length(p_vec),1);
vec(1:2:end)=p_vec;
vec(2:2:end)=error_pca_vec;
sprintf('(%d,%d)',vec)
vec2=zeros(2*length(p_vec),1);
vec2(1:2:end)=p_vec;
vec2(2:2:end)=perf_th_pca;
sprintf('(%d,%d)',vec2)
vec3=zeros(2*length(p_vec),1);
vec3(1:2:end)=p_vec;
vec3(2:2:end)=error_spca_vec;
sprintf('(%d,%d)',vec3)
vec4=zeros(2*length(p_vec),1);
vec4(1:2:end)=p_vec;
vec4(2:2:end)=perf_th_spca;
sprintf('(%d,%d)',vec4)