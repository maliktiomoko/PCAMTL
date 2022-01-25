%% Script comparing the performance of Naive MTL, MTL Optimized and Single task SPCA for increasing number of tasks.
clear all
clc
close all
addpath('utils/')
p=200;
% dataset='synthetic';
% dataset='nlp';
dataset='mnist';
if strcmp(dataset,'synthetic')
    n_task_vec=[2 4 8 16 32 64 128 256];
    betat=abs(rand(n_task_vec(end),1));
elseif strcmp(dataset,'nlp')
    n_task_vec=1:4;
else
    n_task_vec=1:5;
end
sigma=1;
switch dataset
    case 'mnist'
        addpath('data/');
        init_data = loadMNISTImages('train-images-idx3-ubyte');
         init_labels = loadMNISTLabels('train-labels-idx1-ubyte');
          init_test = loadMNISTImages('t10k-images-idx3-ubyte');
         init_test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        [labels_test,idx_init_labels_test]=sort(init_test_labels,'ascend');
        data=init_data(:,idx_init_labels);
        test=init_test(:,idx_init_labels_test);
        X=[data test];
        [coeff,score,latent,~,explained] = pca(X');
        data=score(1:60000,1:100)';test=score(60001:70000,1:100)';
        %%% Add gaussian noise
        data=data+sigma*randn(size(data));
        test=test+sigma*randn(size(test));
end
error1=zeros(length(n_task_vec),1);
error_opt_th=zeros(length(n_task_vec),1);
error_st=zeros(length(n_task_vec),1);
err_opt=zeros(length(n_task_vec),1);
for i=1:length(n_task_vec)
    k=n_task_vec(i)+1;
     n_trial=10;
     error=zeros(n_trial,1);error_opt=zeros(n_trial,1);
     error_opt_th_vec=zeros(n_trial,1);error_st_vec=zeros(n_trial,1);
     for rt=1:n_trial
         switch dataset
             case 'synthetic'
                 beta=[betat(1:n_task_vec(i)-1);betat(length(n_task_vec))];
                    m=2;
                    ns=[50*ones(m*(k-1),1);5*ones(m,1)]';
                    nst=10000*ones(m,1);
                    tot_vec=logspace(log10(1),log10(5),10);
                    param=0;
                    [S,T,X_test_target,y_test,M,~,X_test_source] = generate_mvr(ns,nst,p,m,k,beta,tot_vec(1),'binary',param);
                    MM=M'*M;
                    Source=[];
                    for task=1:k-1
                        Source=[Source S.fts{task}];
                    end
                    X=[Source T.fts'];
                    y_true=[-ones(nst(1),1);ones(nst(2),1)];
             case 'nlp'
                 m=2;
                 ns=[100*ones(m*(k-1),1);50*ones(m,1)]';
                 nst=2000*ones(m,1);
                 domains=[1:i 4];
                 [X,X_test_target,y_true,M] = amazon_rewiew_generate(domains,m,k,ns,nst);
                 MM=M'*M;
                 y_true(y_true==2)=-1;
                 trnY=[];
                 S.fts{1}=X(:,1:sum(ns(1:2)));
                 trnY=[ones(ns(1),1);-ones(ns(2),1)];
                 for task=1:k-2
                    S.fts{task+1}=X(:,sum(ns(1:2*task))+1:sum(ns(1:2+2*(task))))/sqrt(k*p);
                    trnY=[trnY;ones(ns(1+2*task),1);-ones(ns(2+2*task),1)];
                 end
                 T.fts=X(:,sum(ns(1:2*(k-1)))+1:sum(ns(1:2+2*(k-1))))/sqrt(k*p);
                 trnY=[trnY;ones(ns(1+2*(k-1)),1);-ones(ns(2+2*(k-1)),1)];
             case 'mnist'
                 m=2;
                 Matrice=[7 9;3 8;5 6;2 9;3 5];
                 ns=[100*ones(m*(k-1),1);50*ones(m,1)]';
                 nst=500*ones(m,1);
                 selected_labels_target=[1 4];
                 for tr=1:k-1
                     selected_labels{tr}=Matrice(tr,:);
                 end
                 [X,X_test_target,y_true,M] = mnist_generate(data,test,labels,labels_test,selected_labels_target,selected_labels,k,m,ns,nst);
                 MM=M'*M;
                 y_true(y_true==2)=-1;
                 trnY=[];
                 S.fts{1}=X(:,1:sum(ns(1:2)));
                 trnY=[ones(ns(1),1);-ones(ns(2),1)];
                 for task=1:k-2
                    S.fts{task+1}=X(:,sum(ns(1:2*task))+1:sum(ns(1:2+2*(task))))/sqrt(k*p);
                    trnY=[trnY;ones(ns(1+2*task),1);-ones(ns(2+2*task),1)];
                 end
                 T.fts=X(:,sum(ns(1:2*(k-1)))+1:sum(ns(1:2+2*(k-1))))/sqrt(k*p);
                 trnY=[trnY;ones(ns(1+2*(k-1)),1);-ones(ns(2+2*(k-1)),1)];
         end
        [error(rt),error_opt(rt),error_opt_th_vec(rt),~,~,error_st_vec(rt)] = RMTMTLSPCA_binary_train(X,X_test_target,y_true,MM,ns,k,m);
        gamma=ones(k,1);lambda=1;
        Ct=zeros(size(X,1),size(X,1),m*k);
        for tr=1:m*k
            Ct(:,:,tr)=eye(size(X,1));
        end
     end
     err_opt(i)=mean(error_opt);
     error1(i)=mean(error);
     error_opt_th(i)=mean(error_opt_th_vec);
     error_st(i)=mean(error_st_vec);
end
 figure
 hold on
plot(n_task_vec,err_opt)
plot(n_task_vec,error1)
plot(n_task_vec,error_st)
legend('Opt','N','ST')
vec=zeros(2*length(n_task_vec),1);
vec(1:2:end)=n_task_vec;
vec(2:2:end)=err_opt;
sprintf('(%d,%d)',vec)
vec2=zeros(2*length(n_task_vec),1);
vec2(1:2:end)=n_task_vec;
vec2(2:2:end)=error1;
 sprintf('(%d,%d)',vec2)
 vec3=zeros(2*length(n_task_vec),1);
vec3(1:2:end)=n_task_vec;
vec3(2:2:end)=error_st;
 sprintf('(%d,%d)',vec3)