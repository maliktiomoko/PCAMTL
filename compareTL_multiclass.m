%% Script comparing CDLS,MTL-LSSVM, MTL-SPCA,N-SPCA and ST-SPCA on multi class classification on some datasets
clear all
clc
close all
% dataset='Office';
% dataset='officehome';
dataset='Imageclef';
% dataset='synthetic';

%%%%%% Choose source (sou={1,2,3}) and target (tar={1,2,3}) in each dataset

switch dataset
    case 'synthetic'
        p_vec=100:100:1000;
        beta_vec=linspace(0,1,10);
        error_opt2=zeros(length(beta_vec),1);error2=zeros(length(beta_vec),1);
        for j=1:length(beta_vec)
            p=p_vec(5);
            m=10;k=2;
            ns=[100*ones(m*(k-1),1);50*ones(m,1)];
            nst=1000*ones(m*k,1);
            tot_vec=logspace(log10(3),log10(5),10);
            lambda_vec=logspace(-2,2,10);
            param=0.0;
            [S,T,X_test_target,y_test,M,Ct,X_test_source,ys,yt,Moy] = generate_mvr(ns,nst,p,m,k,beta_vec(j),tot_vec(1),'multi',param);
            y_test=y_test';
            X=[S.fts{1}*(eye(size(S.fts{1},2))-(1/(size(S.fts{1},2)))*ones((size(S.fts{1},2)))) T.fts'*(eye(size(T.fts,1))-(1/(size(T.fts,1)))*ones((size(T.fts,1))))]/sqrt(k*p);
            X_test_target=X_test_target-Moy{k};
            [error2(j),error_opt2(j)] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M);
            Xs=X(:,1:sum(ns(1:m)));Xt=X(:,sum(ns(1:m))+1:end);
            [error_pca_vec,error_lssvm_vec] = PCA_ST_multi_class(Xt,X_test_target,y_test,m,ns(m+1:end));
        end
        
    case 'Office'
        addpath('data/');
        m=10;k=2;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains = {'caltech','webcam','dslr','amazon'};
        fileS=load([str_domains{4},'_VGG-FC7.mat']);
        fileT=load([str_domains{3},'_VGG-FC7.mat']);

    case 'officehome' 
        addpath('data/officehome_resnet50')
        m=31;k=2;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains={'Art','RealWorld','Product','Clipart'};
        sou=1;tar=4;
        pie1=readtable([str_domains{sou},'_',str_domains{sou},'.csv']);
        pie2=readtable([str_domains{sou},'_',str_domains{tar},'.csv']);
    case 'office31' 
        addpath('data/resnet50_feature')
        m=30;k=2;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains={'webcam','amazon','dslr'};
        sou=3;tar=2 ;
        pie1=readtable([str_domains{sou},'_',str_domains{sou},'.csv']);
        pie2=readtable([str_domains{sou},'_',str_domains{tar},'.csv']);
    case 'Imageclef' 
        addpath('data/imageCLEF_resnet50')
         m=12;k=2;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains={'p','i','c'};
        sou=1;tar=2;
        pie1=readtable([str_domains{sou},'_',str_domains{sou},'.csv']);
        pie2=readtable([str_domains{sou},'_',str_domains{tar},'.csv']);       
end
if strcmp(dataset,'Office')
    data_source1=fileS.FTS;
    data_target=fileT.FTS';
    label_source{1}=fileS.LABELS;
    label_target=fileT.LABELS;
else
    label_source{1}=table2array(pie1(:,end))+1;
    data_source1=table2array(pie1(:,1:end-1));

    label_target=table2array(pie2(:,end))+1;
    data_target=table2array(pie2(:,1:end-1));
end
data_source{1}=zscore(data_source1);
data_target=zscore(data_target);
n_trial=10;
accuracy_lssvm=zeros(n_trial,1);accuracy_lssvm_opt=zeros(n_trial,1);
error_vec=zeros(n_trial,1);error_opt_vec=zeros(n_trial,1);
accuracy_CDLS_vec=zeros(n_trial,1);
for tr=1:n_trial
    [X,X_test_target,y_test,M,ns,nst,ys,yt]=organize_data(data_source,data_target,label_source,label_target,m,ns,nst,k,data);
    %%%%%%%% Proposed MTL SPCA %%%%%%%%%%%%
    [error_vec(tr),error_opt_vec(tr)] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M);
    Xs{1}=X(:,1:sum(ns(1:m)));Xt=X(:,sum(ns(1:m))+1:end);
    %%%%%%%% Single Task SPCA %%%%%%%%%%%%%
    [error_pca_vec(tr),error_lssvm_vec(tr)] = PCA_ST_multi_class(Xt,X_test_target,y_test,m,ns(m+1:end));
    %%%%%%%% CDLS method      %%%%%%%%%%%%%
    [accuracy_CDLS_vec(tr)] = CDLS_method(Xs{1}',ys{1}',Xt',yt',X_test_target',y_test');
    data_train.source = Xs{1}';
    data_train.target = Xt';
    labels_train.source = ys{1}';
    labels_train.target = yt';
    %%%%%%%% MTL LSSVM        %%%%%%%%%%%%%
    addpath('data/RMT-MTLLSSVM-master')
    [accuracy_lssvm(tr), accuracy_lssvm_opt(tr)] = RMTMTLSSVM_train(Xs, ys, Xt, yt, X_test_target,y_test',m,ns,M,k);
end
accuracy_CDLS=mean(accuracy_CDLS_vec);
accuracy_pca=1-mean(error_pca_vec);
accuracy_lssvm1=1-mean(error_lssvm_vec);
accuracy=1-mean(error_vec);
accuracy_opt=1-mean(error_opt_vec);
formatSpec="ST-SPCA: %f, N-SPCA: %f, LSSVM: %f, CDLS: %f, MTL SPCA: %f";
sprintf(formatSpec,accuracy_pca*100,accuracy*100,mean(accuracy_lssvm_opt)*100,accuracy_CDLS,accuracy_opt*100)