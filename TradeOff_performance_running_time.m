%% Script comparing MTL LSSVM and MTL SPCA on different datasets from a performance and running time point of view

clear all
clc
close all
% dataset='Office';
dataset='officehome';
% dataset='office31';
% dataset='Imageclef';
% dataset='synthetic';
switch dataset
    case 'synthetic'
        p=200;
        beta_vec=linspace(0,1,10);
        m=10;k=3;
        ns=[100*ones(m*(k-1),1);50*ones(m,1)];
        nst=1000*ones(m*k,1);
        tot_vec=logspace(log10(1),log10(5),10);
        lambda_vec=logspace(-2,2,10);
        param=0.0;
        [S,T,X_test_target,y_test,M,Ct,X_test_source,ys,yt] = generate_mvr(ns,nst,p,m,k,beta_vec(5),tot_vec(1),'multi',param);
        y_test=y_test';
        X=[];
        for task=1:k-1
            X=[X S.fts{task}*(eye(size(S.fts{task},2))-(1/(size(S.fts{task},2)))*ones((size(S.fts{task},2))))];
        end
        X=[X T.fts'*(eye(size(T.fts,1))-(1/(size(T.fts,1)))*ones((size(T.fts,1))))]/sqrt(k*p);
        tic
        [error2,error_opt2] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M);
        time_pca=toc;
        for task=1:k-1
            Xs{task}=X(:,sum(ns(1:m*(task-1)))+1:sum(ns(1:m*(task))));
        end
        Xt=X(:,sum(ns(1:m*(k-1)))+1:end);
        addpath('data/RMT-MTLLSSVM-master')
        tic
        [accuracy_lssvm, accuracy_lssvm_opt] = RMTMTLSSVM_train(Xs, ys, Xt, yt, X_test_target,y_test',m,ns,M,k);
        time_lssvm=toc;
    case 'Office'
        addpath('data/');
        m=10;k=4;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains = {'caltech','webcam','amazon','dslr'};
        fileS1=load([str_domains{1},'_VGG-FC7.mat']);
        fileS2=load([str_domains{2},'_VGG-FC7.mat']);
        fileS3=load([str_domains{3},'_VGG-FC7.mat']);
        fileT=load([str_domains{4},'_VGG-FC7.mat']);
        data_source{1}=fileS1.FTS;
        data_source{2}=fileS2.FTS;
        data_source{3}=fileS3.FTS;
        data_target=fileT.FTS;
        label_source{1}=fileS1.LABELS;
        label_source{2}=fileS2.LABELS;
        label_source{3}=fileS3.LABELS;
        label_target=fileT.LABELS;
        accuracy_lssvm=zeros(n_trial,1);accuracy_lssvm_opt=zeros(n_trial,1);
        time_lssvm=zeros(n_trial,1);time_pca=zeros(n_trial,1);
        error2=zeros(n_trial,1);error_opt2=zeros(n_trial,1);
        n_trial=10;
        for tr=1:n_trial
            [X,X_test_target,y_test,M,ns,nst,ys,yt]=organize_data(data_source,data_target,label_source,label_target,m,ns,nst,k,data);
            tic
            [error2(tr),error_opt2(tr)] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M);
            time_pca(tr)=toc;
            for task=1:k-1
                Xs{task}=X(:,sum(ns(1:m*(task-1)))+1:sum(ns(1:m*(task))));
            end
            Xt=X(:,sum(ns(1:m))+1:end);
            data_train.source = Xs';
            data_train.target = Xt';
            labels_train.source = ys';
            labels_train.target = yt';
            addpath('data/RMT-MTLLSSVM-master')
            tic;
            [accuracy_lssvm(tr), accuracy_lssvm_opt(tr)] = RMTMTLSSVM_train(Xs, ys, Xt, yt, X_test_target,y_test',m,ns,M,k);
            time_lssvm(tr)=toc;  
        end
    case 'officehome' 
        addpath('data/officehome_resnet50')
        m=31;k=4;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains={'Art','RealWorld','Product','Clipart'};
        sou=2;tar=4;
        pie11=readtable([str_domains{1},'_',str_domains{1},'.csv']);
        pie12=readtable([str_domains{2},'_',str_domains{2},'.csv']);
        pie13=readtable([str_domains{3},'_',str_domains{3},'.csv']);
        pie2=readtable([str_domains{1},'_',str_domains{4},'.csv']);
        label_source{1}=table2array(pie11(:,end));
        data_source{1}=table2array(pie11(:,1:end-1));
        data_source{1}=zscore(data_source{1});
        label_source{2}=table2array(pie12(:,end));
        data_source{2}=table2array(pie12(:,1:end-1));
        data_source{2}=zscore(data_source{2});
        label_source{3}=table2array(pie13(:,end));
        data_source{3}=table2array(pie13(:,1:end-1));
        data_source{3}=zscore(data_source{3});
        label_target=table2array(pie2(:,end));
        data_target=table2array(pie2(:,1:end-1));
        data_target=zscore(data_target);
        n_trial=10;
        accuracy_lssvm=zeros(n_trial,1);accuracy_lssvm_opt=zeros(n_trial,1);
        time_lssvm=zeros(n_trial,1);time_pca=zeros(n_trial,1);
        error2=zeros(n_trial,1);error_opt2=zeros(n_trial,1);
        for tr=1:n_trial
            [X,X_test_target,y_test,M,ns,nst,ys,yt]=organize_data(data_source,data_target,label_source,label_target,m,ns,nst,k,data);
            tic
            [error2(tr),error_opt2(tr)] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M);
            time_pca(tr)=toc;
            for task=1:k-1
                Xs{task}=X(:,sum(ns(1:m*(task-1)))+1:sum(ns(1:m*(task))));
            end
            
            Xt=X(:,sum(ns(1:m*(k-1)))+1:end);
            data_train.source = Xs';
            data_train.target = Xt';
            labels_train.source = ys';
            labels_train.target = yt';
            addpath('data/RMT-MTLLSSVM-master')
            tic
            [accuracy_lssvm(tr), accuracy_lssvm_opt(tr)] = RMTMTLSSVM_train(Xs, ys, Xt, yt, X_test_target,y_test',m,ns,M,k);
            time_lssvm(tr)=toc;
        end
    case 'office31' 
        addpath('data/resnet50_feature')
        m=30;k=3;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains={'webcam','amazon','dslr'};
        sou=1;tar=2;
        pie11=readtable([str_domains{1},'_',str_domains{1},'.csv']);
        pie12=readtable([str_domains{2},'_',str_domains{2},'.csv']);
        pie2=readtable([str_domains{1},'_',str_domains{3},'.csv']);
        label_source{1}=table2array(pie11(:,end));
        data_source{1}=table2array(pie11(:,1:end-1));
        data_source{1}=zscore(data_source{1});
        label_source{2}=table2array(pie12(:,end));
        data_source{2}=table2array(pie12(:,1:end-1));
        data_source{2}=zscore(data_source{2});
        
        label_target=table2array(pie2(:,end));
        data_target=table2array(pie2(:,1:end-1));
        data_target=zscore(data_target);
        n_trial=10;
        accuracy_lssvm=zeros(n_trial,1);accuracy_lssvm_opt=zeros(n_trial,1);
        time_lssvm=zeros(n_trial,1);time_pca=zeros(n_trial,1);
        error_vec2=zeros(n_trial,1);error_opt_vec2=zeros(n_trial,1);
        for tr=1:n_trial
            [X,X_test_target,y_test,M,ns,nst,ys,yt]=organize_data(data_source',data_target',label_source,label_target,m,ns,nst,k,data);
            tic
            [error_vec2(tr),error_opt_vec2(tr)] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M);
            time_pca(tr)=toc;
            for task=1:k-1
                Xs{task}=X(:,sum(ns(1:m*(task-1)))+1:sum(ns(1:m*(task))));
            end
            Xt=X(:,sum(ns(1:m*(k-1)))+1:sum(ns(1:m*(k))));
            data_train.source = Xs';
            data_train.target = Xt';
            labels_train.source = ys';
            labels_train.target = yt';
            addpath('data/RMT-MTLLSSVM-master')
            tic
            [accuracy_lssvm(tr), accuracy_lssvm_opt(tr)] = RMTMTLSSVM_train(Xs, ys, Xt, yt, X_test_target,y_test',m,ns,M,k);
            time_lssvm(tr)=toc;
        end
    case 'Imageclef' 
        addpath('data/imageCLEF_resnet50')
        m=12;k=3;
        data='all';
        ns=[10*ones(m*(k-1),1);10*ones(m,1)];
        nst=10*ones(m,1);
        str_domains={'p','i','c'};
        sou=1;tar=2;
        pie11=readtable([str_domains{1},'_',str_domains{1},'.csv']);
        pie12=readtable([str_domains{2},'_',str_domains{2},'.csv']);
        pie2=readtable([str_domains{1},'_',str_domains{3},'.csv']);
        label_source{1}=table2array(pie11(:,end))+1;
        data_source{1}=table2array(pie11(:,1:end-1));
        data_source{1}=zscore(data_source{1});
        label_source{2}=table2array(pie12(:,end))+1;
        data_source{2}=table2array(pie12(:,1:end-1));
        data_source{2}=zscore(data_source{2});
        label_target=table2array(pie2(:,end))+1;
        data_target=table2array(pie2(:,1:end-1));
        data_target=zscore(data_target);
        n_trial=10;
        accuracy_lssvm=zeros(n_trial,1);accuracy_lssvm_opt=zeros(n_trial,1);
        time_lssvm=zeros(n_trial,1);time_pca=zeros(n_trial,1);
        error_vec=zeros(n_trial,1);error_opt_vec=zeros(n_trial,1);
        for tr=1:n_trial
            [X,X_test_target,y_test,M,ns,nst,ys,yt]=organize_data(data_source',data_target',label_source,label_target,m,ns,nst,k,data);
            tic
            [error_vec(tr),error_opt_vec(tr)] = RMTMTLSPCA_multiclass(X,X_test_target,y_test,k,m,ns,M);
            time_pca(tr)=toc;
            for task=1:k-1
                Xs{task}=X(:,sum(ns(1:m*(task-1)))+1:sum(ns(1:m*(task))));
            end
            Xt=X(:,sum(ns(1:m*(k-1)))+1:sum(ns(1:m*(k))));  
            data_train.source = Xs';
            data_train.target = Xt';
            labels_train.source = ys';
            labels_train.target = yt';
            addpath('data/RMT-MTLLSSVM-master')
            tic
            [accuracy_lssvm(tr), accuracy_lssvm_opt(tr)] = RMTMTLSSVM_train(Xs, ys, Xt, yt, X_test_target,y_test',m,ns,M,k);
            time_lssvm(tr)=toc;
        end     
end
formatSpec="Dataset: %s, Time SPCA: %f, Error SPCA: %f, Time LSSVM: %f, Error LSSVM: %f";
sprintf(formatSpec,dataset,mean(time_pca),mean(error_opt2),mean(time_lssvm),1-mean(accuracy_lssvm_opt))