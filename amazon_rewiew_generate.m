function [X,X_test,y_true,M] = amazon_rewiew_generate(domain,m,k,ns,nst)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
        addpath('/Users/tiomokomalik/Downloads/amazon_review/');
        % Load data
        str_domains = {'books','dvd','elec','kitchen'};
        for i=1:k-1
            load([str_domains{domain(i)},'_400.mat']);
            Xs{i} = fts;    clear fts;
            Ys{i} = labels; clear labels;
            Ys{i}=Ys{i}+1;
        end
        load([str_domains{domain(k)},'_400.mat']);
        Xt = fts;    clear fts;
        Yt = labels; clear labels;
%          Ys = Ys + 1;
         Yt = Yt + 1;
        for i=1:k-1
            Xs{i} = Xs{i} ./ repmat(sum(Xs{i},2),1,size(Xs{i},2)); 
            Xs{i} = zscore(Xs{i},1)/sqrt(1.0);
        end
        Xt = Xt ./ repmat(sum(Xt,2),1,size(Xt,2)); 
        Xt = zscore(Xt,1)/sqrt(1.0);
%         ns=zeros(m*k,1);
        c1=1:m;c2=1:m;
        X_s=[];
        for task=1:k-1
            X1{task}=[];y1{task}=[];
%             Xu1{task}=[];
            for g=1:m
                X11{task,g}=Xs{task}(Ys{task}==c1(g),:)';
%                 vec_tot{task}=X11{task,g};
                [vec_lab{task},idx{task}]=datasample(X11{task,g},ns(g+m*(task-1)),2);
%                 vec_tot{task}(:,idx{task})=[];
%                 [vec_unla{task},idx_un{task}]=datasample(vec_tot{task},nst(g),2);
                 X1{task}=[X1{task} vec_lab{task}];
%                  Xu1{task}=[Xu1{task} vec_unla{task}];
    %             M11=[M11 mean(X11{g}(:,subs{g}),2)];
    %             y1 = [y1;g*ones(ns(g), 1)];
    %             nr1(g)=size(X11{g},2);
                X_pop{g+m*(task-1)}=X11{task,g};
            end
            X_s=[X_s X1{task}];
        end
%         Z_l1=kron(e1,X1);
        X2=[];y2=[];X_test=[];
        for g=1:m
            X22{g}=Xt(Yt==c2(g),:)';
            vec_tot2=X22{g};
            [vec_lab2,idx2]=datasample(X22{g},ns(g+m*(k-1)),2);
            vec_tot2(:,idx2)=[];
            [vec_unla2,~]=datasample(vec_tot2,nst(g),2);
            X2=[X2 vec_lab2];
            X_test=[X_test vec_unla2];
            X_pop{m*(k-1)+g}=vec_lab2;
        end
        X=[X_s X2];
        y_true=[];
        for g=1:m
            y_true=[y_true;g*ones(nst(g),1)];
        end
        M=compute_statistic(X_pop);
end

