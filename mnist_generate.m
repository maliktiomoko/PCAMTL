function [X,X_test,y_test,M] = mnist_generate(data,test,labels,labels_test,selected_labels_target,selected_labels,k,m,ns,nur)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
        
        init_n=length(data(1,:));test_n=length(test(1,:));
        p=length(data(:,1));
         data = data/max(data(:));test = test/max(test(:));
%          data=zscore(data);test=zscore(test);
        mean_data=mean(data,2);mean_test=mean(test,2);
        norm2_data=0;norm2_test=0;
        for i=1:init_n
            norm2_data=norm2_data+1/init_n*norm(data(:,i)-mean_data)^2;
        end
        for i=1:test_n
            norm2_test=norm2_test+1/test_n*norm(test(:,i)-mean_test)^2;
        end
         data=(data-mean_data*ones(1,size(data,2)))/sqrt(norm2_data)*sqrt(p);
         test=(test-mean_test*ones(1,size(test,2)))/sqrt(norm2_test)*sqrt(p);
        selected_data = cell(k,1);
        selected_data_target = cell(k,1);selected_test = cell(k,1);
        %cascade_selected_data=[];
        for task=1:k-1
            j=1;
            cascade_selected_data{task}=[];
        for i=selected_labels{task}
            selected_data{task,j}=data(:,labels==i);
            ny =size(selected_data{task,j},2) ;
            shuffle = randsample(1:ny,ny) ;
            selected_data{task,j} = selected_data{task,j}(:,shuffle) ;
            selected_data{task,j}=selected_data{task,j}(:,1:ns(j+m*(task-1)));
            cascade_selected_data{task} = [cascade_selected_data{task}, selected_data{task,j}];
            selected_test_source{j+m*(task-1)}=test(:,labels_test==i);
            j = j+1;
        end
        end
        kc=1;
        for i=selected_labels_target
            selected_data_target{kc}=data(:,labels==i);
            ny =size(selected_data_target{kc},2) ;
            shuffle = randsample(1:ny,ny) ;
            selected_data_target{kc} = selected_data_target{kc}(:,shuffle) ;
            selected_data_target{kc}=selected_data_target{kc}(:,1:ns(m*(k-1)+kc));
            selected_test{kc}=test(:,labels_test==i);
            kc=kc+1;
        end
        X_s=[];
            for task=1:k-1
                X_l1{task}=[];
                for j=1:m
                    X_l1{task}=[X_l1{task} selected_data{task,j}];
                end
                X_s=[X_s X_l1{task}];
            end
        X_l2=[];
        for j=1:m
            X_l2=[X_l2 selected_data_target{j}];
        end
        X=[X_s X_l2];
        X_test=[];
        for j=1:m
            ny =size(selected_test{j},2) ;
            shuffle = randsample(1:ny,ny) ;
            selected_test{j} = selected_test{j}(:,shuffle) ;
            X_test=[X_test selected_test{j}(:,1:nur(j))];
        end
        y_test=[];
        for j=1:m
            y_test=[y_test;j*ones(nur(j),1)];
        end
        for task=1:k-1
            for j=1:m
                X_pop{j+m*(task-1)}=selected_test_source{j+m*(task-1)};
            end
        end
        for j=1:m
        X_pop{m*(k-1)+j}=selected_data_target{j};
        end
        M=compute_statistic(X_pop);
end