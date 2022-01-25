function [S,T,X_test_target,y_test,M,Ct,X_test_source,ys,yt,moy] = generate_mvr(ns,nst,p,m,k,beta,rt,method,param)
%UNTITLED13 Summary of this function goes here
%   Detailed explanation goes here
% seed=167;rng(seed);
          X_test_target=[];y_test=[];%M=[];
%             M=zeros(p,k*m);M_orth=zeros(p,k*m);
%          M(:,1)=2*[1;0;0;0;zeros(p-4,1)];Ct(:,:,1)=eye(p);
%          M(:,2)=2*[-1;0;0;0;zeros(p-4,1)];Ct(:,:,2)=eye(p);
%     number4=randi(200);
number=1;
switch method
    case 'binary'
          M=[];
          M(:,1)=rt*[1;0;0;0;0;0;zeros(p-6,1)];
          M(:,2)=rt*[-1;0;0;0;0;0;zeros(p-6,1)];
          M_orth(:,1)=rt*[0;1;0;0;0;0;zeros(p-6,1)];
          M_orth(:,2)=rt*[0;-1;0;0;0;0;zeros(p-6,1)];
%           Ct(:,:,1)=eye(p);Ct(:,:,2)=eye(p);
%           Ct(:,:,3)=eye(p);Ct(:,:,4)=eye(p);
        Ct(:,:,1)=number*toeplitz(param.^(0:p-1));Ct(:,:,2)=number*toeplitz(param.^(0:p-1));
    case 'multi'
        M=zeros(p,m*k);
        M_orth=zeros(p,m*k);
%         M=randn(p,m*k)/3;
%         M_orth=randn(p,m*k)/3;
        for j=1:m
            Ct(:,:,j)=number*toeplitz(param.^(0:p-1));
%             M=[M randn(p,1)];
            M(j,j)=rt;
            M_orth(end-j,j)=rt;
            
        end
end
%          M_orth(:,2)=2*[0;-1;0;0;zeros(p-4,1)];
%            seed=167;rng(seed);
%         number=randi(200);
%          number=randi(10);
%         number=randi(2);
% U=randn(p,1);U=U./norm(U); U_orth_mat=null(U.') ;U_orth=U_orth_mat(:,1);U_orth=U_orth./norm(U_orth);
%         number=1;
         for s=2:k-1
            for jf=1:m
                   M(:,m*(s-1)+jf)=(beta(s-1)*M(:,jf)+sqrt(1-beta(s-1)^2)*M_orth(:,jf));
                   %M_orth(p-m*(s-1)-jf,m*(s-1)+jf)=rt;
%                    M=[M jf*ones(p,1)];
%                     M=[M randn(p,1)];
%                   Ct(:,:,m*(s-1)+jf)=1*toeplitz(abs(rand()).^(0:p-1));
                     Ct(:,:,m*(s-1)+jf)=number*toeplitz(param.^(0:p-1));
            end
         end
%         beta=beta/2;
% M=[U -U beta*U+sqrt(1-beta^2)*U_orth -(beta*U+sqrt(1-beta^2)*U_orth)];
        for jf=1:m
               M(:,m*(k-1)+jf)=(beta(end)*M(:,jf)+sqrt(1-beta(end)^2)*M_orth(:,jf));
               %M=[M jf*ones(p,1)];
%                   M=[M rand(p,1)];
%               Ct(:,:,m*(k-1)+jf)=1*toeplitz((abs(rand())).^(0:p-1));
                 Ct(:,:,m*(k-1)+jf)=number*toeplitz(param.^(0:p-1));
        end
        %M=[M M];
        %M=zeros(p,k*m);
%           M(:,1)=10*[1;0;0;0;zeros(p-4,1)];
%           M(:,2)=10*[0;1;0;0;zeros(p-4,1)];
%           M(:,3)=10*[0;0;1;0;zeros(p-4,1)];
%           M(:,4)=10*[0;0;0;1;zeros(p-4,1)];
%          orth=0.8*[ 0; 0;1;1;zeros(p-4,1)];
%          orth2=-0.8*[ 0; 0;1;-1;zeros(p-4,1)];
%         %mus2=1*rand(p,1)/sqrt(p);
%         %mut1=mus1;
%         %mut2=mus2;
%         %mut1=[ 2;0;0;0;zeros(p-4,1)];
%         %mut2=[-2;0;0;0;zeros(p-4,1)];
%         %mut1=(2/sqrt(2))*[2;-2;0;0;zeros(p-4,1)];
%          beta=0.75;zeta=sqrt(1-beta^2);
%         %mut1=mus1;
%          M(:,m+1)=beta*M(:,1)+zeta*orth;M(:,m+2)=beta*M(:,2)+zeta*orth2;
%          M(:,6)=beta*M(:,3)+zeta*orth;
        
%         M=kron(ones(1,2),M);
% alpha=[0.1 0.2 0.3 0.4];
%  Ct(:,:,1)=alpha(1)*eye(p);Ct(:,:,2)=alpha(2)*eye(p);
%  Ct(:,:,3)=alpha(3)*eye(p);Ct(:,:,4)=alpha(4)*eye(p);
%  Ct(:,:,1)=toeplitz(alpha(1).^(0:p-1));
%  Ct(:,:,2)=toeplitz(alpha(2).^(0:p-1));
%  Ct(:,:,3)=toeplitz(alpha(3).^(0:p-1));
%  Ct(:,:,4)=toeplitz(alpha(4).^(0:p-1));
% Moy=load('Moy.mat');M=Moy.M;
y1=[];y2=[];
        for task=1:k-1
            X1{task}=[];y1{task}=[];
            for j=1:m
                X1{task} = [X1{task} M(:,m*(task-1)+j)+(Ct(:,:,m*(task-1)+j)^(1/2))*randn(p,ns(m*(task-1)+j))];
                y1{task}=[y1{task};j*ones(ns(m*(task-1)+j),1)];
            end
        end
        X2=[];
        for j=1:m
            X2 = [X2 M(:,m*(k-1)+j)+(Ct(:,:,j+m*(k-1))^(1/2))*randn(p,ns(m*(k-1)+j))];
            y2=[y2;j*ones(ns(m*(k-1)+j),1)];
        end
        for j=1:m
            X_test_target = [X_test_target M(:,m*(k-1)+j)+(Ct(:,:,m*(k-1)+j)^(1/2))*randn(p,nst(j))];
            y_test=[y_test;j*ones(nst(j),1)];
        end
        X_test_source=[];
        for j=1:m
            X_test_source = [X_test_source M(:,j)+(Ct(:,:,j)^(1/2))*randn(p,nst(j))];
            %y_test=[y_test;j*ones(nst(j),1)];
        end
        %X_test2=M(:,4)+randn(p,nst(1));
        S.fts=X1';T.fts=X2';
        S.labels=y1';T.labels=y2';y_test=y_test';
        ys=y1{1};yt=y2;
        Moy=[];
        for tr=1:k
            moy{tr}=zeros(p,1);
            for j=1:m
            moy{tr}=moy{tr}+ns(m*(tr-1)+j)*M(:,m*(tr-1)+j)/sum(ns(m*(tr-1)+1:m*tr));
            end
            Moy=[Moy moy{tr}*ones(1,m)];
        end
        M=M-Moy;
%         op_norm1=sqrt(max(eig(Ct(:,:,1)*Ct(:,:,1)')));op_norm2=sqrt(max(eig(Ct(:,:,3)*Ct(:,:,3)')));
%         op_norm2=trace(Ct(:,:,3))/p;
%         S.fts{1}=S.fts{1}./sqrt(op_norm1);T.fts=T.fts./(op_norm2);
%         M(:,1)=M(:,1)/sqrt(op_norm1);M(:,2)=M(:,2)/sqrt(op_norm1);M(:,3)=M(:,3)/(op_norm2);M(:,4)=M(:,4)/(op_norm2);
%          data=[data2 data1];
% %          test=(test-mean_test*ones(1,size(test,2)))/(sqrt(norm2_test));
% %          test=(test-mean_test*ones(1,size(test,2)))/(norm(Ct(:,:,3)));
%           test=zscore(test);
          %X_test_target=X_test_target./(op_norm2);
end
