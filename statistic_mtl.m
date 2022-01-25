function [score_th,score_th_opt,variance_th,variance_th_opt,tilde_y] = statistic_mtl(ntot,MM1,MM,tildev,tildev2,co,k,m)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% n1=sum(ntot(1:2));n2=sum(ntot(3:4));
c=ntot./sum(ntot);
e3=zeros(m*k,1);e3(m*(k-1)+1)=1;
e4=zeros(m*k,1);e4(m*(k))=1;
% yy12=(tilde_y*tilde_y')./(norm(tilde_y)^2);
%  [Up,eir]=eig((diag(c)*MM*diag(c)/co+diag(c)));
tilde_y0=kron(ones(k,1),[1;-1]);
 val=tilde_y0'*(diag(c)*MM*diag(c)/co+diag(c/k))*tilde_y0;
%  [eir_sort,ind]=sort(diag(eir),'descend');Up_sort=Up(:,ind);Up2_sort=(Up_sort);
%  [eir_sort2,ind2]=sort(diag(eir2),'descend');Up_sort2=Up2(:,ind2);Up2_sort2=(Up_sort2);
%  [U,V]=eig(X*J*K*J'*X');[rf,ind2]=sort(diag(V),'descend');U_sort=U(:,ind2);U2_sort=inv(U_sort);
tilde_y=(diag(c.*tildev2)+diag(c)*MM1*diag(c/co))\(diag(c/co)*MM*(e3-e4));
 val_opt=tilde_y'*(diag(c)*MM*diag(c)/co+diag(c/k))*tilde_y;
score_th_opt=(tilde_y'*diag(c)*MM/sqrt(co))./sqrt((tilde_y'*(diag(c.*tildev)+diag(c)*MM*diag(c/co))*tilde_y));
score_th_opt=abs(score_th_opt(3));
variance_th_opt=(1./(tilde_y'*(diag(c.*tildev)+diag(c)*MM*diag(c/co))*tilde_y))*(tilde_y'*(diag(c.*tildev2)+diag(c)*MM1*diag(c/co))*tilde_y);
score_th=(tilde_y0'*diag(c)*MM/sqrt(co))./sqrt((tilde_y0'*(diag(c.*tildev)+diag(c)*MM*diag(c/co))*tilde_y0));
score_th=(score_th(4));
variance_th=(1./(tilde_y0'*(diag(c.*tildev)+diag(c)*MM*diag(c/co))*tilde_y0))*(tilde_y0'*(diag(c.*tildev2)+diag(c)*MM1*diag(c/co))*tilde_y0);
% obj=@(y) -(y'*diag(c)*M'*M*(e3-e4)*(e3-e4)'*M'*M*diag(c)*y/co)./(y'*(diag(c.*tildev2)+diag(c)*M'*Ct(:,:,3)*M*diag(c/co))*y);
% opt=fminunc(obj,tilde_y0);
% r=1;

score_th_opt=real(sqrt((1./(co*val_opt))*e3'*MM*diag(c)*tilde_y*tilde_y'*diag(c)*MM*e3));
score_th=real(sqrt((1./(co*val))*e3'*MM*diag(c)*tilde_y0*tilde_y0'*diag(c)*MM*e3));
% variance_th=real(sqrt((1./(co*val_opt))*tilde_y'*diag(c)*MM1*diag(c)*tilde_y));
% variance_th_opt=real(sqrt((1./(co*val_opt))*tilde_y0'*diag(c)*MM1*diag(c)*tilde_y0));
%score_th_opt=real(sqrt(e3'*MM*diag(c)*yy12*Up_sort(:,1)*diag(1./eir_sort(1))*Up2_sort(1,:)*yy12*diag(c)*MM*e3/(co)));
%score_th=real(sqrt(e3'*MM*diag(c)*Up_sort2(:,1)*diag(1./eir_sort2(1))*Up2_sort2(1,:)*diag(c)*MM*e3/(co)));

end