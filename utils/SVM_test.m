% Learning Cross-Domain Landmarks for Heterogeneous Domain Adaptation
% Yao-Hung Hubert Tsai, Yi-Ren Yeh and Yu-Chiang Frank Wang
% IEEE Computer Vision and Pattern Recognition (CVPR), 2016.
%
% Contact: Yao-Hung Hubert Tsai (yaohungt@andrew.cmu.edu)

function [ac,largest_idx] = SVM_test(Cs,alpha,co_Xs,S_Label,co_Xt,T_Label,co_Xtest,Ttest_Label)

alpha = Cs*alpha;

pivot_label = [S_Label;T_Label];
pivot_feature = [co_Xs;co_Xt];
pivot_weight = [alpha;ones(length(T_Label),1)];

model = train(pivot_weight,pivot_label,sparse(pivot_feature));

[largest_idx,acc,~] = predict(Ttest_Label,sparse(co_Xtest),model, '-q');
%largest_idx
%acc
ac = acc(1);
end
