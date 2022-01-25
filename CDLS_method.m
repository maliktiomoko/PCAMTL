function [accuracy_CDLS] = CDLS_method(S,S_label,T,T_label,Ttest,Ttest_labell)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% S = data_train.source;
% T = data_train.target;
% T_label = labels_train.target;
% S_label = labels_train.source;
% Ttest_labell = label_test;
% Ttest = data_test;
addpath('/Users/tiomokomalik/Documents/RMTMTLLSSVM_matlab/')
addpath('/Users/tiomokomalik/Documents/RMTMTLLSSVM_matlab/utils_2/CDLS_functions')
addpath('/Users/tiomokomalik/Documents/RMTMTLLSSVM_matlab/utils_2/liblinear-weights-2.30/')
S = S ./ repmat(sqrt(sum(S.^2,2)),1,size(S,2));
T = T ./ repmat(sqrt(sum(T.^2,2)),1,size(T,2));
Ttest = Ttest ./ repmat(sqrt(sum(Ttest.^2,2)),1,size(Ttest,2));

Data_C.T = T';
Data_C.Ttest = Ttest';
Data_C.S = S';
Data_C.T_Label = T_label';
Data_C.S_Label = S_label';
Data_C.Ttest_Label = Ttest_labell';
%%%%% Parameter Setting %%%%%
param_C.iter = 5;
param_C.scale = 0.15;
param_C.delta = 1.0; %% You can tune the portion of the weights if you like (0 < delta <= 1 )
param_C.PCA_dimension = 100; %% Make sure this dim. is smaller the source-domain dim.

%%%%% Start CDLS %%%%%
%fprintf('Transfering knowledge from Amazon images with DeCAF features to DSLR images with SURF features ...\n');
accuracy_CDLS = CDLS(Data_C,param_C);
end

