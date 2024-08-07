%=========================================================
%
% DEMO BAND SELECTION
% 
% This code runs a demo band selection using the Tensor based HSI Band
% Selection method based on Generalize 3dTV and Tensor CUR Decomposition
%
% MATLAB R2023b
% Author: Katherine Henneberger
% Institution: University of Kentucky - Math Department
%  
%=========================================================

clear;clc;close all;
addpath(genpath(pwd))


%% Run Proposed method:
dataset_names = {'Indian_Pines'}; 
method_names = {'THBSCUR'}; % this is the proposed method
svm_para = {'-c 10000.000000 -g 0.500000 -m 500 -t 2 -q', '-c 100 -g 16 -m 500 -t 2 -q'};
num_repe = 1; % raise this to increase the number of times classification is performed
classifier_names = {'SVM'};
K = 30;
delta = 3;
x = delta : delta : K; % number of the selected bands

%% initialization
Methods = cell(1, 1);
Methods{1, 1} = get_method_struct(method_names{1}, dataset_names, {'SVM'}, K / delta);

%% band selection
for dataset_id = 1
    % load data
    Dataset = get_data(dataset_names{dataset_id});
    Dataset.svm_para = svm_para{1, dataset_id};
    A = Dataset.A;
    
    [M, N, d] = size(A);
    if size(A,3)>1
        [n1,n2,n3] = size(A);
        n = n1*n2;
        A2 = reshape(A,n,n3);
    else
        A2 = A;
        [n,n3] = size(A2);
    end
    
    % Establish Opts for THBSCUR
    opts.tol = 10e-6;
    opts.max_iter = 50;
    opts.beta = 1;
    opts.DEBUG = 1;
    
    %% calculate the band set for THBSCUR
    cnt = 1;
    for j = x % number of bands
        opts.k = j;
        opts.rs = round(j*log(n*n3));
        opts.cs = round(j*log(n3));
        opts.p = 2;

        load(['results\svm_best_pars_indianpines(',num2str(j),')_v2.mat'])
        opts.beta = best_pars.beta;
        opts.lambda1 = best_pars.lambda1;
        opts.lambda2 = best_pars.lambda2;
        opts.tau = best_pars.tau;
        
        % run THBSCUR method
        fprintf('Running THBSCUR for %s, %d bands...\n', dataset_names{dataset_id}, j);
        [~, bandset, ~] = THBSCUR(A, opts);
        
        Methods{1, 1}.band_set{dataset_id, cnt} = bandset;
        cnt = cnt + 1;
    end

    %% test accuracy
    fprintf('Calculating Accuracy \n');
    for classifier_id = 1:length(classifier_names)
        for j = 1:length(x)
            Methods{1, 1}.accu(dataset_id, classifier_id, j) = 0;
        end
    end
    
    for ite = 1:num_repe
        % refresh the training and testing samples
        if ite > 1
            Dataset = get_data(dataset_names{dataset_id});
            Dataset.svm_para = svm_para{1, dataset_id};
        end
        
        for classifier_id = 1:length(classifier_names)
            cnt = 1;
            for k = x
                fprintf('Calculating Accuracy for %d bands...\n', k);
                cur_accu = test_bs_accu(Methods{1, 1}.band_set{dataset_id, cnt}, Dataset, classifier_names{classifier_id});
                Methods{1, 1}.accu(dataset_id, classifier_id, cnt) = ...
                    Methods{1, 1}.accu(dataset_id, classifier_id, cnt) + cur_accu.OA;
                cnt = cnt + 1;
            end
        end
    end

    %% calculate the mean accuracy over different iterations
    for classifier_id = 1:length(classifier_names)
        Methods{1, 1}.accu(dataset_id, classifier_id, :) = ...
            Methods{1, 1}.accu(dataset_id, classifier_id, :) / num_repe;
    end
end
    
%% Save the Methods variable
save('results\SVM_result_method_THBSCUR.mat', 'Methods');
proposed_method = Methods;
%% Load comparison methods:
load results\SVM_previous_results.mat
Methods{1,6} = proposed_method{1,1}; % use proposed results calculated above

%% Plot all together:
method_names = {'E-FDPC','FNGBS','SR-SSIM','MHBSCUR', 'MGSR','Proposed'}; 
plot_method_ids = [1, 2, 3,4,5,6];
plot_classifier_id = 1; %indicate which classifer to plot
dataset_id = 1;

%% 
for i = dataset_id
    fig = plot_method_improvedv2(Methods(plot_method_ids), x, classifier_names, plot_classifier_id, method_names(plot_method_ids), i, 1);
end

%% helper function
function [method_struct] = get_method_struct(method_name, dataset_names, classifier_names, band_num_cnt)
    method_struct.method_name = method_name;
    dataset_cnt = size(dataset_names, 2);
    classifier_cnt = size(classifier_names, 2);
    method_struct.band_set = cell(dataset_cnt, band_num_cnt);
    method_struct.band_set_corr = cell(dataset_cnt, band_num_cnt);
    method_struct.accu = zeros(dataset_cnt, classifier_cnt, band_num_cnt);
end
