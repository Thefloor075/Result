% Simulation script for the SLM Reservoir Computer setup
% Experimental transfer function (symmetric, homogeneous)
% KTH database, mixing actions from different people
% Classification based on HOG features from images
% PCA dimensionality reduction
% Offline/batch learning
% Written by Piotr Antonik, Jun 2018
% Cleaned up by P. Antonik, Jan 13, 2020

% References: dalal2005histograms, liu2009recognizing, schuldt2004recognizing

tic;
clear; %clc;
close all;
addpath(genpath('.'));
rng(1);


%% Load KTH features (HOG) & labels

% load('db/kth_hog9576_labels.mat');
load('./kth_hog8x8_pca2k_labels.mat');
fprintf('Database loaded.\n');
kth_hog_labels = kth_hog_pca2k_labels;
clear kth_hog_pca2k_labels

% duplicate one missing boxing sequence (person 22, cel 507)
kth_hog_labels = [kth_hog_labels(:,1:507) kth_hog_labels(:,507) kth_hog_labels(:,508:end)];
n_cells = size(kth_hog_labels, 2);
train_ratio = 0.75;

% add cell indexes for ease of tracking
for i_cell=1:size(kth_hog_labels,2)
    kth_hog_labels{3, i_cell} = i_cell * ones(1, size(kth_hog_labels{2, i_cell}, 2));
end

% create inputs & targets from data cells
data_train     = [kth_hog_labels(:, 1:4:end) kth_hog_labels(:, 2:4:end) kth_hog_labels(:, 3:4:end)];
data_train     = data_train(:, randperm(size(data_train,2)));
features_train = [data_train{1, :}];
labels_train   = [data_train{2, :}];
indexes_train  = [data_train{3, :}];

data_test     = kth_hog_labels(:, 4:4:end);
data_test     = data_test(:, randperm(size(data_test,2)));
features_test = [data_test{1, :}];
labels_test   = [data_test{2, :}];
indexes_test  = [data_test{3, :}];

inputs  = [features_train features_test];    % combine train & test data
targets = [labels_train labels_test];
indexes = [indexes_train indexes_test];

size_input = size(inputs, 1);

targets_bin = zeros(6, length(targets));
for i=1:6
    targets_bin(i, :) = targets==i;
end

clear data_train data_test kth_hog_labels i


%% Constants

len_res     = 45; % 32..128
size_res    = len_res^2;
t_train     = size(features_train, 2);
t_test      = size(features_test, 2);
t_inputs    = t_train + t_test;
reg_term    = 0;
n_masks     = 1;

clear features_train features_test labels_train labels_test 

%% Generate input & internal weights

masks     = 2*rand(size_res, size_input, n_masks) - 1;
rand_sgn  = 2*randi([1,2], size_res)-3;

%% Import SLM transfer function data

% --- homogeneous experimental function (same for all pixels)
% tt = importdata('db/slm_setup_transfer_function.data');
tt = importdata('./slm_setup_transfer_function.data');
slm_transf_lut = tt(:,2);
clear tt;
slm_transf = @(x) slm_transf_lut(x);
slm_offset = min(slm_transf_lut);
% --- inhomogeneous function (measured, different for each pixel)
% slm_transf_lut = importdata('db/slm_inhom_trans_fun_n1024.mat');
% slm_transf = @(pixel, intensity) slm_transf_lut(pixel,intensity);
% --- other homogeneous transfer functions
% slm_transf = @(x) x;
% slm_transf = @(x) sin(x);


%% Scanned parameters
[scan_params, scan_list, n_runs] = def_scan_params_slm_mnist(n_masks);


%% Run reservoir

res_err_train = zeros(n_runs, 1);
res_err_test  = zeros(n_runs, 1);
res_scores    = zeros(n_runs, 1);

for i_run=1:n_runs

    % set current parameters, print status
    gain_in    = scan_list(1, i_run);
    gain_fdb   = scan_list(2, i_run);
    gain_inter = scan_list(3, i_run);
    w_density  = scan_list(4, i_run);
    i_mask     = scan_list(5, i_run);
    mask_orig  = masks(:,:,i_mask);
    
    fprintf('Run %3.0f/%d: ', i_run, n_runs);
    fprintf('in: %.3f, ', gain_in);
    fprintf('fdb: %.3f, ', gain_fdb);
    fprintf('int: %.3f, ', gain_inter);
    fprintf('w_d: %.3f, ', w_density);
    fprintf('mask: %1.0f, ', i_mask);

    res_history = zeros(size_res, t_inputs);
    mx_fdb      = zeros(len_res);
    
    rng(2);
    w = sprand(size_res, size_res, w_density) .* rand_sgn * gain_inter;
    w(1:size_res+1:end) = gain_fdb * ones(size_res, 1);
    mask = gain_in * mask_orig;

    fprintf('[');
    for t=1:t_inputs
        % image to SLM (real-valued)
        if t==1
            res_in = mask * inputs(:, t);
        else
            res_in = mask * inputs(:, t) + w * res_history(1:size_res, t-1);
        end
        sgn_in = sign(res_in);
        res_in = round( res_in * 255);
        
        % apply SLM-Camera transfer function
        res_in(res_in>255)  = 255;
        res_in(res_in<-255) = -255;
        res_in(res_in==0)   = 1;
        res_out             = sgn_in .* (slm_transf( abs(res_in) ) - slm_offset);

        % record reservoir history
        res_history(1:size_res,t) = res_out(:) / 255;
        
        % display progess bar
        if mod(t, 5000) == 0
            fprintf('*');
        end
    end
    fprintf(']');
    
    % train reservoir
    reservoir      = [res_history; ones(1, t_inputs)];
    X              = reservoir(:, 1:t_train);
    R              = X*X' + reg_term*eye(size(X,1));
    P              = X * targets_bin(:, 1:t_train)';
    weights        = P' * pinv(R);
    rcouts         = weights * reservoir(:,1:t_inputs);
    [~, rcouts_md] = max(rcouts);
    err_train = sum(rcouts_md(1:t_train)~=targets(1:t_train))/t_train;
    err_test = sum(rcouts_md(t_train+1:t_inputs)~=targets(t_train+1:t_inputs))/t_test;
    
    res_err_train(i_run) = err_train;
    res_err_test(i_run)  = err_test;
    fprintf('TrErr: %.2e, TtErr: %.2e. ', res_err_train(i_run), res_err_test(i_run));
 
    % compute score
    analyse_kth_pca_mix;
    res_scores(i_run) = sum(diag(conf_matrix_test));
    fprintf('Score: %3.0f. ', res_scores(i_run));

    % remaining time
    t_rem = round( (n_runs-i_run));
    fprintf('tRem: %d min, %d sec.\n', floor(t_rem/60), rem(t_rem,60));
end

res_list      = [1:n_runs; scan_list; res_err_train'; res_err_test'; res_scores'];
res_list_srtd = sortrows(res_list', -size(res_list,1));

toc

function [ scan_params, scan_list, n_runs ] = def_scan_params_slm_mnist(n_masks)
% Definition of ranges of scanned parameters
% Written by Piotr Antonik, Jan 2018

    % 1. set labels & values
    scan_params = struct();

    scan_params.lbls{1} = 'Input Gain';
    scan_params.vals{1} = [0.001 0.01 0.1];
    scan_params.lens(1) = length(scan_params.vals{1});

    scan_params.lbls{2} = 'Feedback Gain';
    scan_params.vals{2} = 0.0:0.2:1.0;
    scan_params.lens(2) = length(scan_params.vals{2});
 
    scan_params.lbls{3} = 'Interconnectivity Gain';
    scan_params.vals{3} = 0.01; %[0.001 0.01 0.1];
    scan_params.lens(3) = length(scan_params.vals{3});
    
    scan_params.lbls{4} = 'Interconnectivity Matrix Density';
    scan_params.vals{4} = 0.01;%[0.001 0.01 0.1];
    scan_params.lens(4) = length(scan_params.vals{4});
    
    scan_params.lbls{5} = 'Input Mask';
    scan_params.vals{5} = 1:n_masks;
    scan_params.lens(5) = length(scan_params.vals{5});
    
%   
    % 2. generate permutations list
    n_runs       = prod(scan_params.lens);
    nscan_params = length(scan_params.lbls);
    scan_list    = zeros(nscan_params, n_runs);

    % understanding this loop requires a pencil, a paper and some concentration
    % better try it on a 3x3 example
    for i=1:nscan_params
        rep = prod( scan_params.lens(1:i-1) );
        cyc = prod( scan_params.lens(i+1:end) );
        % create rep x cyc matrix, then reshape columnwise into vector
        scan_list(i,:) = reshape( repmat(scan_params.vals{i}, rep, cyc), ...
            1, n_runs );
    end
end
