%--------------------------------------------------------------------------
% Iterative Convolutional Neural Network (IterCNN) algorithm for PET image
% reconstruction. The details of IterCNN are described in
%
%   Gong, Kuang, et al. "Iterative PET Image Reconstruction Using Convolutional
%   Neural Network Representation." arXiv preprint arXiv:1710.03344 (2017).
%--------------------------------------------------------------------------
% Programmer: Kuang Gong @ MGH and UC DAVIS,
% Contact: kgong@mgh.harvard.edu, kugong@ucdavis.edu
% Last Modified: 09-13-2018
% Note: This version is based on 3D U-net (detailed in our newly accepted TMI paper), 
% results shown in arXiv paper is based on 2D U-net. ---09-13-2018
%--------------------------------------------------------------------------
clear
clc
close all
curr_dir = pwd;
addpath(curr_dir);
addpath([curr_dir, '/utils'])
%% set running parameters
device_num= 0; % GPU device
myrho = 7.5e-4; % the \rho parameter used in ADMM algorithm
multiple_iter = 2; % subproblem 1 iteration number
subiter = 10; % subproblem 1 iteration number
maxit =100; % maximum iterations for the whole ADMM framework
Gopt.disp= 1; % display the computation process or not
Gopt.imgsiz = [180, 180, 49]; % image size
mymask = circmask([Gopt.imgsiz(1) Gopt.imgsiz(2)],Gopt.imgsiz(1)/2,Gopt.imgsiz(2)/2,70); % set the FOV
Gopt.mask = repmat(mymask,[1 1 Gopt.imgsiz(3)]);
Gopt.savestep = 10; % save every 10 iterations
Gopt.imgsiz_trunc = [128,128,49]; % to fit for the U-net input
Gopt.trunc_range = {27:154, 27:154,1:49}; % truncate from 180 x 180 to 128 x 128.
%% create folder to store data
save_folder = sprintf('./result/result_%e_multiiter%d_fititer%d',...
    myrho, multiple_iter, subiter);
if (exist(save_folder,'dir')~=7)
    str_make = sprintf('mkdir %s', save_folder);
    system(str_make);
end
system(sprintf('cp -pr ./utils/ge690_fb_180_github.cfg %s/', save_folder));
%system(sprintf('cp -pr initialize_residual_cnnoutput.py %s/', save_folder));
%system(sprintf('cp -pr BN_unet_1.0_basedonPossion_otsp_new_residual_momentum.py %s/', save_folder));
eval(sprintf('cd %s', save_folder));
mythreshold = 0.1;
%% load initial
load('../../data/xini.mat','xini');
xini = reshape(xini,Gopt.imgsiz).* mythreshold;
%% run recon
[x_cnnada,out] = eml_admm_cnn_multiplerecon_residual(Gopt, xini,  maxit, myrho, device_num, multiple_iter, subiter);
save(sprintf('cnn_%e.mat',myrho),'x_cnnada','out');









