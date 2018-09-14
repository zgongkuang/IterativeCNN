function [output_cnn, out] = eml_admm_cnn_multiplerecon_residual(Gopt, x0, maxit, myrho,device_num,multiple_iter, subiter)
%--------------------------------------------------------------------------
% Iterative Convolutional Neural Network (IterCNN) algorithm for PET image
% reconstruction. The details of IterCNN are described in
%
%   Gong, Kuang, et al. "Iterative PET Image Reconstruction Using Convolutional
%   Neural Network Representation." arXiv preprint arXiv:1710.03344 (2017).
%
%--------------------------------------------------------------------------
% INPUT:
%
%     Gopt            option set for G, can be empty if G is a matlab sparse matrix
%     x0              the initilization for input image
%     maxit           maximum iteration number
%     myrho           the parameter controling the ADMM convergence strengh
%     device_num      specify GPU device number
%     multiple_iter   the number of iterations for subproblem 1, indciated in the paper
%     subiter         the number of iterations for subproblem 2, indciated in the paper
%
% OUTPUT
%     output_cnn      the estimated final IterCNN output
%     out             other running information
%     Likelihood      the objective function for output, this value is not
%                     checked, cannot guarantee it is correct
%--------------------------------------------------------------------------
% Programmer: Kuang Gong @ MGH and UC DAVIS,
% Contact: kgong@mgh.harvard.edu, kugong@ucdavis.edu
% Last Modified: 09-13-2018
% Note: This version is based on 3D U-net (detailed in our newly accepted TMI paper), 
% results shown in arXiv paper is based on 2D U-net. ---09-13-2018
%--------------------------------------------------------------------------
%% check inputs
imgsiz = Gopt.imgsiz;
imgsiz_trunc = Gopt.imgsiz_trunc;
trunc_range = Gopt.trunc_range;
numpix = prod(imgsiz);
if isempty(x0)
    x0 = ones(numpix,1);
end
% set Gopt
if isempty(maxit)
    maxit = 10;
end
%% initialization
x  = max(mean(x0(:))*1e-9,x0(:)); x(~Gopt.mask) = 0;
mythrehold = 0.1; % the threshold of the lose dose count compared with high dose, here we use 1:10 during training for the Xcat phantom.
%% Due to github file limit, save normalization files in two separate files
load('../../data/norm_firsthalf.mat', 'norm1') 
load('../../data/norm_secondhalf.mat', 'norm2') 
AN = [norm1;norm2]; clear norm1 norm2;
load('../../data/randoms.mat','R');% randoms and scatters
load('../../data/sino.mat','Y');% sinogram data
AN(isinf(AN)) = 0.0;
AN = AN ./ mythrehold;
wx = backproject_ge690(AN, Gopt); % compute senstivity image
yeps = 1e-10;
% output
if nargin>1
    out = [];
end
Mu = 0;
%% Note: This scaling number is to match the recon with my pre-training
% scaling. During pre-training, the reconstructed images are scaled to
% around 0 to 5. 
myscaling = 1./1e-3 .* mythrehold; 
%% iterative loop 
for it = 1:maxit
    if Gopt.disp
        fprintf('iteration %d\n',it);
    end  
    if it == 1
        %% first to get f(Xem_30), this is the initialization for x_0 and also for \alpha_0
        mytemp = reshape(x, imgsiz) ./ myscaling;
        myini = mytemp(trunc_range{1},trunc_range{2},trunc_range{3});
        dump(myini, 'image_firstinput.img');
        system(sprintf('CUDA_VISIBLE_DEVICES=%d python ../../initialize_residual_cnnoutput.py', device_num));
        temptest = touch('image_firstoutput.img');
        temptest = reshape(temptest, imgsiz_trunc);
        output_cnn = zeros(imgsiz);
        output_cnn(trunc_range{1},trunc_range{2},trunc_range{3}) = temptest * myscaling;
        x = output_cnn(:);
        output_temp = output_cnn;
        system('mv image_firstoutput.img image_evolve.img'); % set to \alpha_0
        %% second, to get f(\alpha_0), where \alpha_0 = x_0 = f(Xem_30)
        mytemp = reshape(output_cnn, imgsiz) ./ myscaling;
        myini =  mytemp(trunc_range{1},trunc_range{2},trunc_range{3});
        dump(myini, 'image_firstinput.img');
        system(sprintf('CUDA_VISIBLE_DEVICES=%d python ../../initialize_residual_cnnoutput.py', device_num));
        temptest = touch('image_firstoutput.img');
        temptest = reshape(temptest, imgsiz_trunc);
        output_cnn = zeros(imgsiz);
        output_cnn(trunc_range{1},trunc_range{2},trunc_range{3}) = temptest * myscaling;
        
    end
    %% save data    
    if nargout>1 & ( rem(it,Gopt.savestep)==0 || it == 1 )
        if it == 1
            cnnsave = reshape(output_temp, imgsiz); clear output_temp;
        else
            cnnsave = reshape(output_cnn, imgsiz);
        end
        save(sprintf('output_it%d.mat', it), 'cnnsave');
    end
    %% EM update
    for myiter = 1:multiple_iter
        tempfor = forwardproject_ge690(x, Gopt);
        yb = AN .* vector(tempfor) + R;
        yy = Y ./ (yb + yeps);
        yy(yb==0&Y==0) = 1;
        yy = AN .* vector(yy);
        xb = backproject_ge690(yy, Gopt);
        xx_em  = x ./ wx .* xb;
        xx_em(isnan(xx_em)) = 0;
        xx_em(~Gopt.mask) = 0;
        xx_reg = output_cnn(:) - Mu;
        x = eml_prox_sepl2(wx, xx_em, myrho, xx_reg(:)); 
    end
    
    %% Steepest descent to get the network output
    tempsum = x + Mu;
    mytemp = reshape(tempsum, imgsiz) ./ myscaling;
    x_newnetwork = mytemp(trunc_range{1},trunc_range{2},trunc_range{3});
    dump(x_newnetwork, sprintf('image_input.img'));
    myiter = subiter;
    system(sprintf('CUDA_VISIBLE_DEVICES=%d python ../../BN_unet_1.0_basedonPossion_otsp_new_residual_momentum.py %d %d', device_num, it, myiter));
    %system('rm image_input.img');
    temptest = touch('image_output.img');
    temptest = reshape(temptest, imgsiz_trunc);
    output_cnn = zeros(imgsiz);
    output_cnn(trunc_range{1},trunc_range{2},trunc_range{3}) = temptest .* myscaling;
    output_cnn = output_cnn(:);
    system('rm image_output.img');
    %% Update the dual variable
    Mu_temp = x - output_cnn;
    Mu = Mu + Mu_temp;

end

%--------------------------------------------------------------------------
function xi = eml_prox_sepl2(wi, yi, bet, zi, ri)
%--------------------------------------------------------------------------
% estimate an intermediate estimate from Poisson data
% xi = argmax wi * ( yi*log(xi+ri) - (xi+ri) ) - bet/2 * (xi-zi)^2;
%
% Programmer: Guobao Wang @ UC Davis
%

% check
if isempty(wi)
    wi = ones(size(yi));
end
if isscalar(bet)
    bet = bet*ones(size(yi));
end
if nargin<5 | isempty(ri)
    ri = zeros(size(yi));
end

% solve the quadratic equation
bi = wi - bet.*(zi+ri);
di = sqrt(bi.^2+4*bet.*wi.*yi);

xi = zeros(size(yi));
ib = bet==0;
xi(ib) = yi(ib);

ib = bet>0 & bi>0;
xi(ib) = 2*wi(ib).*yi(ib)./(di(ib)+bi(ib));

ib = bet>0 & bi<=0;
xi(ib) = (di(ib)-bi(ib))./(2*bet(ib));

xi = xi - ri;

