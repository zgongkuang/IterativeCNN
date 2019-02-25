# Iterative PET Image Reconstruction Using CNN Representation
### Prerequisites
Matlab, Python 2.7, Tensorflow 1.4
### Instructions
* Step 1. 
Download the data folder 'data' from https://www.dropbox.com/sh/2s93bp45wwwbxqq/AAC4ei5646jWZaCFUFGdD0xva?dl=0
* Step 2. 
Dowload the system matrix folder 'sys_ge690_smat' from https://www.dropbox.com/sh/x4wfuz1fq3okxg9/AADE7UXYi4X-uXPwh9IPWsmva?dl=0
* Step 3. 
Run demo_iterativeCNN.m to get the iterative reconstruction results. 
### Note 
* The 3D U-net is trained and the trained model is stored in pretraining_process. If you want to re-train the model based on your own data sets, you can use Unet3D_train.py for training and Unet3D_test.py for testing. 
* The results runing on Ubuntu server setting penalty parameter rho = 7.5e-4 are uploaded for reference. 
* If you do not have training data, but have the anatomical prior image, you can check my newly published [DIPRecon method](https://ieeexplore.ieee.org/abstract/document/8581448).
### Reference
Gong, K., Guan, J., Kim, K., Zhang, X., Fakhri, G.E., Qi, J.\* and Li, Q.\*, 2017. [Iterative PET image reconstruction using convolutional neural network representation](https://arxiv.org/pdf/1710.03344.pdf). arXiv preprint arXiv:1710.03344. <br />
Gong, K., Guan, J., Kim, K., Zhang, X., Yang, J., Seo, Y.,  Fakhri, G.E., Qi, J.\* and Li, Q.\*, 2018. [Iterative PET image reconstruction using convolutional neural network representation](https://ieeexplore.ieee.org/abstract/document/8463596). IEEE Transactions on Medical Imaging
## License
This project is licensed under the 3-Clause BSD License - see the [LICENSE](LICENSE) file for details.
