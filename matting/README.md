# Survey of image/human matting

- A nearest project&paper produced by Alimama called [Semantic_Human_Matting (SHM)](https://github.com/lizhengwei1992/Semantic_Human_Matting) paper in ACMMM. **SHM** is the first algorithm that learns to jointly fit both semantic information and high quality details with deep networks. (alpha matte)
- And one of the human matting datasets: [Human Matting datasets](https://github.com/aisegmentcn/matting_human_datasets) 
- And another useful repo for mobile devices with NCNN tool: [And mobile_phone_human_matting](https://github.com/lizhengwei1992/mobile_phone_human_matting) (including [datasets](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/tree/master/dataset) ) 
- Another latest or s-o-t-a paper in matting:
  - **A Late Fusion CNN for Digital Matting, CVPR2019.** 
  - Inductive Guided Filter: Real-time Deep Image Matting with Weakly Annotated Masks on Mobile Device, arXiv 2019. Need weakly annotated masks.
  - **2016_Automatic Portrait Segmentation for Image Stylization_CGF** 
  - 2016_Deep Automatic Portrait Matting_ECCV_spotlight
  - 2017_Deep Image Matting_CVPR, oral, need trimap
  - **2017_Fast Deep Matting for Portrait Animation on Mobile Phone_ACMMM,** [github-pytorch](https://github.com/huochaitiantang/pytorch-fast-matting-portrait) 
  - **2019_Towards Real-Time Automatic Portrait Matting on Mobile Devices,**  [GitHub-pytorch](https://github.com/hyperconnect/MMNet) 
- 2019_Learning-based Sampling for Natural Image Matting_CVPR, need trimap
- There are two related challenges maybe useful for comparing relevent methods.
  - [Image Matting](http://www.alphamatting.com/eval_25.php) 
  - [Video Matting](http://videomatting.com/) 
- The almost traditional methods in matting field **need user interactive**: input image and trimap or scribble and output the alpha matte. It is not feasible for us to use. But there are several methods based on deep learning which can end-to-end training and **just for one input image** to output alpha matte. 



## Paper in detail

### Semantic human matting

- **Basic info:** Chen Q, Ge T, Xu Y, et al. Semantic human matting[C]//2018 ACM Multimedia Conference on Multimedia Conference. ACM, 2018: 618-626.

- **Datastes:** self-create (features: humans with some accessories e.g. cellphones, handbags, e-commerce) & DIM ([Deep Image Matting](https://sites.google.com/view/deepimagematting)) 

  ![SHM-datasets](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/SHM-datasets.png)

- **Model:** two sub-networks: T-Net (can be random existing model) and M_Net, final fusion module by mathematical equation

  ![SHM_Model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/SHM_Model.png)

- **Training strategy:** 

  - 1) pre-trained T-Net which is PSPNet-50 based on ResNet-50, initialize relevant layers with off-the-shelf model trained on ImageNet classification task and randomly initialize the rest layers. With data augmented.

  - 2) pre-trained M-Net use entire DIM datasets. With data augmented.

  - 3) end-to-end training jointly. With data augmented.

- **Loss function:** alpha prediction loss, compositional loss and a classification loss.

  ![SHM-loss fuc](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/SHM-loss%20fuc.png)

- **Evaluation:** and real-image testing and visualizing 

  SAD: Sum of Absolute Difference

  MSE: Mean Squared Error

  Gradient and Connectivity is defined in the paper [A Perceptually Motivated Online Benchmark for Image Matting](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/01/cvpr09-matting-Eval_TR.pdf) , Gradient is a convolution derivatives result by with first-order Gaussian derivative filters with variance 1.4.

  ![SHM-evaluation](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/SHM-evaluation.png)

- **Merits:**

  - Only one image as the input without the constraints like trimp or scribble
  - End to end training with two sub-networks and fusion module
  - New loss function conbined by three part

- **Defects:**

  - Author's source codes was not opened
  - The model is hard to train apparently 
  - No mIOU metrics

- **Related resources:**

  - [github-reimplement-not-authors](https://github.com/lizhengwei1992/Semantic_Human_Matting) 

### Towards Real-Time Automatic Portrait Matting on Mobile Devices

- **Basic info:** Seo S, Choi S, Kersner M, et al. Towards Real-Time Automatic Portrait Matting on Mobile Devices[J]. arXiv preprint arXiv:1904.03816, 2019.

- **Datasets:** [Deep automatic portrait matting](http://www.cse.cuhk.edu.hk/~leojia/projects/automatting/) which consists of 2,000 images of 600 × 800 resolution where 1,700 and 300 images are split as training and testing set respectively.

- **Model:** Standard encoded-deconde structure which includs successive encoder, refinement, decoder and enhancement. Almostly the depth-wise separablconvolution is applied in every part. 

  ![MMNet-model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/MMNet-model.png)

  ![MMNet-model-detail](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/MMNet-model-detail.png)

- **Training strategy:** New loss function includes five part with data augmented, use Adam optimizer with a batch size of 32 and a fixed learning rate of 1 × 10−4. Input images were resized to 128 × 128 and 256×256. Weight decays were set to 4 × 10−7. 

- **Loss function:** alpha prediction loss, compositional loss, KL divergence loss, an auxiliary loss helps with the gradient flow by including an additional KL divergence loss between the downsampled ground truth mask and the output of the encoder block #10 and gradient loss.

  - Prediction and compositional loss:

  ​                                ![MMNet-loss fun1](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/MMNet-loss%20fun1.png) 

  - KL divergence loss:

  ![MMNet-loss fun2](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/MMNet-loss%20fun2.png)

  - Gradient loss and total loss:

  ![MMNet-loss fun3](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/MMNet-loss%20fun3.png)

- **Evaluation:** Include Pixel 1, Xiaomi Mi 5, and iPhone 8 in the paper supplementary.

  ![MMNet-evaluaion](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/MMNet-evaluaion.png)

- **Merits:**

  - New loss function with five parts
  - Real-time inference with the encode-decode structure
  - Experiments on mobile phone and use tf-lite include quantization
  - It seems easy to train and there is author's open codes including a mp4 demo

- **Defects**

  - No mIOU metrics
  - The mp4 demo is not very well: the edge problems

- **Related resources:**

  - [author-github](https://github.com/hyperconnect/MMNet)

### A Late Fusion CNN for Digital Matting

- **Basic info:** Zhang Y, Gong L, Fan L, et al. A Late Fusion CNN for Digital Matting[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 7469-7478.

- **Datasets:** 

  - Human matting dataset: 28610 for training based on DIM and 1000 for testing including self-collected and part of DIM ([Deep Image Matting](https://sites.google.com/view/deepimagematting))
  - Natural dataset: 431x100 for training and 1000 for testing based on DIM ([Deep Image Matting](https://sites.google.com/view/deepimagematting))

- **Model:** One encoder, two decoder and one fusion network (I think it is like a refine module through the alpha matte results by foreground and backfround decoder). The supplementary details the network.

  ![A Late Fusion CNN for Digital Matting-model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/A%20Late%20Fusion%20CNN%20for%20Digital%20Matting-model.png)

- **Training strategy:**

  - 1) Use DenseNet-201 network pre-trained with ImageNet-1K as encoder backbone
  - 2) Firstly, pre-train segmentation network for 15 epochs. Secondly, freeze the segmentation stage and train the fusion stage alone for 4 epochs. Finally, perform the end-to-end joint training for 7 epochs.
  - 3) All batch normalization layers are frozen in the joint training step to save the memory footprint.
  - 4) Cyclical learning rate strategy, data augmented and different new loss function in every stage applied to perform a good result.

- **Loss function:** two branchs segmentation network loss through the L1 loss, the L2 loss, and the cross-entropy loss, fusion loss and joint loss.

  - Two branch segmentation loss like foreground loss:

  ![A Late Fusion CNN for Digital Matting-loss fun1](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/A%20Late%20Fusion%20CNN%20for%20Digital%20Matting-loss%20fun1.png)

  ![A Late Fusion CNN for Digital Matting-loss fun2](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/A%20Late%20Fusion%20CNN%20for%20Digital%20Matting-loss%20fun2.png)

  - Fusion network loss:

  ![A Late Fusion CNN for Digital Matting-loss fun3](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/A%20Late%20Fusion%20CNN%20for%20Digital%20Matting-loss%20fun3.png)

  - Joint loss:

  ![A Late Fusion CNN for Digital Matting-loss fun4](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/A%20Late%20Fusion%20CNN%20for%20Digital%20Matting-loss%20fun4.png)

- **Evaluation:** On human image matting testing dataset and composition-1k testing dataset.

  ![A Late Fusion CNN for Digital Matting-evaluation1](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/A%20Late%20Fusion%20CNN%20for%20Digital%20Matting-evaluation1.png)

  ![A Late Fusion CNN for Digital Matting-evaluation2](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/A%20Late%20Fusion%20CNN%20for%20Digital%20Matting-evaluation2.png)

- **Merits:**

  - New loss function for each sub-networks
  - A fusion network like a refine module to improve the performance 
  - Just one input image can address the matting problem

- **Defects:**

  - No mIOU metrics
  - It's not easy to train
  - There is no open codes now (it will open after the patent is filed according to author's statement)

- **Related resources:**

  - [github-offical](https://github.com/yunkezhang/FusionMatting) 

### Deep Automatic Portrait Matting

- **Basic info:** Shen X, Tao X, Gao H, et al. Deep automatic portrait matting[C]//European Conference on Computer Vision. Springer, Cham, 2016: 92-107.

- **Datasets:** self-created 2000 images.

- **Model:** Image as the input and shape mask which can be auto-processed.

  ![Deep Automatic Portrait Matting-model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/Deep%20Automatic%20Portrait%20Matting-model.png)

- **Evaluation:**

  ![Deep Automatic Portrait Matting-evaluation](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/Deep%20Automatic%20Portrait%20Matting-evaluation.png)

- **Merits:**

  - First method based on CNN to ouput alpha matte without user interaction
  - Use shape mask help reduce the error
  - A formulation for CNN output trimap to produce the alpha matte instead of learning trimap directly.

- **Defects:**

  - No official implementation 

  - The datasets must be required by emailing author
  - It nedds extra tools to make shape mask

- **Related resources:**

  - [not-official-github](https://github.com/takiyu/portrait_matting) 
  - [paper's websites](http://www.cse.cuhk.edu.hk/~leojia/projects/automatting/) 

### Automatic Portrait Segmentation for Image Stylization

- **Basic info:** Shen X, Hertzmann A, Jia J, et al. Automatic portrait segmentation for image stylization[C]//Computer Graphics Forum. 2016, 35(2): 93-102.

- **Datasets:** Self-cerated including 1800 images and 1500 image training, 300 image testing/validation.

- **Model:** After inputing image, it will change to 3 parts as the CNN input. 

  ![Automatic Portrait Segmentation for Image Stylization-model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/Automatic%20Portrait%20Segmentation%20for%20Image%20Stylization-model.png)

- **Evaluation:**

  ![Automatic Portrait Segmentation for Image Stylization-evaluation](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/Automatic%20Portrait%20Segmentation%20for%20Image%20Stylization-evaluation.png)

- **Merits:**

  - Extend the FCN-8s framework and make comparison including mIOU
  - Add extra informations: mean mask and normalized x,y with input image to improve effect
  - Talked about its applications and public its datasets
  - Open source code

- **Defects:**

  - Need a lot of pre-processing to make extra information except image

- **Related resources:**

  - [github-not-official](https://github.com/PetroWu/AutoPortraitMatting) 
  - [paper's websites](http://xiaoyongshen.me/webpage_portrait/index.html)

### Deep Image Matting

- **Basic info:** Xu N, Price B, Cohen S, et al. Deep image matting[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 2970-2979.

- **Datasets:** Self-created including 493 unique foreground objects and 49,300 images (N = 100) while our
  testing dataset has 50 unique objects and 1000 images (N = 20). And N is background image number.

- **Model:** Two parts: encode-decode network and refine module.

  ![DIM-model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/DIM-model.png)

- **Evaluation:** 

  ![DIM-evaluation](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/matting/pictures/DIM-evaluation.png)

- **Merits:**

  - Contribution of new datasets
  - First to demonstrate the ability to learn an alpha matte end-to-end given an image and trimap. Don't need to constrain one NN for producing trimap.
  - It can be conbined other method which can output trimap.

- **Defects:**

  - Need trimap as the input 

- **Related resources:**

  - [not-official-github](https://github.com/Joker316701882/Deep-Image-Matting) 
  - [paper's websites](https://sites.google.com/view/deepimagematting) 
