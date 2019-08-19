### Research detail and notes about deeplab series :

------

#### Deeplab V3

- **Basic info:** Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.

- **Datasets:** PASCAL VOC 2012 semantic segmentation benchmark, CItyScapes, COCO, JFT, ImageNet

- **Model:**

  - Cascaded modules without and with atrous convolution to deep

    ![deeplabv3-deep conv model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/Deeplab-research/pic/deeplabv3-deep%20conv%20model.png)

  - Parallel modules with atrous convolution (ASPP), augmented with image-level features

    ![deeplabv3-deep with aspp model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/Deeplab-research/pic/deeplabv3%2B-model.png)

- **Training strategy:** 

  - Please refer to ***4.1. Training Protocol*** in paper

- **Loss function:**

  - Cross entropy loss in general

- **Evaluation:**

  - See datail in paper which do many experiments

- **Merits:**

  - Add multi_grid into network combined with ASPP
  - Add image pooling in ASPP
  - No CRF post-processing module

- **Defects:**

  - Need several pre-trained in other datasets

- **Notes:**

  - **What is multi_grid**: do different taros rate in adjacent conv layers, like rate=(1,2,4) which atrous rate=1 in the first conv layer, and 2 in second and 4 in third layer. Note multi_grid in ResNet block in paper is a little different, which means there are uint rate and block rate. Multi_grid is defined as **HDC** (hybrid dilated convolution) in [Understanding Convolution for Semantic Segmentation](https://arxiv.org/abs/1702.08502)
  - **Inference stratrgy:** multi-scale input, left-right flip and different output_stride by changing the last block's strides in the backbone

- **Related resources:**

  [**deeplabv3**](https://github.com/fregu856/deeplabv3) 

#### Deeply V3+

- **Basic info:** Chen L C, Zhu Y, Papandreou G, et al. Encoder-decoder with atrous separable convolution for semantic image segmentation[C]//Proceedings of the European conference on computer vision (ECCV). 2018: 801-818.

- **Datasets:** PASCAL VOC 2012 semantic segmentation benchmark, CItyScapes, COCO, JFT, ImageNet

- **Model:**

  ![deeplabv3+-model overview](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/Deeplab-research/pic/deeplabv3%2B-model%20overview.png)

  ![deeplabv3+-model](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/Deeplab-research/pic/deeplabv3+-model.png)

- **Training strategy:** 

  - Please refer to ***4 in paper*** and  ***4.1. Training Protocol*** in **deeplabv3** paper

- **Loss function:**

  - Cross entropy loss in general

- **Evaluation:**

  - See datail in paper which do many experiments

- **Merits:**

  - A novel encoder-decoder structure which employs DeepLabv3 as a powerful encoder module and a simple yet effective decoder module 
  - Adapt the Xception (modified aligned Xception) model for the segmentation task and apply depthwise separable convolution to both ASPP module and decoder module, resulting in a faster and stronger encoder-decoder network
  - Make comparison with trimap width in void label

- **Defects:**

  - Need several pre-trained in other datasets

- **Notes:**

  - **Image-level:** is image pooling in deeplabv3.
  - **About Trimap:** there is a width which decides the width of trimap where smaller width requires more effective and precise method. It is talked in ***4.4 Improvement along Object Boundaries*** in paper.
  - **No multi-grids:** found it does not improve the performance
  - **Lots of tricks:** please refer to paper for detail, such as ***inference stratrgy*** in deeplabv3

- **Related resources:**

  [**pytorch-deeplab-xception**](https://github.com/jfzhang95/pytorch-deeplab-xception) 
