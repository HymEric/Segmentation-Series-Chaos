# Segmentation-Series-Chaos

### Summary of "2019 Survey on semantic segmentation using deep learning techniques_Neurocomputing"

------

| model／year                  | para   | infer time (ms) | FLOPs               | accuracy (VOC2012 /COCO /Cityscapes : %) | paper                                                        | code                                                |
| ---------------------------- | ------ | --------------- | :------------------ | ---------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| FCN-8s/2015                  | ～134M | 175             | -                   | 67.20/-/65.30                            | Fully Convolutional Networks for Semantic Segmentation       | https://github.com/shelhamer/fcn.berkeleyvision.org |
| PSPNet／2017                 | 65.7M  | -               | -                   | 85.40/-/80.20                            | Pyramid Scene Parsing Network                                | https://github.com/hszhao/PSPNet                    |
| DeepLab V3-JFT／2017         |        |                 |                     | 86.9/-/-                                 | Rethinking Atrous Convolution for Semantic Image Segmentation | https://github.com/rishizek/tensorflow-deeplab-v3   |
| DeepLab V3/2017              |        |                 |                     | 85.7/-/81.3                              | Rethinking Atrous Convolution for Semantic Image Segmentation | https://github.com/rishizek/tensorflow-deeplab-v3   |
| DeepLab V3+Xception/2018     |        |                 |                     | 87.8/-/82.1                              | Encoder-decoder with atrous separable convolution for semantic image segmentation | https://github.com/fyu/dilation                     |
| DeepLab V3+Xception-JFT/2018 |        |                 |                     | 89.0/-/-                                 | Encoder-decoder with atrous separable convolution for semantic image segmentation | https://github.com/fyu/dilation                     |
| ESPNet/2018                  | 0.364M |                 |                     | 63.01/-/60.2                             | SPNet-Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation | https://github.com/sacmehta/ESPNet                  |
| FC-DRN-P-D + ST/2018         | 3.9M   |                 |                     | CamVid:69.4                              | On the iterative refinement of densely connected representation levels for semantic segmentation | https://github.com/ArantxaCasanova/fc-drn           |
| ERFNet/2018                  | ~ 2.1M | 24              |                     | -/-/69.7                                 | ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation | https://github.com/Eromera/erfnet                   |
| RefineNet/2017               |        |                 |                     | 83.40/-/73.60                            | RefineNet-Multi-Path Refinement Networks for High-Resolution Semantic Segmentation | https://github.com/guosheng/refinenet               |
| FastFCN/2019                 |        |                 |                     | Pascal Context: 53.1, ADE20K: 44.34      | FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation | https://github.com/wuhuikai/FastFCN                 |
| Fast-SCNN/2019               | 1.11M  |                 |                     | -/-/68.0                                 | Fast-SCNN: Fast Semantic Segmentation Network                | https://github.com/kshitizrimal/Fast-SCNN           |
| efficient                    |        |                 | 3.05G 640 × 360 × 3 | -/-/70.33                                | An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions | https://github.com/sercant/mobile-segmentation      |

- An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions

![image](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/efficient.png)

- In term of VOC and cityscapes, **deeplab V3/V3+ is the best** from the related leaderboarder: [VOC2012](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6) , [Cityspaces](https://www.cityscapes-dataset.com/benchmarks/) and [https://paperswithcode.com/task/semantic-segmentation](https://paperswithcode.com/task/semantic-segmentation) 

  ![Survey on semantic segmentation using deep learning techniques](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/Survey%20on%20semantic%20segmentation%20using%20deep%20learning%20techniques.png)

- Good advice of mobile devices: less than 2 GFLOPs from AI in RTC challenge group.

- Google‘s solution in [Mobile Real-time Video Segmentation](http://ai.googleblog.com/2018/03/mobile-real-time-video-segmentation.html)

  from [视频分割在移动端的算法进展综述](https://zhuanlan.zhihu.com/p/60621619) * includeing some other method

- Greate tools for implementing segmentation model easily : [Semantic Segmentation Suite in TensorFlow. Implement, train, and test new Semantic Segmentation models easily!](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) 

- National University of Singapore and Best Student Paper Award at ACM MM 2018 about multi-human-parsing  [Official Repository for Multi-Human-Parsing (MHP)](https://github.com/ZhaoJ9014/Multi-Human-Parsing) 

- Similar project in GitHub about human segmetation: [Human-Segmentation-PyTorch](https://github.com/AntiAegis/Human-Segmentation-PyTorch) 

  ![Human-Segmentation-PyTorch](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/Human-Segmentation-PyTorch.png)

- A nearest project&paper produced by Alimama called [Semantic_Human_Matting (SHM)](https://github.com/lizhengwei1992/Semantic_Human_Matting) paper in ACMMM. SHM is the first algorithm that learns to jointly fit both semantic information and high quality details with deep networks. (alpha matte)

  And one of the human matting datasets: [Human Matting datasets](https://github.com/aisegmentcn/matting_human_datasets) 

  And another useful repo for mobile devices with NCNN tool: [And mobile_phone_human_matting](https://github.com/lizhengwei1992/mobile_phone_human_matting) (including [datasets](https://github.com/lizhengwei1992/Fast_Portrait_Segmentation/tree/master/dataset) ) 

  Another latest or s-o-t-a paper in matting:

  - A Late Fusion CNN for Digital Matting, CVPR2019.
  - Inductive Guided Filter: Real-time Deep Image Matting with Weakly Annotated Masks on Mobile Device, arXiv 2019.
  - 2016_Automatic Portrait Segmentation for Image Stylization_CGF
  - 2017_Deep Image Matting_CVPR
  - 2017_Fast Deep Matting for Portrait Animation on Mobile Phone_ACMMM, [github-pytorch](https://github.com/huochaitiantang/pytorch-fast-matting-portrait) 

- The largest and popular collection of semantic segmentation: [**awesome-semantic-segmentation**](https://github.com/mrgloom/awesome-semantic-segmentation) which includes many useful resources e.g. architecture, benchmark, datasets, results of related challenge, projects et.al.

- A blog conclusion about image semantic segmentation [Review of Deep Learning Algorithms for Image Semantic Segmentation](https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57) 

  ![Review of Deep Learning Algorithms for Image Semantic Segmentation](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/Review%20of%20Deep%20Learning%20Algorithms%20for%20Image%20Semantic%20Segmentation.png)

------

### Updated from 20190710:

- Latested **lightweight** model maybe useful: [**mobileNetV3**](https://arxiv.org/abs/1905.02244) (*First Submitted on 6 May 2019*) and [**efficientNet**](https://arxiv.org/abs/1905.11946) (*First Submitted on 28 May 2019*) using NAS (Neural Architectures Search) techs.

- An **useful algorithm** CVPR2019 about how to use knowledge distillation to improve accuracy of lightweight semantic segmentation models without increasing the params size and GFlops: [Structured Knowledge Distillation for Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Structured_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2019_paper.pdf) proposed by microsoft research asia.

  ![Structured Knowledge Distillation for Semantic Segmentation](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/Structured%20Knowledge%20Distillation%20for%20Semantic%20Segmentation.png)

- New upsampling method called DUpsample: the **W** can be learned and a speciall feature fusion tech like inverted fusion decreases the compuation greatly. **It outperform deeplabv3+ but only 30% computation.** [Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation](https://arxiv.org/abs/1903.02120) CVPR2019

  ![Decoders Matter for Semantic Segmentation-Data-Dependent Decoding Enables Flexible Feature Aggregation](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/Decoders%20Matter%20for%20Semantic%20Segmentation-Data-Dependent%20Decoding%20Enables%20Flexible%20Feature%20Aggregation.png)

- Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation

  ![Auto-deeplab](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/Auto-deeplab.png)

- Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network

  ![ESPNetv2](https://github.com/HymEric/Segmentation-Series-Chaos/blob/master/pictures/ESPNetv2.png)

- 



#### Semantic segmentation research in CVPR2019

| model        | para   | Infer time (ms) | GFlops                  | accuracy (VOC2012 /COCO /Cityscapes %) | paper                                                        | code                                                         | more                                                         |
| ------------ | ------ | --------------- | ----------------------- | -------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DFANet       | 7.8M   | 10              | 3.4G (input 1024x 1024) | -/-/71.3 CamVid: 64.7                  | DFANet：Deep Feature Aggregation for Real-Time Semantic Segmentation | https://github.com/Tramac/awesome-semantic-segmentation-pytorch | Proposed  by Beijing Megvii Co., Ltd.                        |
| Auto-DeepLab | 44.42M |                 |                         | 85.6/-/82.1                            | Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation | https://github.com/tensorflow/models/tree/master/research/deeplab | NAS,  less computa-tion than deeplap, Li feifei, TensorFLow applied, **oral** |
| ESPnetV2     | ~ 6M   |                 |                         | 68.0/-/66.2                            | Espnetv2: A light-weight, power efficient, and general purpose convolutional neural network | https://github.com/sacmehta/ESPNetv2                         |                                                              |
| Improving    |        |                 |                         | -/-/83.5 CamVid: 81.7                  | Improving Semantic Segmentation via Video Propagation and Label Relaxation | https://nv-adlr.github.io/publication/2018-Segmentation      | [video](https://www.bilibili.com/video/av46912559/) ,**oral**, |
|              |        |                 |                         |                                        |                                                              |                                                              |                                                              |



------

### TODO:

[done] Semantic segmentation research in CVPR2019

[*] matting in detail
