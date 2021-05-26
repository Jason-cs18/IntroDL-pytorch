# Introduction to Deep Learning using PyTorch

A curated list of research in deep learning algorithms (especially on models). I also summarize classic papers and provide corresponding PyTorch implementations. Because they are pre-trained on large-scale labeled datasets (*i.e.,* ImageNet and Microsoft COCO), I just provide implementation on inference.<br>

There are  three main architectures of recent deep learning models: 1) Convolutional Neural Networks (**CNN**); 2) Recurrent Neural Networks (**RNN**); and 3) Generative Adversarial Networks (**GAN**). I will introduce them at first and then describe corresponding applications (*i.e.,* object detection using CNN). Besides, I also introduce an effective module named attention which is widely used in Natural Language Processing algorithms (*i.e.,* BERT and Transformer).<br>

## 1. Convolutional Neural Networks (CNN)

**Convolutional neural network** (**CNN**, or **ConvNet**) is a class of [deep neural network](https://en.wikipedia.org/wiki/Deep_neural_network), most commonly applied to analyze visual imagery. They are also known as **shift invariant** or **space invariant artificial neural networks** (**SIANN**), based on the shared-weight architecture of the convolution kernels or filters that slide along input features and provide translation [equivariant](https://en.wikipedia.org/wiki/Equivariant_map) responses known as feature maps.<sup>[1]</sup>

![](https://github.com/YanLu-nyu/IntroDL-pytorch/blob/master/images/cnn_arch.jpg)

1. Image Classification (AlexNet, VGG, Inception and ResNet) <sup>[2]</sup>
2. Object Detection (RCNN series, YOLOvx and SSD) <sup>[3]</sup>
3. Semantic Segmentation (FCN, U-Net, PSPNet, Mask RCNN and DeepLab) <sup>[4, 5]</sup>

## 2. Recurrent Neural Networks (RNN)

**Recurrent neural network** (**RNN**) is a class of [artificial neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) where connections between nodes form a [directed graph](https://en.wikipedia.org/wiki/Directed_graph) along a temporal sequence. This allows it to exhibit temporal dynamic behavior. <sup>[6]</sup>

![](https://github.com/YanLu-nyu/IntroDL-pytorch/blob/master/images/rnn_arch.jpg)

1. Three classic modules to capture history (RNN, LSTM and GRU)
2. Image Captioning (One-to-Many)
3. Sentiment Classification (Many-to-One)
4. Machine Translation (Many-to-Many)

## 3. Generative Adversarial Networks (GAN)

1. Convolutional GAN (DCGAN, LapGAN, ResGAN, SRGAN and CycleGAN)
2. Conditional GAN (CGAN and InfoGAN)

## Additional Effective Modules

### 1. Attention mechanism

## References

1. (Wikipedia) [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network).
2. (PIEAS) [A Survey of the Recent Architectures of Deep Convolutional Neural Networks.](https://arxiv.org/ftp/arxiv/papers/1901/1901.06032.pdf) Artificial Intelligence Review, 2020. 
3. (Oulu, NUDT, USYD, CUHK and UWaterloo) [Deep Learning for Generic Object Detection: A Survey.](https://link.springer.com/content/pdf/10.1007/s11263-019-01247-4.pdf) International Journal of Computer Vision, 2020 (IJCV).
4. (Snapchat, UWaterloo, Qualcomm, UEX, UTD and UCLA) [Image Segmentation Using Deep Learning: A Survey.](https://arxiv.org/abs/2001.05566) IEEE Transactions on *Pattern Analysis and Machine Intelligence*, 2021 (PAMI).
5. [awesome-semantic-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation)
6. (Wikipedia) [Recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
7. 

