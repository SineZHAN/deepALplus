# DeepAL+: Deep Active Learning Toolkit

DeepAL+ is a extended toolkit originated from [DeepAL toolkit](https://github.com/ej0cl6/deep-active-learning). 
Including python implementations of the following active learning algorithms:

- Random Sampling
- Least Confidence [1]
- Margin Sampling [2]
- Entropy Sampling [3]
- Uncertainty Sampling with Dropout Estimation [4]
- Bayesian Active Learning Disagreement [4]
- Core-Set Selection [5]
- Adversarial margin [6]
- Mean Standard Deviation [7]
- Variation Ratios [8]
- Cost-Effective Active Learning [9]
- KMeans with scikit-learn library and with faiss-gpu library
- Batch Active learning by Diverse Gradient Embeddings [10]
- Loss Prediction Active Learning [11]
- Variational Adversarial Active Learning [12]
- Wasserstein Adversarial Active Learning [13]

We support 10 datasets, *MNIST*, *FashionMNIST*, *EMNIST*, *SVHN*, *CIFAR10*, *CIFAR100*, *Tiny ImageNet*, *BreakHis*, *PneumoniaMNIST*, *Waterbirds*. One can add new dataset by adding new function `get_newdataset()` in `data.py`.

*Tiny ImageNet*, *BreakHis*, *PneumoniaMNIST* need to be download manually, the corresponding data addresses could be found in `data.py`.

In DeepAL+, we use **ResNet18** as basic classifier. One can replace it to other basic classifiers and add them to `nets.py`.

## Prerequisites 

- numpy            1.21.2
- scipy            1.7.1
- pytorch          1.10.0
- torchvision      0.11.1
- scikit-learn     1.0.1
- tqdm             4.62.3
- ipdb             0.13.9
- openml           0.12.2  
- faiss-gpu        1.7.2
- toma             1.1.0
- opencv-python    4.5.5.64
- wilds            2.0.0 (for waterbirds dataset only)

You can also use the following command to install conda environment

```
conda env create -f environment.yml
```

`faiss-gpu` and wilds should use `pip install`.

## Demo 

```
  python demo.py \
      -a RandomSampling \
      -s 100 \
      -q 1000 \
      -b 100 \
      -d MNIST \
      --seed 4666 \
      -t 3 \
      -g 0
```
See `arguments.py` for more instructions.
We have also construct a comparative survey based on DeepAL+. 
Please refer [here](https://arxiv.org/pdf/2203.13450.pdf) for more details.

## Citing

If you use our code in your research or applications, please consider citing our paper.

```
@article{zhan2022comparative,
  title={A comparative survey of deep active learning},
  author={Zhan, Xueying and Wang, Qingzhong and Huang, Kuan-hao and Xiong, Haoyi and Dou, Dejing and Chan, Antoni B},
  journal={arXiv preprint arXiv:2203.13450},
  year={2022}
}
```

## Reference

[1] A Sequential Algorithm for Training Text Classifiers, SIGIR, 1994

[2] Active Hidden Markov Models for Information Extraction, IDA, 2001

[3] Active learning literature survey. University of Wisconsin-Madison Department of Computer Sciences, 2009

[4] Deep Bayesian Active Learning with Image Data, ICML, 2017

[5] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[6] Adversarial Active Learning for Deep Networks: a Margin Based Approach, arXiv, 2018

[7]  Semantic segmentation of small objects and modeling of uncertainty in urban remote sensing images using deep convolutional neural networks, CVPR, 2016

[8] Elementary applied statistics: for students in behavioral science. New
York: Wiley, 1965

[9] Cost-effective active learning for deep image classification. TCSVT, 2016

[10] Deep batch active learning by diverse, uncertain gradient lower bounds. ICLR, 2020

[11] Learning loss for active learning. CVPR, 2019

[12] Variational adversarial active learning, ICCV, 2019

[13] Deep active learning: Unified and principled method for query and training. AISTATS, 2020


## Contact

If you have any further questions or want to discuss Active Learning with me or want to contribute your own Active Learning approaches into our toolkit, please contact xyzhan2-c@my.cityu.edu.hk (my spare email is sinezhan17@gmail.com).




