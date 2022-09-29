# PMCnet
## Efficient Bayes Inference in Neural Networks through Adaptive Importance Sampling
This repository contains the Pytorch implementation of the PMCnet and PMCnet-light applied to the problem of approximating the complete posterior distribution of the unknown weight and bias parameters of neural network in Bayesian neural networks (BNNs). It provides uncertainty quantification when predicting new data. Adaptive importance sampling (AIS) is one of the most prominent Monte Carlo methodologies benefiting from sounded convergence guarantees and ease for adaptation. This work aims to show that AIS constitutes a successful approach for designing BNNs. More precisely, we propose a novel algorithm PMCnet that includes an efficient adaptation mechanism, exploiting geometric information on the complex (often multimodal) posterior distribution.

## Dependencies
Python version 3.6.10\
Pytorch 1.7.0\
CUDA 11.0\
scikit-learn 0.24.2\
numpy 1.19.5

## Training
To initialize the parameters of BNN, we make use of the learnt parameters derived by MLE, which is put in the folder _params_. To start the training, run

```
CUDA_VISIBLE_DEVICE=0 python3 train_PMCnet_binray_classification.py
```

## Demo file
Here we give three examples of applying PMCnet under binay classification, multi-class classification and regression task respectively.\
train_PMCnet_binray_classification.py: shows how to train PMCnet under binay classification task for dataset _Ionosphere_\
train_PMCnet_multiclass_classification.py: shows how to train PMCnet under multi-class classification task for dataset _Glass_\
train_PMCnet_regression.py: shows how to train PMCnet under regression task for dataset _autoMPG_\
PMCnet_algo.py: the PMCnet algorithm under classification task and regression task\
PMCnet_light_algo.py: the PMCnet-light algorithm under classification task for large-scale problem\
PMCnet_light_algo_regression.py: the PMCnet-light algorithm under regression task for large-scale problem

## Authors
Yunshi Huang - e-mail: yunshi.huang@centralesupelec.fr - PhD Student\
Víctor Elvira -[website](https://victorelvira.github.io/) \
Emilie Chouzenoux -[website](https://pages.saclay.inria.fr/emilie.chouzenoux/)\
Jean-Christophe Pesquet -[website](https://jc.pesquet.eu/)
