# Object classification with the full and a limited version of the CIFAR-10 data set

## Table of contents
* [General Info](#general-info)
* [Content of repository](#content-of-repository)
* [Convolutional Neural Network](#convolutional-neural-network-CNN)
* [k Nearest Neighbor](#k-nearest-Neighbor-KNN)

## General info
This repository contains the source code for the two methods implemented in the project by group 7 in TDT4173. The two methods are a Convolutional Neural Network and a k-Nearest Neighbor classifier. This project aims to investigate how sensitive these two methods are to the amount of training data. In order to compare the performance of the methods with varying amount of data, necessary alterations to co. The data set used, CIFAR-10 python version, can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html). If the user does not have eny programming environments or python downloaded, procees with downloading [Anaconda Navigator](https://docs.anaconda.com/anaconda/navigator/). The files can be run in Spyder or similar IDE.

Note that running PCA and grid search in [kNN.py](/kNN.py) and training the network in [DeepNetFinal.py](/DeepNetFinal.py) can take extensive amount of time with regular computational power. Running on a GPU is recommended.

## Content of repository
This repository contains three files:
- **README.md**: Introduce repository and files
- **DeepNetFinal.py**: Code for the implemented Convolutional Neural Network, including load files, build model, model training and evaluation
- **kNN.py**: Code for the implemented k-Nearest Neighbor classifier, including load files, grid search, PCA, model building and evaluation


## Convolutional Neural Network CNN
Run the convolutional neural network by using [DeepNetFinal.py](/DeepNetFinal.py)

To load the CIFAR-10 data set:
- Change «cifar_10_dir» to your location of CIFAR-10

To load the full data set:
- Use range(1,6) in line 26

To load the limited data set:
- Use range(1,2) in line 26

To run CNN with data augmentation:
- Comment out ‘’history = model_cnn.fit()’’ on line 160
- Run ‘’history = model_cnn.fit_generator()’’ on line 165 instead

## k Nearest Neighbor KNN
Run the convolutional neural network by using [kNN.py](/kNN.py)

To load the CIFAR-10 data set:
- Change «cifar_10_dir» to your location of CIFAR-10

To load the full data set:
- Use range(1,6) in line 49

To load the limited data set:
- Use range(1,2) in line 49

After loading the data (line 1 until line 104), the kNN can be run by running the code below line 171. (Line 104-171 is used to plot images and to find optimal k. The grid search can be run from line 143-171, if doing so, use then the k value printed on line 170 in the following). 

To run kNN with limited data set: 
-	Set n_neighbors=11 i line 209. 

To run kNN with full data set: 
-	Set n_neighbors=8 in line 209.
