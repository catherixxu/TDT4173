# Object classification with the full and a limited version of the CIFAR-10 data set

## Table of contents
* [General Info](#General Info)
* [Convolutional Neural Network](#CNN)
* [k-Nearest Neighbor](#kNN)

## General Info:
This repository contains the code for the two methods implemented in the project of object classification with the full and a limited version of the CIFAR-10 data set. The two methods are a Convolutional Neural Network and a k-Nearest Neighbor classifier. This project aims to investigate how sensitive these two methods are to the amount of training data. In order to compare the performance of the methods with varying amount of data, some alteration to the code needs to be made, as descirbed below.


## CNN:
Run the convolutional neural network by using [DeepNetFinal.py](src/DeepNetFinal.py)

To load the CIFAR-10 data set:
- Change «cifar_10_dir» to your location of CIFAR-10

To load the full data set:
- Use range(1,6) in line 26

To load the limited data set:
- Use range(1,2) in line 26

To run CNN with data augmentation:
- Comment out ‘’history = model_cnn.fit()’’ on line 160
- Run ‘’history = model_cnn.fit_generator()’’ on line 165 instead

## KNN:
Run the convolutional neural network by using [kNN.py](src/kNN.py)

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
