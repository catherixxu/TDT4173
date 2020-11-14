# Project - Object classification with the full and alimited version of the CIFAR-10 data set
##CNN:

To load the CIFAR-10 data set:
- Change «data_dir» to your location of CIFAR-10

To load the full data set:
- Use range(1,6) in line 23

To load the limited data set:
- Use range(1,2) in line 23

To run CNN with data augmentation:
- Comment out ‘’history = model_cnn.fit()’’ on line 157
- Run ‘’history = model_cnn.fit_generator()’’ on line 162 instead

KNN:

