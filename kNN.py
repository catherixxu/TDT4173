import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition

#----Load Data----
# This method is collected from: 
#https://github.com/snatch59/load-cifar-10/blob/master/load_cifar_10.py?fbclid=IwAR31fr1JREYYF8OGcZE_eioCwElnTW0iGDYsQW-JNV73_zv4e-rSa4PL0P8'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
  
def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 2): #1-6
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names
 
#Set directory        
cifar_10_dir = 'C:\\Users\\amali\\Desktop\\A2\\Data\\cifar-10-batches-py'

#Collect the data into training and testing 
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

#Print the shapes, to get an idea of how the data looks like 
print("Train data: ", train_data.shape)
print("Train filenames: ", train_filenames.shape)
print("Train labels: ", train_labels.shape)
print("Test data: ", test_data.shape)
print("Test filenames: ", test_filenames.shape)
print("Test labels: ", test_labels.shape)
print("Label names: ", label_names.shape)
print(label_names)

#----Plotting images from the training data----
labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

#Convert the training labels to a list 
liste=[]
for element in train_labels: 
    liste.append(element)
       
#Plot some images     
f, axes = plt.subplots(ncols=7, nrows=10)
f.set_figheight(18)
f.set_figwidth(18)
#This list contains 7 indexes for each class (10 classes)
index_list = [29, 30, 35, 49, 77, 93, 115, 4, 5, 32, 44, 45, 46, 60, 6, 13, 18, 24, 41, 42, 47, 9, 17, 21, 26, 33, 36, 38, 3, 10, 20, 28, 34, 58, 66, 0, 19, 22, 23, 25, 72, 95, 27, 40, 51, 56, 70, 81, 83, 7, 11, 12, 37, 43, 52, 68, 8, 62, 69, 92, 100, 106, 111, 1, 2, 14, 15, 16, 31, 50]
ind=0
for i in range(10): 
    for j in range(7):
        index=index_list[ind]
        axes[i,j].set_title(labels[liste[index]]) 
        axes[i,j].imshow(train_data[index])
        axes[i,j].get_xaxis().set_visible(False)
        axes[i,j].get_yaxis().set_visible(False)
        ind += 1
plt.show()

#----Investigate training data distribution----
# Some of this code is borrowed from  
# https://www.kaggle.com/roblexnana/cifar10-with-cnn-for-beginer

fig, axs = plt.subplots(1,2,figsize=(15,5)) 
sns.countplot(train_labels.ravel(), ax=axs[0])
axs[0].set_title('Training data distribution (50 000 images)') #If the code is run with only one training batch, change this to "for 10 000 training images"
axs[0].set_xlabel('Classes')
sns.countplot(test_labels.ravel(), ax=axs[1])
axs[1].set_title('Testing data distribution')
axs[1].set_xlabel('Classes')
plt.show()

#------Find the best k-------------------------
#See the shape of the data
print(train_data.shape)
print(train_labels.shape)

#Reshape the data to fit the format of the KNeighborsClassifier (32x32x3=3072)
train_data_formated = train_data.reshape(train_data.shape[0], -1)
train_labels_formated = train_labels.reshape(train_labels.shape[0], -1)
train_labels_formated = train_labels_formated.ravel()

#Apply PCA to training data
pca = decomposition.PCA()
pca.fit(train_data_formated)
transformed = pca.transform(train_data_formated)

#Collect training data where 90% variance is explained 
cum_sum = numpy.cumsum(pca.explained_variance_ratio_, axis=0)
top90 = numpy.where(cum_sum > 0.90)[0][0]
transformed_train_90 = transformed[:, 0:top90]

#GRID SEARCH
knn = KNeighborsClassifier()       
#Looping over 1-25 neighbors  
neighbors = {'number_of_neighbors': np.arange(1, 25)}
#Gridsearch to find optimal number of neighbours by use of cross validation 
knn_gscv = GridSearchCV(knn, neighbors, cv=5)
clf = knn_gscv.fit(transformed_train_90, train_labels_formated)
print(knn_gscv.best_params_)

# -------PCA and KNN-----------
#Reshape the data to fit the format of the KNeighborsClassifier (32x32x3=3072)
#Training data
train_data_formated = train_data.reshape(train_data.shape[0], -1)
train_labels_formated = train_labels.reshape(train_labels.shape[0], -1)
train_labels_formated = train_labels_formated.ravel()

#Testing data
test_data_formated = test_data.reshape(test_data.shape[0], -1)
test_labels_formated = test_labels.reshape(test_labels.shape[0], -1)
test_labels_formated = test_labels_formated.ravel()

#Function for evalutaing the kNN
def plot_confmatrix(model, trainingdata, traininglabels, testingdata, testinglabels):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    confmatrix = plot_confusion_matrix(model, trainingdata, traininglabels, normalize='true', ax=ax)
    confmatrix.ax_.set_title('Training Set Performance');
    ax = fig.add_subplot(1, 2, 2)
    confmatrix = plot_confusion_matrix(model, testingdata, testinglabels, normalize='true', ax=ax)
    confmatrix.ax_.set_title('Test Set Performance');
    pred = model.predict(testingdata)
    print('Test Accuracy: ' + str(sum(pred == testinglabels)/len(testinglabels)))

#Appply PCA to training and test data 
pca_model = decomposition.PCA()
pca_model.fit(train_data_formated)
train_PCA90= pca_model.transform(train_data_formated)
test_PCA90= pca_model.transform(test_data_formated)

cum_sum= numpy.cumsum(pca_model.explained_variance_ratio_, axis=0)
index_90 = numpy.where(cum_sum > 0.90)[0][0]

trans_train_PCA90 = train_PCA90[:, 0:index_90]
trans_test_PCA90 = test_PCA90[:, 0:index_90]

#Run kNN on the data in the reduced PCA-space    
knn_90PCA = KNeighborsClassifier(n_neighbors=11, weights='distance')
knn_90PCA .fit(trans_train_PCA90, train_labels_formated)
plot_confmatrix(knn_90PCA , trans_train_PCA90, train_labels_formated, trans_test_PCA90, test_labels_formated)   


