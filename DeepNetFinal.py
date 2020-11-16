import numpy as np
import matplotlib.pyplot as plt
import pickle

#The methods for loading CIFAR-10 is collected from:
#https://github.com/snatch59/load-cifar-10/blob/master/load_cifar_10.py?fbclid=IwAR31fr1JREYYF8OGcZE_eioCwElnTW0iGDYsQW-JNV73_zv4e-rSa4PL0P8

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data,
    test_filenames, test_labels
    """

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)
    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []
    for i in range(1, 6): #Change to for i in range(1,2) for getting the limited data set
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
            cifar_train_filenames = cifar_train_data_dict[b'filenames']
            cifar_train_labels = cifar_train_data_dict[b'labels']
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
    return cifar_train_data, cifar_train_filenames, cifar_train_labels, cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names

cifar_10_dir = "/Users/marenlarsen/Desktop/CIFAR"
train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

#train_data = train_data/255
#test_data = test_data/255

#Printing some of the images
num_plot = 10
f, ax = plt.subplots(num_plot, num_plot)
for m in range(num_plot):
    for n in range(num_plot):
        idx = np.random.randint(0, train_data.shape[0])
        ax[m, n].imshow(train_data[idx])
        ax[m, n].get_xaxis().set_visible(False)
        ax[m, n].get_yaxis().set_visible(False)
f.subplots_adjust(hspace=0.1)
f.subplots_adjust(wspace=0)
plt.show()

#Checking the shape of the training and testing sets
print("Train data: ", train_data.shape)
print("Train filenames: ", train_filenames.shape)
print("Train labels: ", train_labels.shape)
print("Test data: ", test_data.shape)
print("Test filenames: ", test_filenames.shape)
print("Test labels: ", test_labels.shape)
print("Label names: ", label_names.shape)


import numpy as np
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
tf.keras.backend.clear_session()

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

'''
Code for data augmentation
'''
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
datagen.fit(train_data)


num_classes = 10

'''
Building the network
'''
def build_model(num_classes):
    inputs = keras.Input(shape=(32,32,3, ), name='img')
    x = layers.Conv2D(filters=32, kernel_size=(3,3), kernel_regularizer=l2(0.001), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=32, kernel_size=(3,3), kernel_regularizer=l2(0.001), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), kernel_regularizer=l2(0.001), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=64, kernel_size=(3,3), kernel_regularizer=l2(0.001), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), kernel_regularizer=l2(0.001), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=128, kernel_size=(3,3), kernel_regularizer=l2(0.001), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Flatten()(x)
    #Dense Layers
    x = layers.Dense(128, kernel_regularizer=l2(0.001), activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    #Building the model and print summary
    model_cnn = keras.Model(inputs=inputs, outputs=outputs, name='CNN_CIFAR')
    
    return model_cnn

model_cnn = build_model(10)
model_cnn.summary()

opt = SGD(lr=0.001, momentum=0.9)
model_cnn.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=opt,
              metrics=['accuracy'])

'''
Run on of the history blocks below. 
The first is without data augmentation, and the second is with data augmentation. 
'''
history = model_cnn.fit(train_data, train_labels,
                        batch_size=64,
                        epochs=200,
                        validation_data=(test_data, test_labels))

#history = model_cnn.fit_generator(datagen.flow(train_data, train_labels, batch_size=64),
                                  #steps_per_epoch=train_data.shape[0] // 64, 
                                  #epochs=300,
                                  #validation_data=(test_data, test_labels),
                                  #verbose=1)
                                  

'''
Plotting accuracy and loss
'''
fig = plt.figure(figsize=[20, 6])
ax = fig.add_subplot(1, 2, 1)
ax.plot(history.history['loss'], label="Training Loss")
ax.plot(history.history['val_loss'], label="Validation Loss")
ax.legend()

ax = fig.add_subplot(1, 2, 2)
ax.plot(history.history['accuracy'], label="Training Accuracy")
ax.plot(history.history['val_accuracy'], label="Validation Accuracy")
ax.legend()

'''
Making a confusion matrix
'''
def evaluate_model(model, x_test, y_test):
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])

    pred = model.predict(x_test);
    indexes = tf.argmax(pred, axis=1)

    cm = confusion_matrix(y_test, indexes)
    fig = plt.figure(figsize=[40, 12])
    ax = fig.add_subplot(1, 2, 1)
    c = ConfusionMatrixDisplay(cm, display_labels=range(10))
    c.plot(ax = ax)

evaluate_model(model_cnn, test_data, test_labels)