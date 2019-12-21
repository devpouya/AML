from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
import h5py
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
import keras
import numpy as np
import tensorflow as tf
import random as rn
import pandas as pd
import keras
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# for CNNs
# from keras.layers import Conv2D, MaxPooling2D

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    fig.savefig(title)
    return ax


np.set_printoptions(precision=2)
def make_training_data_for_subject(data_input,subject,training_or_test='training'):
    
    file_name_eeg1 = 'extracted_features_eeg1_'
    file_name_eeg2 = 'extracted_features_eeg2_'
    file_name_emg = 'extracted_features_emg_'

    height = 24
    breadth = 32
    num_subjects = len(subject) * 21600
    for ind, sub in enumerate(subject):
        if(training_or_test in ['training']):
            hf_eeg1 = h5py.File(file_name_eeg1 + str(sub) + '.h5', 'r')
            data_eeg1 = np.array(hf_eeg1.get('data'))

            hf_eeg2 = h5py.File(file_name_eeg2 + str(sub) + '.h5', 'r')
            data_eeg2 = np.array(hf_eeg2.get('data'))

            hf_emg = h5py.File(file_name_emg + str(sub) + '.h5', 'r')
            data_emg = np.array(hf_emg.get('data'))
        else:
            hf_eeg1 = h5py.File(file_name_eeg1 + str(sub) + '_test.h5', 'r')
            data_eeg1 = np.array(hf_eeg1.get('data'))

            hf_eeg2 = h5py.File(file_name_eeg2 + str(sub) + '_test.h5', 'r')
            data_eeg2 = np.array(hf_eeg2.get('data'))

            hf_emg = h5py.File(file_name_emg + str(sub) + '_test.h5', 'r')
            data_emg = np.array(hf_emg.get('data'))





        for i in [0]:
            data_input[21600 * ind + i,:,0:breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,0] = data_eeg1[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,0] = data_eeg1[i+2].copy()

            data_input[21600 * ind + i,:,0:breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,1] = data_eeg2[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,1] = data_eeg2[i+2].copy()

            data_input[21600 * ind + i,:,0:breadth,2] = data_emg[i].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,2] = data_emg[i].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,2] = data_emg[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,2] = data_emg[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,2] = data_emg[i+2].copy()


        for i in [1]:
            data_input[21600 * ind + i,:,0:breadth,0] = data_eeg1[i-1].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,0] = data_eeg1[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,0] = data_eeg1[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,0] = data_eeg1[i+2].copy()

            data_input[21600 * ind + i,:,0:breadth,1] = data_eeg2[i-1].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,1] = data_eeg2[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,1] = data_eeg2[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,1] = data_eeg2[i+2].copy()

            data_input[21600 * ind + i,:,0:breadth,2] = data_emg[i-1].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,2] = data_emg[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,2] = data_emg[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,2] = data_emg[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,2] = data_emg[i+2].copy()

        for i in range(2,21600-2):
            data_input[21600 * ind + i,:,0:breadth,0] = data_eeg1[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,0] = data_eeg1[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,0] = data_eeg1[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,0] = data_eeg1[i+2].copy()

            data_input[21600 * ind + i,:,0:breadth,1] = data_eeg2[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,1] = data_eeg2[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,1] = data_eeg2[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,1] = data_eeg2[i+2].copy()

            data_input[21600 * ind + i,:,0:breadth,2] = data_emg[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,2] = data_emg[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,2] = data_emg[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,2] = data_emg[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,2] = data_emg[i+2].copy()

        for i in [21600-2]:
            data_input[21600 * ind + i,:,0:breadth,0] = data_eeg1[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,0] = data_eeg1[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,0] = data_eeg1[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,0] = data_eeg1[i+1].copy()

            data_input[21600 * ind + i,:,0:breadth,1] = data_eeg2[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,1] = data_eeg2[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,1] = data_eeg2[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,1] = data_eeg2[i+1].copy()

            data_input[21600 * ind + i,:,0:breadth,2] = data_emg[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,2] = data_emg[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,2] = data_emg[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,2] = data_emg[i+1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,2] = data_emg[i+1].copy()

        for i in [21600-1]:
            data_input[21600 * ind + i,:,0:breadth,0] = data_eeg1[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,0] = data_eeg1[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,0] = data_eeg1[i].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,0] = data_eeg1[i].copy()

            data_input[21600 * ind + i,:,0:breadth,1] = data_eeg2[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,1] = data_eeg2[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:3*breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,1] = data_eeg2[i].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,1] = data_eeg2[i].copy()

            data_input[21600 * ind + i,:,0:breadth,2] = data_emg[i-2].copy()
            data_input[21600 * ind + i,:,breadth:2*breadth,2] = data_emg[i-1].copy()
            data_input[21600 * ind + i,:,2*breadth:2*breadth+16,2] = data_emg[i,:,0:16].copy()
            data_input[21600 * ind + i,:,2*breadth+16:3*breadth,2] = data_emg[i,:,0:16].copy()
            data_input[21600 * ind + i,:,3*breadth:4*breadth,2] = data_emg[i-1].copy()
            data_input[21600 * ind + i,:,4*breadth:5*breadth,2] = data_emg[i-1].copy()
        
    return data_input
    
label_path = "train_labels.csv"
df_labels = pd.read_csv(label_path)

labels = df_labels.values
labels = labels[:,1]

def count_instances(labels):
    unique, counts = np.unique(labels,return_counts=True)
    return dict(zip(unique,counts))

class_counts = count_instances(labels)

# print("Class 0 Counts: {}".format(class_counts[0]))
print("Class 1 Counts: {}".format(class_counts[1]))
print("Class 2 Counts: {}".format(class_counts[2]))
print("Class 3 Counts: {}".format(class_counts[3]))


import matplotlib.pyplot as plt
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

data = np.zeros((64800, 24, 160, 3))
data = make_training_data_for_subject(data_input=data, subject=[0,1,2],training_or_test='training')


from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def balanced_acc(y_true, y_pred):
    return [K.mean([
        (y_true[pred_idx] == K.round(y_pred_single)) for pred_idx, y_pred_single in enumerate(y_pred)
      if y_true[pred_idx] == class_label
                    ]) for class_label in keras.utils.to_categorical([0,1,2])]

def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

# def minortiy_acc(y_true, y_pred):


print("check some entries")
# print(data[0,0,20,0:3])
# print((labels-1)[0:20])


# starting to build their setup
def make_CNN():
    model = Sequential()

    model.add( MaxPooling2D(pool_size=(3, 2), strides=(3,2), padding='valid', data_format=None, input_shape= (24, 160, 3)) )

    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='valid', data_format=None,
                     dilation_rate=(1, 1), activation='relu', use_bias=True,
                     kernel_initializer='he_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                     kernel_constraint=None, bias_constraint=None,
                     )
             )

    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None) )

    model.add(Dense(512,
                    activation='relu',
                    kernel_initializer='he_uniform' ))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512,
                    activation='relu',
                    kernel_initializer='he_uniform' ))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    opt = keras.optimizers.adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.99, amsgrad=False)
    model.compile(optimizer=opt, loss=focal_loss(alpha=.25, gamma=2), metrics=['accuracy'])
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # TODO: The weight decay rates of first and second moment were set to 0.9 and 0.99
    # TODO: cost-sensitve loss as in paper
    return model

# build a simple CNN, wrong metric used at the moment
def make_CNN_small():
    model = Sequential()

    #add model layers
    # model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (24,160,3)))
    model.add( MaxPooling2D(pool_size=(3, 2), strides=(3,2), padding='valid', data_format=None, input_shape= (24, 160, 3)) )

    model.add(Conv2D(64, kernel_size=3, activation='relu'
                     # , input_shape = (24,160,3)
                     ))

    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None) )
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))

    model.add(Dense(1024,
                    activation='relu',
                    kernel_initializer='he_uniform' ))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.adam(learning_rate= 0.0005, beta_1=0.9, beta_2=0.99, amsgrad=False)
    # model.compile(optimizer=opt, loss=focal_loss(alpha=.25, gamma=2), metrics=['accuracy', f1_m])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1_m])

    return model

def make_CNN_mathi():
    model = Sequential()

    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape= (24, 160, 3)))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.99, amsgrad=False)
    # model.compile(optimizer='adam', loss=focal_loss(alpha=.25, gamma=2), metrics=['accuracy', f1_m])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1_m])

    return model

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=3):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    # x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def make_resnet():
    model = resnet_v1((24, 160, 3), 8, 3)
    opt = keras.optimizers.adam(learning_rate= 0.0005, beta_1=0.9, beta_2=0.99, amsgrad=False)
    # model.compile(optimizer=opt, loss=focal_loss(alpha=.25, gamma=2), metrics=['accuracy', f1_m])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy', f1_m])
    return model

class CustomImagesGenerator:
    def __init__(self, x, y,
                 # zoom_range, shear_range, rescale, horizontal_flip,
                 batch_size):
        self.x = x
        self.y = y
        # self.zoom_range = zoom_range
        # self.shear_range = shear_range
        # self.rescale = rescale
        # self.horizontal_flip = horizontal_flip
        self.batch_size = batch_size
        self.__img_gen = ImageDataGenerator()
        self.__batch_index = 0

    def __len__(self):
        # steps_per_epoch, if unspecified, will use the len(generator) as a number of steps.
        # hence this
        return np.floor(self.x.shape[0] / self.batch_size)

    def next(self):
        return self.__next__()

    def __next__(self):
        start = self.__batch_index * self.batch_size
        stop = start + self.batch_size

        # print("start = {}".format(start))
        # print("stop = {}".format(stop))
        # print("self.__batch_index = {}".format(self.__batch_index))

        self.__batch_index += 1
        self.__batch_index = int(self.__batch_index % (self.x.shape[0] / self.batch_size))
        if stop > len(self.x):
            raise StopIteration
            # self.__batch_index = 0
        # else:
        transformed = np.array(self.x[start:stop])  # loads from hdf5
        labels = np.array(self.y[start:stop])
        # for i in range(len(transformed)):
        #     zoom = uniform(self.zoom_range[0], self.zoom_range[1])
        #     transformations = {
        #         'zx': zoom,
        #         'zy': zoom,
        #         'shear': uniform(-self.shear_range, self.shear_range),
        #         'flip_horizontal': self.horizontal_flip and bool(randint(0,2))
        #     }
        #     transformed[i] = self.__img_gen.apply_transform(transformed[i], transformations)
        # return transformed * self.rescale
        # print(np.argmax(labels, axis=1))
        return transformed, labels

import pickle
def generate_batches(files, y, batch_size):
   counter = 0
   y_train = y.astype(np.float32)
   y_train = y_train.flatten()
   y_train = keras.utils.to_categorical(y_train, 3)
   while True:
     fname = files[counter]
     # print(fname)
     f = h5py.File(fname, 'r+')
     # print("Keys: %s" % f.keys())
     a_group_key = list(f.keys())[0]

     data_bundle = f[a_group_key][:]
     # print(data_bundle[0:2])
     # print(data_bundle[0,0,20,0:3])

     X_train = data_bundle.astype(np.float32)
     # print(X_train.shape)
     # print(y_train[0:20])
     for cbatch in range(0, X_train.shape[0], batch_size):
         yield (X_train[cbatch:(cbatch + batch_size),:,:], y_train[counter*1080+cbatch:(counter*1080+cbatch + batch_size),:])
     f.close()
     counter = (counter + 1) % len(files)
# training_data_1
# model = Sequential()
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



batch_size = 100
epochs = 2
from sklearn.utils import class_weight
from sklearn.metrics import balanced_accuracy_score



# the i now indicate the left out fold index. for instance, when i == 0, fold 0 is left out and folds 1 & 2 are in use
# y_train_0, y_train_1, y_train_2 = np.split(labels - 1, 3)

# print(y_train_0[0:20])

# y_train_splits = [y_train_0, y_train_1, y_train_2]
# data = data/255.  #maybe will decrease memory consumption idk
# X_train_0, X_train_1, X_train_2 = np.split(data, 3)
# X_train_splits = [X_train_0, X_train_1, X_train_2]

# y = [np.hstack((y_train_1, y_train_2)), np.hstack((y_train_0, y_train_2)), np.hstack((y_train_0, y_train_1))]
# X = [np.vstack((X_train_1, X_train_2)), np.vstack((X_train_0, X_train_2)), np.vstack((X_train_0, X_train_1))]

############################ cv #############################################

cv_acc_score = [0, 0, 0]
for i in [0, 1, 2]:
    if i == 0:
        x_val = data[0:21600]
        x_train = data[21600:64800]
        y_val = labels[0:21600] - 1
        y_train = labels[21600:64800] - 1
    elif i == 1:
        x_val = data[21600:43200]
        x_train = np.vstack((data[0:21600], data[43200:64800]))
        y_val = labels[21600:43200] - 1
        y_train = np.hstack((labels[0:21600], labels[43200:64800])) - 1
    else:
        x_val = data[43200:64800]
        x_train = data[0:43200]
        y_val = labels[43200:64800] - 1
        y_train = labels[0:43200] - 1


    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    print("class weights:")
    print(class_weights)
    print(np.unique(y_train))

    # one hot encoding
    y_train = keras.utils.to_categorical(y_train, 3)
    y_val = keras.utils.to_categorical(y_val, 3)

    model = make_resnet()

    from keras.preprocessing.image import ImageDataGenerator
    # create a data generator
    datagen = ImageDataGenerator()

    # rain_it = datagen.flow_from_directory('train/', class_mode='binary', batch_size=batch_size)
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    my_gen = CustomImagesGenerator(
        x_train, y_train,
        batch_size=batch_size
    )

    es = EarlyStopping(monitor=f1_m, mode='max', patience=50)

    # model.fit_generator(my_gen,
    #                class_weight={0:class_weights[0], 1:class_weights[1], 2:class_weights[2]},
    #                epochs = epochs,
    #                steps_per_epoch= np.ceil(x_train.shape[0]/batch_size),
    #                validation_data=(x_val, y_val),
    #                verbose=1)
    model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, verbose = 1, validation_data=(x_val, y_val), shuffle=True
              , class_weight = {0:class_weights[0], 1:class_weights[1], 2:class_weights[2]})

    # do cross validation on the left out
    y_cv_val_pred = np.argmax(model.predict(x_val), axis=1)
    cv_acc_score[i] = balanced_accuracy_score(np.argmax(y_val, axis=1), y_cv_val_pred)
    print("the validation balanced accuracy on train split " + str(i) + "is:")
    print(cv_acc_score[i])

    plot_confusion_matrix(np.argmax(y_val, axis=1), y_cv_val_pred,
                          normalize=True,
                          title="val_" + str(i),
                          cmap=plt.cm.Blues)
    plt.show()

    plot_confusion_matrix(np.argmax(y_train, axis=1), np.argmax(model.predict(x_train),axis=1),
                          normalize=True,
                          title="train_" + str(i),
                          cmap=plt.cm.Blues)
    plt.show()

print(cv_acc_score)
#
# del data
############################ cv #############################################

############################ train on all #####################################
train_files = []
for i in range(60):
    train_files.append("./train/" + "training_data_" + str(i) + ".h5")

model = make_resnet()
# gen = generate_batches(files=train_files, y = labels-1, batch_size=batch_size)
class_weights_entire_train = class_weight.compute_class_weight('balanced', np.unique(labels-1), labels-1)
# history = model.fit_generator(gen, steps_per_epoch=64800/batch_size, nb_epoch=epochs, verbose=1, class_weight=class_weights)
# history = model.fit_generator(gen, steps_per_epoch=10, nb_epoch=epochs, verbose=1, class_weight=class_weights)
model.fit(data, keras.utils.to_categorical(labels-1), batch_size=batch_size, epochs=epochs, verbose=1,
          shuffle=True
          , class_weight={0: class_weights_entire_train[0], 1: class_weights_entire_train[1], 2: class_weights_entire_train[2]})

# np.savetxt('pre.csv', X2, delimiter = ',')

############################ train on all #####################################



###############evaluation######################
test_data = np.zeros((43200, 24, 160, 3))
test_data = make_training_data_for_subject(data_input=test_data, subject=[0,1],training_or_test='testing')
print("check test data shape")
print(test_data.shape)
# print()
sample = pd.read_csv("sample.csv", delimiter=',')
prediction_test = model.predict(test_data)
prediction_test = np.argmax(prediction_test, axis=1)
print(prediction_test[0:10])
print(prediction_test.shape)

sample['y'] = prediction_test + 1
print("after adding 1 back")
print(sample['y'][0:10])
sample.to_csv("y_full_train.csv", index = False)
