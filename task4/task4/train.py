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

data = np.zeros((64800,24,160,3))

data = make_training_data_for_subject(data_input=data, subject=[0,1,2],training_or_test='training')

# print("check some entries")
# print(data[0,0,0,0:])


# starting to build their setup
def make_CNN():
    model = Sequential()

    model.add( MaxPooling2D(pool_size=(3, 2), strides=(3,2), padding='valid', data_format=None, input_shape= (24, 160, 3)) )

    model.add(Conv2D(filters=5, kernel_size=(3,3), strides=(1, 1), padding='valid', data_format=None,
                     dilation_rate=(1, 1), activation='relu', use_bias=True,
                     kernel_initializer='he_uniform', bias_initializer='zeros',
                     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                     kernel_constraint=None, bias_constraint=None,
                     )
             )

    model.add( MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='valid', data_format=None) )

    model.add(Dense(1000,
                    activation='relu',
                    kernel_initializer='he_uniform' ))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1000,
                    activation='relu',
                    kernel_initializer='he_uniform' ))
    model.add(BatchNormalization(axis=1))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # TODO: The weight decay rates of first and second moment were set to 0.9 and 0.99
    # TODO: cost-sensitve loss as in paper
    return model

# build a simple CNN, wrong metric used at the moment
def make_CNN_small():
    model = Sequential()

    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (24,160,3)))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.adam(learning_rate= 0.00005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

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

        print("start = {}".format(start))
        print("stop = {}".format(stop))
        print("self.__batch_index = {}".format(self.__batch_index))

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
        return transformed, labels


batch_size = 100
epochs = 1
from sklearn.utils import class_weight

# the i now indicate the left out fold index. for instance, when i == 0, fold 0 is left out and folds 1 & 2 are in use
y_train_0, y_train_1, y_train_2 = np.split(labels - 1, 3)

# data = data/255.  #maybe will decrease memory consumption idk
X_train_0, X_train_1, X_train_2 = np.split(data, 3)

y = [np.hstack((y_train_1, y_train_2)), np.hstack((y_train_0, y_train_2)), np.hstack((y_train_0, y_train_1))]
X = [np.vstack((X_train_1, X_train_2)), np.vstack((X_train_0, X_train_2)), np.vstack((X_train_0, X_train_1))]
for i in [0, 1, 2]:

    # y_train_0 = labels[0:43200] - 1

    print(y_train_0.shape)
    #X = np.vstack((training_data_0,training_data_1,training_data_2))
    # print(y_train_0[0:20])
    print(y[i].shape)
    print(X[i].shape)


    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(y[i]),
                                                      y[i])
    print(class_weights)
    # one hot encoding
    y_train = keras.utils.to_categorical(y[i], 3)
    X_train = X[i]
    # print(X_train_0.shape)
    # print(y_train_0.shape)


    model = make_CNN_small()

    from keras.preprocessing.image import ImageDataGenerator
    # create a data generator
    datagen = ImageDataGenerator()

    # rain_it = datagen.flow_from_directory('train/', class_mode='binary', batch_size=batch_size)
    from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
    my_gen = CustomImagesGenerator(
        X_train, y_train,
        batch_size=batch_size
    )

    model.fit_generator(my_gen,
                   class_weight=class_weights,
                   epochs = epochs,
                   steps_per_epoch= np.ceil(X[i].shape[0]/batch_size),
                   verbose=1)
