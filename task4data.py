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
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPooling2D,Conv2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
# for CNNs
# from keras.layers import Conv2D, MaxPooling2D
 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


batch_size = 100

import h5py

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


#test_data_1 = np.zeros((21600,24,160,3)) 
#test_data_1 = make_training_data_for_subject(data_input = test_data_1,subject=[1],training_or_test='test')



# build a simple CNN, wrong metric used at the moment
def make_CNN():
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
def make_CNN2():
    model = Sequential()

    model.add( MaxPooling2D(pool_size=(3, 2), strides=(3,2), padding='valid', data_format=None) )

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

from sklearn.utils import class_weight
batch_size = 100
epochs=2

label_path = "train_labels.csv"
df_labels = pd.read_csv(label_path)

labels = df_labels.values
labels = labels[:,1]

import h5py
hfr = h5py.File('training_data_2.h5', 'r')
hfr.keys()
n1 = hfr.get('data')
data = np.array(n1)

y = np.split(labels,3)
X_train_0 = data
y_train_0 = labels - 1
#X = np.vstack((training_data_0,training_data_1,training_data_2))
print(y_train_0)

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train_0),
                                                y_train_0)
print(class_weights)
# one hot encoding


print(X_train_0.shape)
print(y_train_0.shape)
model = make_CNN2()



from sklearn.metrics import balanced_accuracy_score


def leave_one_out(data, labels, clf, class_weights,epochs):
    x = np.split(data,3)
    y = np.split(labels,3)
    vals = []
    for i in range(3):
        
        x_val = x[i]
        y_val = labels[i*21600:(i+1)*21600]
        
    
        x_train = np.zeros((x_val.shape[0]*2,x_val.shape[1],x_val.shape[2],x_val.shape[2]))
        y_train = np.zeros((x_val.shape[0]*2,1))
        x_train[:21600,:] = x[(i+1)%3]
        x_train[21600:,:] = x[(i+2)%3]
        y_train[:21600] = y[(i+1)%3].reshape(x_val.shape[0],1)
        y_train[21600:] = y[(i+2)%3].reshape(x_val.shape[0],1)
        y_train = keras.utils.to_categorical(y_train,3)
        y_val = keras.utils.to_categorical(y_val,3)

        print("FITTING")
        clf.fit(x_train,y_train, class_weight=class_weights, epochs=epochs,verbose=3)
        y_pred = clf.predict(x_val)
        y_pred = np.argmax(y_pred, axis=1)
        print(y_pred.shape)
        print(y_val.shape)
        val = balanced_accuracy_score(y_val,y_pred)
        print("UUUUU {}".fotmat(val))
        vals.append(val)
    return vals


#vals = leave_one_out(X_train_0, y_train_0, model, class_weights, epochs)

y_train_0 = keras.utils.to_categorical(y_train_0,3)

model.fit(X_train_0, y_train_0,
           class_weight=class_weights,
           epochs = epochs,
           verbose=1 
         )


test_data_0 = np.zeros((21600*2,24,160,3)) 
test_data_0 = make_training_data_for_subject(data_input = test_data_0,subject=[0,1],training_or_test='test')
y_predict_0 = model.predict(test_data_0)
y_final = np.argmax(y_predict_0,axis=1) + 1

sample = pd.read_csv("sample.csv", delimiter=',')
sample['y'] = y_final
sample.to_csv("predictions_4_3.csv", index = False)