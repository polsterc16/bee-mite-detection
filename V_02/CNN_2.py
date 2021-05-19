# -*- coding: utf-8 -*-
"""
Created on Mon May 17 19:55:52 2021

@author: Admin

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import pickle
import seaborn as sns

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# from tensorflow.keras.models import Sequential

from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import Adam

#%%



print("cv2.version = {}".format(cv2.__version__))
print("numpy.version = {}".format(np.__version__))
print("matplotlib.version = {}".format(mpl.__version__))
print("pandas.version = {}".format(pd.__version__))
print("tensorflow.version = {}".format(tf.__version__))
print()




#%%


# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
HP_NUM_UNITS =  hp.HParam('num_units', hp.Discrete([128, 256, 512]))
HP_NUM_FILTERS =    hp.HParam('filters', hp.Discrete([8, 16, 32]))
# HP_KERNEL_SIZE =    hp.HParam('kernel_size', hp.Discrete([3, 5]))
# HP_STRIDES =    hp.HParam('strides', hp.Discrete([1,2,3]))
# HP_DROPOUT =    hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
# HP_DROPOUT =    hp.HParam('dropout', hp.Discrete( [0.1, 0.2, 0.3, 0.4, 0.5] ))
# HP_POOL_SIZE =  hp.HParam('pool_size', hp.Discrete([2, 4]))

# HP_OPTIMIZER =  hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'



def train_test_model(dg_train, dg_test, input_shape, output_len, hparams, epochs:int=1):
    
    model = keras.models.Sequential()
    
    
    model.add(keras.layers.Conv2D(
        filters =       hparams[HP_NUM_FILTERS],
        kernel_size =   (3,3),
        # strides =       (hparams[HP_STRIDES],hparams[HP_STRIDES]),
        input_shape = temp_img.shape, activation = 'relu' ))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D( pool_size = ( 2,2 ) ))
    model.add(keras.layers.Dropout( 0.3 ))
    
    
    model.add(keras.layers.Conv2D(
        filters =       2*hparams[HP_NUM_FILTERS],
        kernel_size =   (3,3),
        # strides =       (hparams[HP_STRIDES],hparams[HP_STRIDES]),
        activation = 'relu' ))
    
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D( pool_size = ( 2,2 ) ))
    model.add(keras.layers.Dropout( 0.4 ))
    
    # model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 16*16
    # model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(keras.layers.Flatten())
    
    model.add(keras.layers.Dense(units = hparams[HP_NUM_UNITS], activation = 'relu')) # hidden layer 1
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout( 0.5 ))
    
    model.add(keras.layers.Dense(units = hparams[HP_NUM_UNITS], activation = 'relu')) # hidden layer 2
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout( 0.5 ))
    
    model.add(keras.layers.Dense(units=2, activation='sigmoid'))   # output layer
    
    model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
    # early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

    
    model.fit(dg_train,
              # steps_per_epoch=64,
              epochs=epochs, verbose=1) 
    _, accuracy = model.evaluate(dg_test)
    return accuracy

def run(run_dir, dg_train, dg_test, input_shape, output_len, hparams, epochs:int=1):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        
        accuracy = train_test_model(dg_train, dg_test, input_shape, output_len, hparams, epochs)
        
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


#%%

# Window Cleanup
cv2.destroyAllWindows()
plt.close('all')
#%%


# setup datagenerator
print("setup datagenerator")
image_generator = ImageDataGenerator(
                rotation_range=45,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=.1,
                horizontal_flip=True,
                vertical_flip=True,
                rescale=1./255,
                validation_split=0.2
                )

df = pd.read_csv("D:\\ECM_PROJECT\\pythoncode\\V_02/learning/data__csv.csv", index_col=0)
df = df.sample(frac = 1)

dgen_train = image_generator.flow_from_dataframe(
    df,
    x_col='fpath',
    y_col=["has_bee","has_mite"],
    weight_col="weight",
    color_mode="grayscale",
    target_size = (64,64),
    class_mode="raw",
    # class_mode='categorical',
    subset="training"
    )

dgen_val = image_generator.flow_from_dataframe(
    df,
    x_col='fpath',
    y_col=["has_bee","has_mite"],
    weight_col="weight",
    color_mode="grayscale",
    target_size = (64,64),
    class_mode="raw",
    # class_mode='categorical',
    subset="validation"
    )
temp_batch = dgen_val.next()
temp_img = temp_batch[0][0]


#%%
session_num = 0
path_log = "D:\\ECM_PROJECT\\pythoncode\\V_02\\learning\\logs\\tuning"

for num_units in HP_NUM_UNITS.domain.values:
    for filters in HP_NUM_FILTERS.domain.values:
        # for kernel_size in HP_KERNEL_SIZE.domain.values:
            # for strides in HP_STRIDES.domain.values:
        # for dropout in HP_DROPOUT.domain.values:
        # for pool_size in HP_POOL_SIZE.domain.values:
            # for optimizer in HP_OPTIMIZER.domain.values:
        hparams = {
            HP_NUM_UNITS: num_units,
            HP_NUM_FILTERS: filters,
            # HP_KERNEL_SIZE: kernel_size,
            # HP_STRIDES: strides,
            # HP_DROPOUT: dropout,
            # HP_POOL_SIZE: pool_size,
            # HP_OPTIMIZER: optimizer,
        }
        run_name = "run-{:012}".format(session_num)
        print('\n--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        print()
        path_file = os.path.join(path_log, run_name)
        run(path_file, dgen_train,dgen_val,temp_img.shape,2,hparams,10)
        session_num += 1



    
    
    
    
    
    
    
    
    
    
    