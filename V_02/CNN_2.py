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

# Window Cleanup
cv2.destroyAllWindows()
plt.close('all')
#%%

# from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D


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

# 5x5 kernels, stride 3:
# 128 / 3 = ca 43 -> 43x43 img
# 5*5 * 43*43 = 46225 multiplications per img
# subpooling 3x3:
# 43 / 3 = ca 15 -> 15x15 img
# 15*15=225 flattened parameters


# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.5))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'


# setup model
print("setup model")
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3,3), strides=(1,1),
                        input_shape=temp_img.shape, activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2,2))) # 64*64
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2,2))) # 32*32
model.add(keras.layers.Dropout(0.4))

# model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 16*16
# model.add(tf.keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())
units = 512 # num of hidden layers

model.add(keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 1
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 2
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(units=2, activation='sigmoid'))   # output layer

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

# start training
print("start training")
model.fit(dgen_train, 
        epochs = 64, 
        validation_data = dgen_val,
        verbose=1, 
        callbacks=early_stop )

plt.close("all")
metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()
plt.hlines(1,0,model.history.epoch[-1],linestyles="dashed")
plt.title("Loss")
metrics[['accuracy','val_accuracy']].plot()
plt.hlines(1,0,model.history.epoch[-1],linestyles="dashed")
plt.title("Accuracy")

print(model.summary())
# model.evaluate(x_test,y_test,verbose=1)

# predictions = model.predict_classes(x_test)



    
    
    
    
    
    
    
    
    
    
    
    