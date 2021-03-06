# -*- coding: utf-8 -*-

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
from tensorflow.keras.optimizers import Adam


#%%

print("\n## Calling main function.\n")

print("cv2.version = {}".format(cv2.__version__))
print("numpy.version = {}".format(np.__version__))
print("matplotlib.version = {}".format(mpl.__version__))
print("pandas.version = {}".format(pd.__version__))
print("tensorflow.version = {}".format(tf.__version__))
print()


# Window Cleanup
cv2.destroyAllWindows()
plt.close('all')

# mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#%%
# fancy plotting function

def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                  color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                  color=colors[n], label='Val ' + label,
                  linestyle="--",linewidth=1)
    plt.xlabel('Epoch')
    plt.title('Loss')

def plot_acc(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.plot(history.epoch, history.history['accuracy'],
                  color=colors[n], label='Train ' + label)
    plt.plot(history.epoch, history.history['val_accuracy'],
                  color=colors[n], label='Val ' + label,
                  linestyle="--",linewidth=1)
    plt.xlabel('Epoch')
    plt.title('Accuracy')

    
# %%

print("detect mite standalone")

# setup datagenerator
print("setup datagenerator")
image_generator = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=.1,
                horizontal_flip=True,
                vertical_flip=True,
                rescale=1./255,
                validation_split=0.2
                )

# get csv as dataframe
df = pd.read_csv("D:\\ECM_PROJECT\\pythoncode\\V_02/extracted/LearningImages_csv.csv", index_col=0)
# shuffle the DataFrame rows
df = df.sample(frac = 1)

# create iterators
dgen_train = image_generator.flow_from_dataframe(
    df,
    x_col='fpath',
    y_col=["has_mite"],
    # y_col=["has_bee","has_mite"],
    weight_col="weight",
    color_mode="grayscale",
    target_size = (64,64),
    class_mode="raw",
    # class_mode='categorical',
    subset="training" )

dgen_val = image_generator.flow_from_dataframe(
    df,
    batch_size=64,
    x_col='fpath',
    y_col=["has_mite"],
    # y_col=["has_bee","has_mite"],
    weight_col="weight",
    color_mode="grayscale",
    target_size = (64,64),
    class_mode="raw",
    # class_mode='categorical',
    subset="validation" )

# fetch one image to get its shape for the network input shape
temp_batch = dgen_val.next()
temp_img = temp_batch[0][0]

#%%
# Calculate initial output bias initializer
col_slice = df["has_mite"]
pos = len( df[df["has_mite"] > 0] )     # number of positive labels
neg = len( df[df["has_mite"] <= 0] )    # number of negative labels

bias_0 = None
# bias_0 = np.log(pos/neg)

# print("pos/neg = {:.4f}".format(pos/neg))
# print("bias_0 = {:.4f}".format(bias_0))
#%%
# setup model
print("setup model")
model = keras.models.Sequential()

# input dimension: 64*64
model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3,3),
                        input_shape=temp_img.shape, activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2,2))) # 32*32
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(2,2))) # 16*16
model.add(keras.layers.Dropout(0.3))

# model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 16*16
# model.add(tf.keras.layers.Dropout(0.5))

model.add(keras.layers.Flatten())
units = 256 # num of hidden layers

model.add(keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 1
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 2
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))

# setup output bias initializer
if bias_0 is not None:
    bias_0 = keras.initializers.Constant(bias_0)

# Output layer
model.add(keras.layers.Dense(
    units=1, 
    # units=1, 
    # activation=None, 
    activation="sigmoid",
    bias_initializer=bias_0
    ))   # output layer


# Compiling the Model
model.compile(
    optimizer = 'adam', 
    # optimizer = 'rmsprop', 
    # loss = 'mse', 
    # loss = tf.nn.sigmoid_cross_entropy_with_logits, 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy'])

# definition of early stop condition
early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5)

print(model.summary())
#%%
# start training
print("start training")
model.fit(dgen_train, 
        epochs = 101, 
        validation_data = dgen_val,
        verbose=1, 
        # callbacks=early_stop 
        )

#%%
plt.close("all")

history = model.history

plt.figure(1)
plot_loss( history, "", 0)
# plt.ylim(top=0.6)
plt.legend(loc="upper right")
plt.grid(True,"both")

plt.figure(2)
plot_acc(history, "", 0)
# plt.ylim(bottom=0.7,top=0.86)
plt.legend(loc="lower center")
plt.grid(True,"both")



pass