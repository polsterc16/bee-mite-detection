# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:18:49 2021

@author: Admin
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

#%% Setup

f_dir = "imgdatagenerator_data"
f_names = ["0.png", "1.png", "2.png", "3.png"]
save_dir = "imgdatagenerator_data/output"

data={"fnames":["0.png", "1.png", "2.png", "3.png"],
      "categ":["a","b","c","a"]}
df=pd.DataFrame.from_dict(data)
#%% Initiate ImageDataGenerator settings
image_generator = ImageDataGenerator(
                    rotation_range=45,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=.1 )
#%% Create object of this ImageDataGenerator for img files
gen = image_generator.flow_from_dataframe(df, 
                                          directory="input",
                                          x_col="fnames",
                                          y_col="categ",
                                          target_size=(64,64),
                                          color_mode="grayscale",
                                          save_to_dir="output",
                                          save_prefix="test_" )
#%% create output images 4 times for comparison
for i in range(4):
    ret=gen.next()
    pass


import pandas as pd

data={"fnames":["0.png", "1.png", "2.png", "3.png"],
      "categ":["a","b","c","a"]}
df=pd.DataFrame.from_dict(data)

gen = image_generator.flow_from_dataframe(
    df, 
    directory="input",
    x_col="fnames",
    y_col="categ",
    target_size=(64,64),
    color_mode="grayscale",
    save_to_dir="output",
    save_prefix="test_" )



