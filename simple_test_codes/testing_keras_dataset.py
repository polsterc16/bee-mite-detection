# -*- coding: utf-8 -*-
"""
Created on Tue May 18 09:39:07 2021

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 17 19:55:52 2021

@author: Admin

https://medium.com/analytics-vidhya/image-classification-with-tf-keras-introductory-tutorial-7e0ebb73d044
https://blog.eduonix.com/artificial-intelligence/convolutional-neural-networks-keras/
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
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

#%%


#%%
    
# %% 

class DataSetClass:
    def __init__(self):
        
        pass
    
    def parse_png(self, filename, label):
        # https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72
        
        IMG_SIZE = 128
        
        image_string = tf.io.read_file(filename)
        
        image_decoded = tf.image.decode_png(image_string)
        image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
        image_normalized  = image_resized / 255.0
        
        return image_normalized, label
    
    
    def create_dataset(self, filenames, labels, is_training=True):
        BATCH_SIZE = 256 # Big enough to measure an F1-score
        AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time
        SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations
        
        # Create a first dataset of file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # Parse and preprocess observations in parallel
        dataset = dataset.map(self.parse_png, num_parallel_calls=AUTOTUNE)
        
        if is_training == True:
            # This is a small dataset, only load it once, and keep it in memory.
            dataset = dataset.cache()
            # Shuffle the data each buffer size
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
            pass
        
        # Batch the data for multiple steps
        dataset = dataset.batch(BATCH_SIZE)
        # Fetch batches in the background while the model is training.
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        
        return dataset





#%%

if __name__== "__main__":
    print("\n## Calling main function.\n")
    
    print("cv2.version = {}".format(cv2.__version__))
    print("numpy.version = {}".format(np.__version__))
    print("matplotlib.version = {}".format(mpl.__version__))
    print("pandas.version = {}".format(pd.__version__))
    print()
    
    
    # Window Cleanup
    cv2.destroyAllWindows()
    plt.close('all')
    
    TEST = 3
    
    # %%
    if TEST == 1:
        X_data = ["D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00000_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00001_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00002_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00003_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00004_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00005_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00006_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00007_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00008_10.png",
                  "D:\ECM_PROJECT\pythoncode\V_02\learning\imgs\00009_10.png"]
        y_data = [ [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0], [1,0] ]
        
        myDSC = DataSetClass()
        ds = myDSC.create_dataset(X_data, y_data)
        
    
    # %%
    if TEST == 2:
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        pass
    
    
    
    # %%
    # https://studymachinelearning.com/keras-imagedatagenerator/
    # https://rubikscode.net/2019/12/09/creating-custom-tensorflow-dataset/
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    # https://godatadriven.com/blog/keras-multi-label-classification-with-imagedatagenerator/
    
    if TEST == 3:
        from keras.preprocessing.image import ImageDataGenerator
        from sklearn.model_selection import train_test_split
        
        path_imgs = "D:\\ECM_PROJECT\\pythoncode\\V_02\\learning\\imgs"
        path_pickle = "D:\\ECM_PROJECT\\pythoncode\\V_02\\learning\\imgs.p"
        
        image_generator = ImageDataGenerator(
                        rotation_range=90,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True,
                        vertical_flip=True,
                        rescale=1./255)
        #%%
        
        
        df = pd.read_csv("D:\\ECM_PROJECT\\pythoncode\\V_02/learning/data__csv.csv", index_col=0)
        # df["labels"]=""
        # for i in tqdm( range(len(df)), desc="going through df"):
        #     row = df.iloc[i]
        #     bee = row["has_bee"]
        #     mite = row["has_mite"]
        #     if bee>0 and mite>0:
        #         label=["bee","mite"]
        #     elif bee>0:
        #         label=["bee"]
        #     else:
        #         label=[]
        #     df.at[i,"labels"]=label
            
        img_iter = image_generator.flow_from_dataframe(
            df,
            x_col='fpath',
            y_col='labels',
            weight_col="weight",
            color_mode="grayscale",
            target_size = (128,128),
            class_mode='categorical',
            save_to_dir="D:\\ECM_PROJECT\\pythoncode\\V_02/learning/output", 
            save_prefix="img_gen_"
        )
        
        temp=img_iter.next()
        imgs=temp[0]
        img=imgs[0]
        
        #%%
        
        x_d, y_d, w_d = pickle.load( open( path_pickle, "rb" ) )
        
        my_flow = image_generator.flow(x_d, y_d, sample_weight=w_d, seed=42, 
                                        save_to_dir="D:\\ECM_PROJECT\\pythoncode\\V_02/learning/output", 
                                        save_prefix="img_gen_")
        temp = my_flow.next()
        
        # dataset = image_generator.flow_from_directory(directory=str(path_imgs),
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    