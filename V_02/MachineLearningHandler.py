# -*- coding: utf-8 -*-
"""
Created on Mon May 17 19:55:52 2021

@author: Admin

https://medium.com/analytics-vidhya/image-classification-with-tf-keras-introductory-tutorial-7e0ebb73d044
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
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

#%%


#%%

class ImportFromExtraction:
    def __init__(self, dir_learning, path_source_csv):
        self.set_dir_learning(dir_learning)
        self.set_path_source(path_source_csv)
        
        self.df_startup()
        pass
    # -------------------------------------------------------------------------
    def check_isdir(self,path):
        """Stop object creation, if no valid directory path is given. Returns the absolute path."""
        if (os.path.isdir(path) == False):
            raise Exception("Requires a legal directory path!")
        return os.path.abspath(path)
    def check_isfile(self,path):
        """Stop object creation, if no valid file path is given. Returns the absolute path."""
        if (os.path.isfile(path) == False):
            raise Exception("Requires a legal file path!")
        return os.path.abspath(path)
    
    def set_dir_learning(self,path):
        img_dir_name = "imgs"
        self._dir_learning = self.check_isdir(path)
        self._dir_learning_imgs = self.check_isdir( os.path.join(self._dir_learning, img_dir_name) )
        pass
    
    def set_path_source(self,path):
        abs_path = self.check_isfile(path)
        
        path_sep = abs_path.split(".")
        if path_sep[-1] != "csv": raise Exception("Must be a valid csv file!")
        
        self._path_source_csv = abs_path
        pass
    
    def df_startup(self):
        min_length_csv = 10 # the minimum length the labeled df must have for us to accept it
        
        # read the source csv
        self._df_source = pd.read_csv(self._path_source_csv, index_col=0)
        
        # check if there are at least SOME labeled images in the df
        if len(self._df_source) < min_length_csv:
            raise Exception("Source csv-file is too short!" ); pass
        
        columns = self._df_source.columns
        fname_data = "data"
        self._df_fname_data_csv =  os.path.join(self._dir_learning, "{}__csv.csv".format(fname_data))
        self._df_fname_data_scsv = os.path.join(self._dir_learning, "{}__scsv.csv".format(fname_data))
        
        self._df_data = pd.DataFrame(columns=columns)
        pass
    
    def df_store(self):
        """Saves the dataframe"""
        self._df_data.to_csv(self._df_fname_data_csv,  sep=",")
        self._df_data.to_csv(self._df_fname_data_scsv, sep=";")
        print("Stored {}".format(self._df_fname_data_csv))
        pass
    
    def process(self):
        df = self._df_source
        # got thorugh all elements of source csv, load the images to grayscale and store them in the target img location
        i = 0
        for idx, row in tqdm( df.iterrows(), desc="Processing Images", total=len(df)):
            fname = "{:05}_{}{}.png".format(i, int(row["has_bee"]), int(row["has_mite"]) )
            fpath = os.path.join(self._dir_learning_imgs, fname)
            
            img = cv2.imread(row["fpath"], cv2.IMREAD_GRAYSCALE)    # read img as grayscale
            cv2.imwrite(fpath, img)     # store img to target destination
            
            data_slice = {"fname": fname,
                          "fpath": fpath,
                          "src_fname":  row["src_fname"],
                          "src_fpath":  row["src_fpath"],
                          "roi_fpath":  row["roi_fpath"],
                          "pos_center": row["pos_center"],
                          "pos_anchor": row["pos_anchor"],
                          "has_bee":    row["has_bee"],
                          "has_mite":   row["has_mite"]}
            
            self._df_data.at[i] = pd.Series(data_slice)
            
            i += 1
            pass
        
        
        self.df_store()
        pass
    
    
# %% 

class DataSetClass:
    def __init__(self):
        
        pass
    
    def parse_img(self, filename, label):
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
        dataset = dataset.map(self.parse_img, num_parallel_calls=AUTOTUNE)
        
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
    
    TEST = 2
    
    # %%
    if TEST == 1:
        print("Prepares (grayscale) and copies the images over to the learning folder")
        dir_learn = "learning"
        
        dir_extr = "extracted"
        fname_learning = "LearningImages_csv.csv"
        fpath_learning = os.path.join(dir_extr, fname_learning)
        
        
        myIFE = ImportFromExtraction(dir_learn, fpath_learning)
        myIFE.process()
        
        pass
    
    # %%
    if TEST == 2:
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    