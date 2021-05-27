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
        
        columns = self._df_source.columns.tolist()
        fname_data = "data"
        self._df_fname_data_csv =  os.path.join(self._dir_learning, "{}__csv.csv".format(fname_data))
        self._df_fname_data_scsv = os.path.join(self._dir_learning, "{}__scsv.csv".format(fname_data))
        
        columns.append("labels")
        self._df_data = pd.DataFrame(columns=columns)
        pass
    
    def df_store(self):
        """Saves the dataframe"""
        self._df_data.to_csv(self._df_fname_data_csv,  sep=",")
        self._df_data.to_csv(self._df_fname_data_scsv, sep=";")
        print("Stored {}".format(self._df_fname_data_csv))
        pass
    
    def process(self, make_pickle=False):
        if make_pickle:
            self._process_pickle()
        else:
            self._process()
        pass
    
    def _process(self):
        df = self._df_source
        # got thorugh all elements of source csv, load the images to grayscale and store them in the target img location
        i = 0
        for idx, row in tqdm( df.iterrows(), desc="Processing Images", total=len(df)):
            fname = "{:05}_{}{}.png".format(i, int(row["has_bee"]), int(row["has_mite"]) )
            fpath = os.path.join(self._dir_learning_imgs, fname)
            
            img = cv2.imread(row["fpath"], cv2.IMREAD_GRAYSCALE)    # read img as grayscale
            cv2.imwrite(fpath, img)     # store img to target destination
            
            bee = row["has_bee"]
            mite = row["has_mite"]
            label=[]
            if bee>0:
                label.append("bee")
                if mite>0:
                    label.append("mite")
            
            data_slice = {"fname": fname,
                          "fpath": fpath,
                          "src_fname":  row["src_fname"],
                          "src_fpath":  row["src_fpath"],
                          "roi_fpath":  row["roi_fpath"],
                          "pos_center": row["pos_center"],
                          "pos_anchor": row["pos_anchor"],
                          "has_bee":    row["has_bee"],
                          "has_mite":   row["has_mite"],
                          "weight":     row["weight"],
                          "labels":     label}
            
            self._df_data.at[i] = pd.Series(data_slice)
            
            i += 1
            pass
        
        
        self.df_store()
        pass
    
    def _process_pickle(self):
        df = self._df_source
        
        row = df.iloc[0]
        fpath = row["fpath"]
        img = cv2.imread(fpath)
        dim = img.shape[0:2]
        
        x_data = np.zeros( (len(df), dim[0],dim[1],1), dtype=np.uint8 )
        y_data = np.zeros( (len(df), 2), dtype=np.uint8 )
        w_data = np.zeros( len(df), dtype=np.float16 )
        
        # got thorugh all elements of source csv, load the images to grayscale and store them in the target img location
        for i in tqdm( range( len(df) ), desc="Processing Images"):
            row = df.iloc[i]
            fname = "{:05}_{}{}.png".format(i, int(row["has_bee"]), int(row["has_mite"]) )
            fpath = os.path.join(self._dir_learning_imgs, fname)
            
            img = cv2.imread(row["fpath"], cv2.IMREAD_GRAYSCALE)    # read img as grayscale
            cv2.imwrite(fpath, img)     # store img to target destination
            img_rank4 = np.reshape(img, (128,128,1) )
            x_data[i] = img_rank4
            y_data[i] = [int(row["has_bee"]), int(row["has_mite"])]
            w_data[i] = row["weight"]
            
            bee = row["has_bee"]
            mite = row["has_mite"]
            label=[]
            if bee>0:
                label.append("bee")
                if mite>0:
                    label.append("mite")
            
            data_slice = {"fname": fname,
                          "fpath": fpath,
                          "src_fname":  row["src_fname"],
                          "src_fpath":  row["src_fpath"],
                          "roi_fpath":  row["roi_fpath"],
                          "pos_center": row["pos_center"],
                          "pos_anchor": row["pos_anchor"],
                          "has_bee":    row["has_bee"],
                          "has_mite":   row["has_mite"],
                          "weight":     row["weight"],
                          "labels":     label}
            
            self._df_data.at[i] = pd.Series(data_slice)
            
            i += 1
            pass
        
        p_name = "imgs.p"
        path = os.path.join(self._dir_learning, p_name)
        pickle.dump( (x_data, y_data, w_data), open( path, "wb" ) )
        self.df_store()
        pass
    
    pass


#%%


#%%

if __name__== "__main__":
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
    
    TEST = 1
    
    # %%
    if TEST == 1:
        print("Prepares (grayscale) and copies the images over to the learning folder")
        dir_learn = "learning"
        
        dir_extr = "extracted"
        fname_source_csv = "LearningImages_csv.csv"
        fpath_source_csv = os.path.join(dir_extr, fname_source_csv)
        
        
        myIFE = ImportFromExtraction(dir_learn, fpath_source_csv)
        myIFE.process()
        
        pass
    
    
    
    
    
    
    
    
    
    
    
    