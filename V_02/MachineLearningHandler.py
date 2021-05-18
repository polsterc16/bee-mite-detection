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

class conv2d_class:
    def __init__(self, dir_learning):
        
        
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
    
    pass


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
    
    TEST = 4
    
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
        
        # 128x128
        # 5x5 kernels, stride 3:
        # 128 / 3 = ca 43 -> 43x43 img
        # 5*5 * 43*43 = 46225 multiplications per img
        # subpooling 3x3:
        # 43 / 3 = ca 15 -> 15x15 img
        # 15*15=225 flattened parameters
        
        # setup model
        print("setup model")
        model = tf.keras.models.Sequential()
        model.add(layers.Conv2D(filters = 32, kernel_size = (5,5), strides=(3,3),
                                input_shape=(128,128,1), activation = 'relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))
        model.add(tf.keras.layers.Flatten())
        dropout = 0.5
        units = 512 # num of hidden layers
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 1
        model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 2
        model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))   # output layer
        
        model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
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
                        validation_split=0.2)
        
        
        
        path_pickle = "D:\\ECM_PROJECT\\pythoncode\\V_02\\learning\\imgs.p"
        x_d, y_d, w_d = pickle.load( open( path_pickle, "rb" ) )
        
        # X_train, X_test, y_train, y_test
        x_train,x_test,y_train,y_test = train_test_split(x_d, y_d, test_size=0.20, random_state=33)
        w_train,w_test = train_test_split(w_d, test_size=0.20, random_state=33)
        
        # start training
        print("start training")
        model.fit(x=x_train, y=y_train, sample_weight=w_train, epochs = 10, verbose=1, 
                  validation_split=0.2)
        
        
        metrics = pd.DataFrame(model.history.history)
        metrics[['loss','val_loss']].plot()
        metrics[['accuracy','val_accuracy']].plot()
        
        
        model.evaluate(x_test,y_test,verbose=1)
        
        predictions = model.predict_classes(x_test)
        
        
        pass
    
    # %%
    if TEST == 3:
        from tensorflow.keras.optimizers import Adam
        # from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
        
        
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
                        validation_split=0.2)
        
        df = pd.read_csv("D:\\ECM_PROJECT\\pythoncode\\V_02/learning/data__csv.csv", index_col=0)
        # len_df = (len(df)//32)*32
        # df = df.iloc[:len_df]
        
        dgen = image_generator.flow_from_dataframe(
            df,
            x_col='fpath',
            y_col='labels',
            # weight_col="weight",
            color_mode="grayscale",
            target_size = (128,128),
            # class_mode="raw",
            class_mode='categorical',
            )
        val_data = dgen.next()
        
        
        # 128x128
        # 5x5 kernels, stride 3:
        # 128 / 3 = ca 43 -> 43x43 img
        # 5*5 * 43*43 = 46225 multiplications per img
        # subpooling 3x3:
        # 43 / 3 = ca 15 -> 15x15 img
        # 15*15=225 flattened parameters
        
        # setup model
        print("setup model")
        model = tf.keras.models.Sequential()
        
        model.add(layers.Conv2D(filters = 16, kernel_size = (3,3), strides=(1,1),
                                input_shape=val_data[0][0].shape, activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 64*64
        model.add(tf.keras.layers.Dropout(0.3))
        
        model.add(layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 32*32
        model.add(tf.keras.layers.Dropout(0.4))
        
        model.add(layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) # 16*16
        model.add(tf.keras.layers.Dropout(0.5))
        
        model.add(tf.keras.layers.Flatten())
        units = 128 # num of hidden layers
        
        model.add(tf.keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 1
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        
        model.add(tf.keras.layers.Dense(units = units, activation = 'relu')) # hidden layer 2
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.5))
        
        model.add(tf.keras.layers.Dense(units=3, activation='sigmoid'))   # output layer
        
        
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        
        # start training
        print("start training")
        model.fit(dgen, epochs = 10, verbose=1, 
                  validation_data=val_data )
        # model.fit(dgen_train, epochs = 10, verbose=1, 
        #           validation_data=dgen_test)
        
        
        metrics = pd.DataFrame(model.history.history)
        metrics[['loss','val_loss']].plot()
        metrics[['accuracy','val_accuracy']].plot()
        
        
        # model.evaluate(x_test,y_test,verbose=1)
        
        # predictions = model.predict_classes(x_test)
        
        
        pass
    
    # %%
    if TEST == 4:
        from tensorflow.keras.optimizers import Adam
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
        # len_df = (len(df)//32)*32
        # df = df.iloc[:len_df]
        # shuffle the DataFrame rows
        df = df.sample(frac = 1)
        # split_ratio = 0.2
        # split_i = int( len(df)*split_ratio )
        # df_train = df.iloc[split_i:]
        # df_val = df.iloc[:split_i]
        
        dgen_train = image_generator.flow_from_dataframe(
            df,
            # batch_size=64,
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
            # batch_size=64,
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
        
        # v_x,v_y,v_w = dgen_val.next()
        # for i in tqdm(range( int(dgen_val.samples/1.5)//dgen_val.batch_size) ):
        #     v_x2,v_y2,v_w2 = dgen_val.next()
        #     v_x = np.concatenate((v_x,v_x2))
        #     v_y = np.concatenate((v_y,v_y2))
        #     v_w = np.concatenate((v_w,v_w2))
        
        
        # 128x128
        # 5x5 kernels, stride 3:
        # 128 / 3 = ca 43 -> 43x43 img
        # 5*5 * 43*43 = 46225 multiplications per img
        # subpooling 3x3:
        # 43 / 3 = ca 15 -> 15x15 img
        # 15*15=225 flattened parameters
        
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
                # steps_per_epoch=int(4000/dgen_train.batch_size),
                epochs = 64, 
                validation_data = dgen_val,
                # validation_steps=int(1000/dgen_val.batch_size),
                verbose=1, 
                callbacks=early_stop )
        # model.fit(dgen_train, epochs = 10, verbose=1, 
        #           validation_data=dgen_test)
        
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
        
        
        pass
    
    
    
    
    
    
    
    
    
    
    
    