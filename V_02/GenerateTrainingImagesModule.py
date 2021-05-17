# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:22:43 2021

@author: Admin

Purpose: 
    1) Generate bee images WITH the mite image place at the badomen position
    2) Generate additional empty focus images

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import random

from tqdm import tqdm
from ast import literal_eval

import ImageHandlerModule as IHM

#%%

class GenerateLearningImagesClass:
    def __init__(self, dir_extraction, dir_mites, ratio_empty=1, fract_mite=0.5):
        """
        'ratio_empty' will set how many empty images shall be used in realtion to bee images (we can generate them if needed)
        
        'fract_mite' will set how many of the bee images shall have mites placed on them.
        """
        self.set_dir_extracted(dir_extraction)
        self.set_dir_mites(dir_mites)
        self.set_dataset_fractions(ratio_empty, fract_mite)
        
        self.df_startup()
        self.df_check_sizes()
        
        # self.generate_start()
        
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
    
    def set_dir_extracted(self,path):
        self._dir_extracted = self.check_isdir(path)
        self._dir_focus_empty = self.check_isdir( os.path.join(self._dir_extracted, "focus_empty") )
        self._dir_focus_mite = self.check_isdir( os.path.join(self._dir_extracted, "focus_mite") )
        
        pass
    
    def set_dir_mites(self,path):
        """Sets path of mite imgs and creates an IFC object to that folder."""
        self._dir_mites = self.check_isdir(path)
        self._mites_IFC = IHM.ImageFinderClass(self._dir_mites,("png",))
        
        pass
    
    def fetch_mite(self, idx:int):
        """returns items with circular indexing"""
        return self._mites_IFC.file_list[ idx % self._mites_IFC.size ]
    
    def set_dataset_fractions(self, ratio_empty, fract_mite):
        if (ratio_empty < 0):
            raise Exception("'ratio_empty' must be a number greater than 0! (default 1)")
        if not(0 <= fract_mite <= 1):
            raise Exception("'fract_mite' must be a number between 0 and 1! (default 0.5)")
        
        self._dataset_ratio_empty = ratio_empty
        self._dataset_fraction_mite = fract_mite
        pass
    
    def df_startup(self):
        min_length_labels = 10 # the minimum length the labeled df must have for us to accept it
        
        fname_labels = "Labels"
        fname_learning = "LearningImages"
        fname_FE = "Find_Empty"
        path_dir = self._dir_extracted
        
        self._df_fname_labels_csv =  os.path.join(path_dir, "{}_csv.csv".format(fname_labels))
        self._df_fname_FE_csv =  os.path.join(path_dir, "{}_csv.csv".format(fname_FE))
        
        self._df_fname_learning_csv =  os.path.join(path_dir, "{}_csv.csv".format(fname_learning))
        self._df_fname_learning_scsv = os.path.join(path_dir, "{}_scsv.csv".format(fname_learning))
        
        
        # check if the [comma] separated value file exists
        self.check_isfile(self._df_fname_labels_csv)    # labels csv file MUST exist!
        self.check_isfile(self._df_fname_FE_csv)        # Find Empy csv file MUST exist!
        
        
        # read the labels csv
        self._df_labels = pd.read_csv(self._df_fname_labels_csv, index_col=0)
        self._df_labels.dropna(inplace=True) # get rid of all NaN value Rows (only labeled rows remain)
        
        # check if there are at least SOME labeled images in the df
        if len(self._df_labels) < min_length_labels:
            raise Exception("Too few labeled focus images in '{}'".format(self._df_fname_labels_csv) ); pass
        
        self._df_labels = self._df_labels[["fname", "fpath", 
                          "parent_fname", "parent_fpath", "roi_fpath",
                          "pos_center", "pos_anchor",
                          "has_bee", "img_sharp", "rel_pos_abdomen"]]
        
        # read the Find EMpty csv
        self._df_FE = pd.read_csv(self._df_fname_FE_csv, index_col=0)
        self._df_FE = self._df_FE.loc[ self._df_FE["empty"] > 0 ]
         
        
        
        # make the learning df
        cols_learning =  ["fname", "fpath", 
                          "src_fname", "src_fpath", "roi_fpath",
                          "pos_center", "pos_anchor",
                          "has_bee", "has_mite", "weight"]
        self._df_learning = pd.DataFrame(columns=cols_learning)
         
         
        
        # sort df_labels by "has_bee" (just to get a good overview when looking at it directly)
        self._df_labels.sort_values(by=["has_bee","rel_pos_abdomen"], inplace=True)
        
        # make df of empty focus image
        self._df_isEmpty = self._df_labels.copy()
        self._df_isEmpty = self._df_isEmpty.loc[ self._df_isEmpty["has_bee"] <= 0 ]
        
        # make df of bee focus images (also one where we have the abdom pos)
        self._df_isBee = self._df_labels.copy()
        self._df_isBee = self._df_isBee.loc[ self._df_isBee["has_bee"] > 0 ]
        self._df_isBee_wA = self._df_isBee.loc[self._df_isBee['rel_pos_abdomen'].str.contains('\(') ]
        
        pass
    
    def df_check_sizes(self):
        self._len_labels =  len(self._df_labels)
        self._len_isEmpty = len(self._df_isEmpty)
        self._len_isBee =   len(self._df_isBee)
        self._len_isBee_wA = len(self._df_isBee_wA)
        
        # calculate how many images to be used / generated
        self._target_isEmpty = int(self._len_isBee * self._dataset_ratio_empty)
        self._target_hasMite = int(self._len_isBee * self._dataset_fraction_mite)
        
        
        # Tell us, what is possible
        print()
        print("The dataframe has {} labeled focus images.".format(self._len_labels) )
        print("We have {} 'bee focus' images.".format(self._len_isBee) )
        
        print("We want {} 'mite-bee' images. We have {} 'abdomen-bee' images.".format(self._target_hasMite, self._len_isBee_wA) )
        if (self._target_hasMite > self._len_isBee_wA):
            self._target_hasMite = self._len_isBee_wA
            print("\tWe can only generate {} 'mite-bee' images!".format(self._target_hasMite) )
            
        print("We want {} 'empty focus' images. We have {} 'empty focus' images.".format(self._target_isEmpty, self._len_isEmpty) )
        if (self._target_isEmpty > self._len_isEmpty):
            print("\tWe can generate the missing {} 'empty focus' images!".format(self._target_isEmpty - self._len_isEmpty) )
        
        print()
        pass
    
    def generate(self):
        # generate mite imgs at the specified fraction
        self._generate_mites()
        
        # write prepared bee df to output df 
        for i in tqdm( range( len(self._df_isBee_prepared) ),
                       desc="Write bee/mites to df learning" ):
            row = self._df_isBee_prepared.iloc[i]
            
            if row["img_sharp"] > 0:
                weight = 1.0
            else:
                weight = 0.5
            
            #["fname", "fpath", "src_fname", "src_fpath", "roi_fpath", 
            # "pos_center", "pos_anchor", "has_bee", "has_mite"]
            data_slice = {"fname": row["fname"],
                          "fpath": row["fpath"],
                          "src_fname":  row["parent_fname"],
                          "src_fpath":  row["parent_fpath"],
                          "roi_fpath":  row["roi_fpath"],
                          "pos_center": row["pos_center"],
                          "pos_anchor": row["pos_anchor"],
                          "has_bee":    row["has_bee"],
                          "has_mite":   row["has_mite"],
                          "weight":     weight}
            
            self._df_learning.at[row["fname"]] = pd.Series(data_slice)
            #----------------------
        
        
        #generate additional empty images, if necessary
        self._generate_empty(self._target_isEmpty - self._len_isEmpty)
        
        # write empty to output df (not mor then self._target_isEmpty, if we have more)
        for i in tqdm( range( min( [len(self._df_isEmpty),self._target_isEmpty] ) ),
                       desc="Write empty to df learning" ):
            row = self._df_isEmpty.iloc[i]
            
            if row["img_sharp"] > 0:
                weight = 1.0
            else:
                weight = 0.5
            
            #["fname", "fpath", "src_fname", "src_fpath", "roi_fpath", 
            # "pos_center", "pos_anchor", "has_bee", "has_mite"]
            data_slice = {"fname":      row["fname"],
                          "fpath":      row["fpath"],
                          "src_fname":  row["parent_fname"],
                          "src_fpath":  row["parent_fpath"],
                          "roi_fpath":  row["roi_fpath"],
                          "pos_center": row["pos_center"],
                          "pos_anchor": row["pos_anchor"],
                          "has_bee":    0,
                          "has_mite":   0,
                          "weight":     weight}
            
            self._df_learning.at[row["fname"]] = pd.Series(data_slice)
            pass #----------------------
        
        
        # store output df
        self._df_learning.to_csv(self._df_fname_learning_csv,  sep=",")
        self._df_learning.to_csv(self._df_fname_learning_scsv, sep=";")
        
        print("Stored {}".format(self._df_fname_learning_csv))
        pass
    
    def _generate_mites(self):
        mite_scale = 0.5 # by how much we ant to scale down the mite imgs (they are too big)
        # get local copy of df
        df_bees = self._df_isBee.copy()
        df_bees["has_mite"] = 0 #add "has_mite" col with default assignment
        # delete rows via < df.drop(index_label, inplace=true) >
        
        list_img_bees_wA = self._df_isBee_wA.index.values.tolist()
        
        for i in tqdm( range(self._target_hasMite), desc="Generate mite focus images"):
            pos = random.randrange(len(list_img_bees_wA))   # get random position in the list
            # pos = 645
            # i=758
            row_Idx = list_img_bees_wA[pos]                 # fetch item drom the list
            list_img_bees_wA.remove(row_Idx)                # remove the item from the list
            
            row = df_bees.loc[row_Idx]                      # fetch that row from the df
            df_bees.drop(row_Idx, inplace=True)             # remove that row from the df
            row_orig = self._df_labels.loc[row_Idx]         # we need the original row to access the mite pos
            pos_mite = literal_eval(row_orig["rel_pos_abdomen"])
            if type(pos_mite) not in [list,tuple]: raise Exception("Expected a tuple as 'rel_pos_abdomen'!")
            
            
            # new file name and path (for savign the img and writing to df)
            fname = "mite_{:06}.png".format(i)
            fpath = os.path.join(self._dir_focus_mite, fname)
            
            
            # print("DEBUG")
            # pos_mite = (0,127) # this is just for debugging!
            
            focus_path = row["fpath"]
            img_focus = cv2.imread(focus_path) #read that focus image
            img_rows,img_cols = img_focus.shape[0:2]
            
            # get the mite image (BGRA)
            mite = self.fetch_mite(i)                               # fetch mite at pos i
            mite_path = os.path.join(self._dir_mites,mite)
            img_mite = cv2.imread(mite_path, cv2.IMREAD_UNCHANGED ) #load mite img
            img_mite = cv2.resize(img_mite, (0,0), fx=mite_scale, fy=mite_scale)
            
            img_mask = img_mite[:,:,3]                              # fetch alpha channel as mask
            ret,img_mask = cv2.threshold(img_mask, 127, 255, cv2.THRESH_BINARY)
            img_mite = cv2.cvtColor(img_mite, cv2.COLOR_BGRA2BGR)   # cvt to bgr
            
            #get center offset for placing the mite img
            mite_rows,mite_cols = img_mask.shape[0:2]
            _, contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0];  M = cv2.moments(cnt)
            mite_cx = int(M['m10']/M['m00'])    # get center of mite img
            mite_cy = int(M['m01']/M['m00'])
            
            
            # get two anchor points of bounding rect of mite img (if they are outside the focus image, we have trouble)
            a1 = (pos_mite[0]-mite_cx, pos_mite[1]-mite_cy)
            a2 = (a1[0]+mite_cols, a1[1]+mite_rows)
            
            b1 = list(a1)
            b2 = list(a2)
            # cv2.imshow("before",img_mask)
            
            # check if a below [0,0]
            if a1[0] < 0:   # if a1 has negative x value
                b1[0] = 0
                img_mite = img_mite[:, -a1[0]:mite_cols, :].copy()
                img_mask = img_mask[:, -a1[0]:mite_cols].copy()
                
            if a1[1] < 0:   # if a1 has negative y value
                b1[1] = 0
                img_mite = img_mite[-a1[1]:mite_rows, :, :].copy()
                img_mask = img_mask[-a1[1]:mite_rows, :].copy()
                pass
            
            #update, if changed
            mite_rows,mite_cols = img_mask.shape[0:2]
            
            # check if a above [focus img shape]
            if a2[0] > img_cols:   # if a2 exceeds in x axis
                b2[0] = img_cols
                img_mite = img_mite[:, 0:mite_cols - (a2[0]-img_cols), :].copy()
                img_mask = img_mask[:, 0:mite_cols - (a2[0]-img_cols)].copy()
                
            if a2[1] > img_rows:   # if a1 has negative y value
                b2[1] = img_rows
                img_mite = img_mite[0:mite_rows - (a2[1]-img_rows), :, :].copy()
                img_mask = img_mask[0:mite_rows - (a2[1]-img_rows), :].copy()
                pass
            # cv2.imshow("after",img_mask)
            
            # Now we have reduced the mite imge, so that it does not clip - and the anchors for accessing the ROI as well (b1,b2)
            img_roi = img_focus[b1[1]:b2[1], b1[0]:b2[0], :]  # remember: np.arrays are accessed [rows,cols]!!!
            img_mask_inv = cv2.bitwise_not(img_mask)
            
            try:
                img1_bg = cv2.bitwise_and(img_roi, img_roi, mask = img_mask_inv ) # make bg only img
                img2_fg = cv2.bitwise_and(img_mite,img_mite,mask = img_mask)      # make fg only img
            except:
                print(i)
                print(pos)
                print(mite)
                print(a1,a2)
                print(b1,b2)
                print(img_mite.shape)
                print(img_mask.shape)
                raise Exception("DEBUG")
                
            img_dst = cv2.add(img1_bg,img2_fg)              # add fg to bg
            img_focus[b1[1]:b2[1], b1[0]:b2[0]] = img_dst   # write new img to where the roi comes from
            # cv2.imshow("mite places",img_focus)
            cv2.imwrite(fpath, img_focus)
            
            
            # prepare the seried to be added to the empty df
            data_slice = {"fname": fname, 
                          "fpath": fpath, 
                          "parent_fname": row["parent_fname"], 
                          "parent_fpath": row["parent_fpath"], 
                          "roi_fpath":  row["roi_fpath"],
                          "pos_center": row["pos_center"], 
                          "pos_anchor": row["pos_anchor"],
                          "has_bee":    1, 
                          "img_sharp":  row["img_sharp"], 
                          "has_mite":   1}
            # write data slice to df
            df_bees.at[fname] = pd.Series(data_slice)
            # print(pos_FE)
            
        self._df_isBee_prepared = df_bees
        
        pass
    
    def _generate_empty(self, number_added:int):
        if number_added <= 0: return
        
        # print("Will generate {} new empty images.".format(number_added))
        df_FE = self._df_FE
        
        # fetch the first focus image to get the dimensions
        row_0 = self._df_labels.iloc[0]
        path_img_0 = row_0["fpath"]
        img_0 = cv2.imread(path_img_0)
        self._focus_img_shape = img_0.shape
        
        # create empty images and add them to the df_empty
        for i in tqdm( range(number_added), desc="Generate empty focus images"):
            pos_FE = random.randrange(0, len(df_FE) ) # fetches a random index for the df_FE
            row_FE = df_FE.iloc[pos_FE]
            
            fname = "empty_{:06}.png".format(i)
            fpath = os.path.join(self._dir_focus_empty, fname)
            src_fname = row_FE["src_fname"]
            src_fpath = row_FE["src_fpath"]
            
            img = cv2.imread(src_fpath)
            shape_src = img.shape
            shape_focus = self._focus_img_shape
            
            # get the limits for focus img anchor position (from the anchor we need space for the img dims itself)
            range_row = shape_src[0] - shape_focus[0]
            range_col = shape_src[1] - shape_focus[1]
            
            # get random position for anchor of focus image
            c_row = random.randrange(0, range_row )
            c_col = random.randrange(0, range_col )
            pos_anchor = (c_col, c_row)
            
            # get our focus region & write to fpath
            img_focus = img[c_row:c_row+shape_focus[0], c_col:c_col+shape_focus[1]]
            cv2.imwrite(fpath, img_focus)
            
            # prepare the seried to be added to the empty df
            data_slice = {"fname": fname, 
                          "fpath": fpath, 
                          "parent_fname": src_fname, 
                          "parent_fpath": src_fpath, 
                          "roi_fpath":  str(None),
                          "pos_center": str(pos_anchor), 
                          "pos_anchor": str(pos_anchor),
                          "has_bee":    0, 
                          "img_sharp":  1, 
                          "rel_pos_abdomen": str(None)}
            # write data slice to df
            self._df_isEmpty.at[fname] = pd.Series(data_slice)
            # print(pos_FE)
            
        
        pass
    
    def pie(self):
        df = self._df_learning
        df_bee = df[df["has_bee"] > 0]
        df_bmN = df_bee[df_bee["has_mite"] <= 0]    # bee: yes, mite: no
        df_bmY = df[df["has_mite"] > 0]             # bee: yes, mite: yes
        df_bN = df[df["has_bee"] <= 0]              # bee: no
        dct = {}
        dct["bYmY"] = len(df_bmY)
        dct["bYmN"] = len(df_bmN)
        dct["bN"] = len(df_bN)
        
        wedge = [ dct["bYmY"], dct["bYmN"], dct["bN"] ]
        label = ["Bee \n(with mites)\n{}".format(dct["bYmY"]),
                 "Bee \n(no mites)\n{}".format(dct["bYmN"]),
                 "No Bee\n{}".format(dct["bN"])]
        color = ["royalblue", "dodgerblue", "yellow"]
        title = "Learning CSV"
        # plt.close('all')
        
        # make new figure
        self.fig, self.ax = plt.subplots(1)
        
        self.ax.pie(wedge, labels=label, autopct='%1.1f%%', colors=color, startangle=90)
        self.ax.set_title(title)
        
        self.fig.tight_layout()
        self.fig.show()
        
        pass


# %% 

if __name__== "__main__":
    print("## Calling main function.)\n")
    
    print("cv2.version = {}".format(cv2.__version__))
    print("numpy.version = {}".format(np.__version__))
    print("matplotlib.version = {}".format(mpl.__version__))
    print("pandas.version = {}".format(pd.__version__))
    print()
    
    
    # Window Cleanup
    cv2.destroyAllWindows()
    plt.close('all')
    
    TEST = 1
    
    # %%
    if TEST == 1:
        path_extr = "extracted"
        path_mites = "D:\\ECM_PROJECT\\milben_img\\Scanner\\extr"
        
        myGLIC = GenerateLearningImagesClass(path_extr, path_mites)
        myGLIC.generate()
        
        #%%
        myGLIC.pie()
        
        
        pass
    
    
    
    
    
    
    
    
    
        