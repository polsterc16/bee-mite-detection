# -*- coding: utf-8 -*-
"""
Created on Tue May 11 09:37:02 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


class SimpleImageViewer:
    """
    Shows the given images in the specified grid

    Parameters
    ----------
    grid : tuple of two ints
        defines the grid layout of the subplots.
    imgs : tuple or lsit of cv2-images
        images to be plotted.
    labels : list of strings, optional
        will be used for titles of the subplots. The default is None.
    windowname : String, optional
        Label of the Window. The default is None.
    """
    
    def __init__(self,grid,imgs,labels=None, windowname=None, 
                 posX=100, posY=100, w=650, h=550):
        
        # enforce correct data types for grid
        assert type(grid) in [list, tuple]
        assert type(grid[0]) == int
        assert type(grid[1]) == int
        self.n_rows = grid[0]
        self.n_cols = grid[1]
        
        assert all([type(v)==int for v in (posX,posY,w,h)])
        self.posX = posX
        self.posY = posY
        self.w = w
        self.h = w
        
        #enforce that the imgs must be a list of cv2 images (np arrays)
        assert type(imgs) in [list,tuple]
        for item in imgs:
            assert type(item)==np.ndarray
        self.image_list = imgs
        
        # make sure, that the lable list is a list or make it an empty tuple
        if type(labels) in [list, tuple]:
            self.label_list = labels
        else:
            self.label_list = tuple()
        
        # get window title (and format it)
        now = datetime.now()
        if windowname==None:
            self.w_name = str(now)
        else:
            self.w_name = str(windowname) #+ " " + str(now)
        
        
        #show the window
        self.show()
        pass
    
    def show(self):
        
        # make new subplots as specified bythe  grid
        self.fig, self.ax = plt.subplots( self.n_rows,self.n_cols, num=self.w_name )
        #adding buttons
        plt.tight_layout()
        
        # make a list of the possible indexes
        index_list=[]
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                index_list.append((i,j))
        index_length = len(index_list)
        
        if (self.n_rows == 1) and  (self.n_cols == 1):
            index_list = [0,]
        if (self.n_rows == 1) and  (self.n_cols != 1):
            index_list = [j for j in range(self.n_cols)]
        elif (self.n_rows != 1) and  (self.n_rows == 1):
            index_list = [i for i in range(self.n_rows)]
        
        img_length = len(self.image_list)
        
        # remove the grid from all plots
        for idx in index_list:
            if img_length == 1:
                self.ax.axis("off")
                self.ax.axis("off")
            else:
                self.ax[idx].axis("off")
                self.ax[idx].axis("off")
        
        # limit the forloop iterations to the max plottable number of images in the grid
        max_plots = min([self.n_rows*self.n_cols, img_length])
        
        for i in range( min([img_length,max_plots]) ):
            idx = index_list[i]
            img = self.image_list[i]
            try:
                #plot image
                if len(img.shape)==2: #we have a grayscale image
                    if img_length == 1:
                        self.ax.imshow(img,cmap='gray',vmin=0, vmax=255)
                    else:
                        self.ax[idx].imshow(img,cmap='gray',vmin=0, vmax=255) 
                else: # we have a BGR image
                    if img_length == 1:
                        self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    else:
                        self.ax[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                #set title from labellist (if any)
                if i < len(self.label_list):
                    if img_length == 1:
                        self.ax.title.set_text( str(self.label_list[i]) )
                    else:
                        self.ax[idx].title.set_text( str(self.label_list[i]) )
            except:
                raise Exception('Problem at index {} when plotting imgs'.format(str(idx)))
        
        
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(self.posX, self.posY, self.w, self.h)
        plt.show()
        pass
    
    
    
    
        # START
    
    
    
#%%
if __name__ == '__main__':
    import os
    
    myPath = "C:\\Users\\Admin\\0_FH_Joanneum\\ECM_S3\\PROJECT\\bee_images\\01_8_2020\\5"
    myFile1 = "0_0_image0000_0.jpg"
    myFile2 = "0_0_image0002_0.jpg"
    myFile3 = "0_0_image0003_0.jpg"
    myFile4 = "0_0_image0004_0.jpg"
    
    path = os.path.join(myPath,myFile1)
    im1 = cv2.imread(path)
    
    path = os.path.join(myPath,myFile2)
    im2 = cv2.imread(path)
    
    path = os.path.join(myPath,myFile3)
    im3 = cv2.imread(path)
    
    path = os.path.join(myPath,myFile4)
    im4 = cv2.imread(path)
    
    im1 = cv2.resize(im1, (400,300), interpolation = cv2.INTER_AREA )
    im2 = cv2.resize(im2, (400,300), interpolation = cv2.INTER_AREA )
    im3 = cv2.resize(im3, (400,300), interpolation = cv2.INTER_AREA )
    im4 = cv2.resize(im4, (400,300), interpolation = cv2.INTER_AREA )
    
    im5 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im6 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    plt.close('all')
    imlist=[im2,im3,im4,im5,im6]
    # mySIV = SimpleImageViewer((3,2),imlist)
    
    labels = ["im2 col","im3 col","im3 col","im3 col","im3 col","im3 col","im3 col"]
    mySIV = SimpleImageViewer((2,2),imlist,labels,"test 123 test")
    
    
    
    
    
    
    