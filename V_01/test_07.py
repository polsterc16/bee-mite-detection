# -*- coding: utf-8 -*-
"""
Spyder Editor

(MEAN on resize)
RESIZE
DIFF
THRES

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import GetImageList as GIL


path_img = "../images/"

cv2.destroyAllWindows()
plt.close('all')



# %%

class img_handler_mean_diff_sobel:
    def __init__(self,img_path="../images/", img_dim=(400,300), \
                 mean_alpha=0.1, threshold=0.5):
        self.set_img_path(img_path)
        self.set_img_dim(img_dim)
        # self.set_blurr_gaussian_size(blurr)
        self.set_mean_alpha(mean_alpha)
        self.set_threshold_relative(threshold)
        
        
        print("-- Handler Object created")
        self.restart()
        pass
    
    def set_img_path(self,img_path):
        self.img_path = img_path
        pass
    def set_img_dim(self, img_dim):
        # image dimension must be an int tuple
        assert type(img_dim) in [list,tuple]
        assert len(img_dim) == 2
        self.img_dim = ( int(img_dim[0]), int(img_dim[1]) )
        pass
    # def set_blurr_gaussian_size(self, blurr_gaussian_size):
    #     # gauss kernel size msut be an odd integer
    #     self.blurr_gaussian_size = int(blurr_gaussian_size)
    #     if (self.blurr_gaussian_size % 2) != 1:
    #         self.blurr_gaussian_size += 1
    #     pass
    def set_mean_alpha(self, mean_alpha):
        self.mean_alpha = float(mean_alpha)
        pass
    def set_threshold_relative(self, threshold_relative):
        self.threshold_relative = float(threshold_relative)
        pass
    
    
    def load_img(self, index):
        # get image path to indexed element in img_name_list
        imgpath_current = self.img_path + self.img_name_list[index]
        
        # set original image and resized version
        self.img_set_original = cv2.imread(imgpath_current, 0)
        self.img_set_resize = cv2.resize(self.img_set_original, \
                                     self.img_dim, \
                                     interpolation = cv2.INTER_AREA )
        pass
    # def blurr_img(self):
    #     ksize = (self.blurr_gaussian_size,self.blurr_gaussian_size)
    #     self.img_set_blurred = cv2.GaussianBlur(self.img_set_diff, ksize, 0)
    #     pass
    def weighted_mean(self, overwrite=False):
        if (overwrite):
            # overwrite mean to current blurred image
            self.img_set_mean = np.float32( self.img_set_resize )
        else:
            # weighted accumulation
            cv2.accumulateWeighted(self.img_set_resize, \
                                    self.img_set_mean, \
                                    self.mean_alpha)
        pass
    def difference(self):
        # self.img_set_diff = np.int16(self.img_set_mean) - np.int16(self.img_set_resize)
        self.img_set_diff = np.int16(self.img_set_mean) - np.int16(self.img_set_resize)
        pass
    
    def threshold(self):
        _,img = cv2.threshold(self.img_set_diff,0,255,cv2.THRESH_TOZERO)
        self.img_set_blurr = cv2.GaussianBlur(np.uint8(img),(5,5),0)
        self.img_blurr_max = np.max(self.img_set_blurr)
        # self.img_blurr_thres = int(self.img_blurr_max*self.threshold_relative)
        
        mean=np.mean(self.img_set_blurr)
        std=np.std(self.img_set_blurr)
        self.img_blurr_thres = int(mean+std*1.2)
        _,self.img_set_threshold = cv2.threshold(self.img_set_blurr,self.img_blurr_thres,255,cv2.THRESH_TOZERO)
        pass
    
    def restart(self):
        print()
        print("-- Restarting Handler")
        
        # reset (create) the img index
        self.img_index = 0
        
        # clear (create) the list of image names and fill them
        self.img_name_list = []
        GIL.get_image_list(self.img_name_list, self.img_path, "jpg")
        print("Image path: {}".format(self.img_path))
        
        # load the first image
        self.load_img(self.img_index)
        print("Images are resized to: {}".format(self.img_dim))
        
        # perform weighted mean (set mean to current image)
        self.weighted_mean(overwrite=True)
        print("Weighted Mean Alpha: {}".format(self.mean_alpha))
        
        
        pass
    
    def iterate(self, times = 1):
        for i in range(times):
            # increment the index and check if index is still inside the list
            self.img_index += 1
            if (self.img_index >= len(self.img_name_list)):
                print("No more Iterations possible. \
                      Index has reached end of img_name_list.")
                return
            
            # update weighted mean (BEFORE loading a new image)
            self.weighted_mean()
            
            # load the current image
            self.load_img(self.img_index)
            
            # calc difference from blurr to mean
            self.difference()
            
            
            # otsus threshold
            self.threshold()
        
        if(times==1):
            print("Max diff: {}".format(self.img_blurr_max))
            print("Threshold: {} ({})".format(self.img_blurr_thres,self.threshold_relative))
        pass
    
    def iter_and_plot(self, times = 1):
        self.iterate(times)        
        plt.close('all')
        
        self.fig, self.ax = plt.subplots( 2,2 )
        
        row,col = 0,0
        title = "foto: {}".format(self.img_index)
        self.ax[row,col].imshow(self.img_set_resize, cmap = 'gray',vmin=0, vmax=255) 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        
        row,col = 0,1
        title = "mean"
        self.ax[row,col].imshow(self.img_set_mean, cmap = 'gray',vmin=0, vmax=255) 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        row,col = 1,0
        title = "diff"
        self.ax[row,col].imshow(self.img_set_diff, cmap = 'gray',vmin=-255, vmax=255) 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        row,col = 1,1
        title = "thres"
        self.ax[row,col].imshow(self.img_set_threshold, cmap = 'gray') 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        
    def iter_and_plot_update(self, times = 1):
        self.iterate(times)        
        # plt.close('all')
        
        # fig,ax = plt.subplots( 2,2 )
        
        
        row,col = 0,0
        title = "foto: {}".format(self.img_index)
        self.ax[row,col].imshow(self.img_set_resize, cmap = 'gray',vmin=0, vmax=255) 
        self.ax[row,col].title.set_text(title)
        
        row,col = 0,1
        self.ax[row,col].imshow(self.img_set_mean, cmap = 'gray',vmin=0, vmax=255) 
        
        row,col = 1,0
        self.ax[row,col].imshow(self.img_set_diff, cmap = 'gray',vmin=-255, vmax=255) 
        
        row,col = 1,1
        self.ax[row,col].imshow(self.img_set_threshold, cmap = 'gray') 


# %%
plt.close('all')
# %%








# %%
if __name__== "__main__":
    print("Calling main function.)\n")
    # main()
    
    myHandler = img_handler_mean_diff_sobel(mean_alpha=0.05)
    # %%
    myHandler.iterate(times=8)
    # %%
    myHandler.iter_and_plot(times=1)
    # %%
    myHandler.iter_and_plot_update(times=1)
    # %%
    
    
    
    
    
    
    
    
    
    
    
    