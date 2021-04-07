# -*- coding: utf-8 -*-
"""
Spyder Editor

(MEAN on resize)
RESIZE
DIFF
THRES

gaus - thres - dilate


1) das vorherige bild wird zum MEAN dazugewichtet
2) das nächste bild wird gelesen und auf 400x300 resized
3) DIFFERENCE bilden zwischen RESIZE und MEAN
4) mean value & standard deviation berechnen von DIFFERENCE
5) THRES: gauss (5x5 kernel) & threshold (th=mean + 1.2*std) auf DIFF
6) REDUCED: gauss (grosser kernel) & threshold (th=200) auf THRES to get only body of bee
7) DILATE: dilate the REDUCED to get the full bee
8) make overlay mask to show results

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
                 mean_alpha=0.1, reduce_kernel=31):
        self.set_img_path(img_path)
        self.set_img_dim(img_dim)
        self.set_mean_alpha(mean_alpha)
        self.set_blurr_reduce_size(reduce_kernel)
        
        
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
    def set_mean_alpha(self, mean_alpha):
        self.mean_alpha = float(mean_alpha)
        pass
    def set_blurr_reduce_size(self, blurr_gaussian_size):
        # gauss kernel size msut be an odd integer
        temp = int(blurr_gaussian_size)
        if (temp % 2) != 1: temp += 1
        self.blurr_reduce_kernel = (temp, temp)
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
        
        mean=np.mean(self.img_set_blurr)
        std=np.std(self.img_set_blurr)
        self.img_blurr_thres = int(mean+std*1.2)
        # _,self.img_set_threshold = cv2.threshold(self.img_set_blurr,self.img_blurr_thres,255,cv2.THRESH_TOZERO)
        _,self.img_set_threshold = cv2.threshold(self.img_set_blurr, \
                                                 self.img_blurr_thres, 255, \
                                                 cv2.THRESH_BINARY)
        pass
    
    def gauss_blurr_reduce(self):    
        self.img_set_blurr_2 = cv2.GaussianBlur(self.img_set_threshold, \
                                                self.blurr_reduce_kernel, 0 ) # blurr the images
        _,self.img_set_reduced = cv2.threshold(self.img_set_blurr_2, \
                                                 200, 255, \
                                                 cv2.THRESH_BINARY)
        pass
    
    # def erosion(self):
    #     ks=15
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
    #     self.img_set_erosion = cv2.erode(self.img_set_threshold,kernel,iterations = 1)
    #     pass
    
    def dilation(self):
        ks=31
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        self.img_set_dilation = cv2.dilate(self.img_set_reduced,kernel,iterations = 1)
        pass
    
    def overlay(self):
        mask_g = self.img_set_dilation - self.img_set_reduced
        mask_r = self.img_set_reduced
        img = np.float32( cv2.cvtColor(self.img_set_resize, cv2.COLOR_GRAY2BGR) )
        maskImg = np.zeros(img.shape, np.uint8)
        maskImg[:,:,1] = mask_g
        maskImg[:,:,2] = mask_r
        cv2.accumulateWeighted(maskImg, img, 0.2)
        self.img_set_mask_RGB = np.uint8( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
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
            
            
            # threshold
            self.threshold()
            
            # gaussian blurr
            self.gauss_blurr_reduce()
            
            # threshold reduce
            # self.erosion()
            
            # dilate
            self.dilation()
            
            #overlay
            self.overlay()
            
            pass
        
        if(times==1):
            print("Max diff: {}".format(self.img_blurr_max))
            print("Threshold: {} ".format(self.img_blurr_thres))
        pass
    
    def iter_and_plot(self, times = 1):
        self.iterate(times)        
        plt.close('all')
        
        self.fig, self.ax = plt.subplots( 3,2 )
        
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        row,col = 0,0
        title = "foto: {}".format(self.img_index)
        self.ax[row,col].imshow(self.img_set_mask_RGB)
        # self.ax[row,col].imshow(self.img_set_resize, cmap = 'gray',vmin=0, vmax=255) 
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
        
        row,col = 2,0
        title = "gauss reduced"
        self.ax[row,col].imshow(self.img_set_reduced, cmap = 'gray') 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        row,col = 2,1
        title = "dilated"
        self.ax[row,col].imshow(self.img_set_dilation, cmap = 'gray') 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        
    def iter_and_plot_update(self, times = 1):
        self.iterate(times)        
        # plt.close('all')
        
        # fig,ax = plt.subplots( 2,2 )
        
        
        row,col = 0,0
        title = "foto: {}".format(self.img_index)
        self.ax[row,col].imshow(self.img_set_mask_RGB)
        # self.ax[row,col].imshow(self.img_set_resize, cmap = 'gray',vmin=0, vmax=255) 
        self.ax[row,col].title.set_text(title)
        
        row,col = 0,1
        self.ax[row,col].imshow(self.img_set_mean, cmap = 'gray',vmin=0, vmax=255) 
        
        row,col = 1,0
        self.ax[row,col].imshow(self.img_set_diff, cmap = 'gray',vmin=-255, vmax=255) 
        
        row,col = 1,1
        self.ax[row,col].imshow(self.img_set_threshold, cmap = 'gray') 
        
        row,col = 2,0
        self.ax[row,col].imshow(self.img_set_reduced, cmap = 'gray') 
        
        row,col = 2,1
        self.ax[row,col].imshow(self.img_set_dilation, cmap = 'gray') 


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
    
    
    
    
    
    
    
    
    
    
    