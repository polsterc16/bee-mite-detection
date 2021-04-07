# -*- coding: utf-8 -*-
"""
Spyder Editor

(MEAN on resize)
RESIZE
DIFF
THRES

gaus - thres - dilate


1) das vorherige bild wird zum MEAN dazugewichtet
2) das nächste bild wird gelesen und auf 400x300 resized (added black bar on left)
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
from matplotlib.widgets import Button

import GetImageList as GIL



# %%

class img_handler:
    def __init__(self,img_path="../images/", reduced_img_dim=(400,300), \
                 median_filter_size=5, mean_weight_alpha=0.1, \
                 gauss_reduce_kernel=41, gauss_reduce_threshold=160, \
                 min_pixel_area=1000, dilate_size=32):
        self.set_img_path(img_path)
        self.set_reduced_img_dim(reduced_img_dim)
        self.set_median_filter_size(median_filter_size)
        self.set_mean_weight_alpha(mean_weight_alpha)
        self.set_gauss_reduce_kernel(gauss_reduce_kernel)
        self.set_gauss_reduce_threshold(gauss_reduce_threshold)
        self.set_min_pixel_area(min_pixel_area)
        
        
        print("-- Handler Object created")
        self.restart()
        pass
    
    def set_img_path(self,img_path):
        self.prop_img_path = img_path
        pass
    def set_reduced_img_dim(self, reduced_img_dim):
        # image dimension must be an int tuple
        temp = reduced_img_dim
        assert type(temp) in [list,tuple]
        assert len(temp) == 2
        self.prop_reduced_img_dim = ( int(temp[0]), int(temp[1]) )
        pass
    def set_median_filter_size(self, median_filter_size):
        self.prop_median_filter_size= int(median_filter_size)
        pass
    def set_mean_weight_alpha(self, mean_weight_alpha):
        self.prop_mean_weight_alpha = float(mean_weight_alpha)
        pass
    def set_gauss_reduce_kernel(self, gauss_reduce_kernel):
        # gauss kernel size msut be an odd integer
        temp = int(gauss_reduce_kernel)
        if (temp % 2) != 1: temp += 1
        self.prop_gauss_reduce_kernel = (temp, temp)
        pass
    def set_gauss_reduce_threshold(self, gauss_reduce_threshold):
        self.prop_gauss_reduce_threshold= int(gauss_reduce_threshold)
        pass
    def set_min_pixel_area(self, min_pixel_area):
        self.prop_min_pixel_area = int(min_pixel_area)
        pass
    def set_dilate_kernel_size(self, dilate_kernel_size):
        ks = 5
        self.prop_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        self.prop_dilate_iterations = int( dilate_kernel_size/(ks-1) )
        pass
    
    
    def load_img(self, index):
        """img_set_resize"""
        # get image path to indexed element in img_name_list
        imgpath_current = self.prop_img_path + self.img_name_list[index]
        
        # set original image and resized version
        self.img_set_original = cv2.imread(imgpath_current, 0)
        self.img_set_resize = cv2.resize(self.img_set_original, \
                                     self.prop_reduced_img_dim, \
                                     interpolation = cv2.INTER_AREA )
        
        # block left side of image
        blk_width = 30
        cv2.rectangle(self.img_set_resize,\
                      (0,0),(blk_width,self.img_set_resize.shape[0]),(0),cv2.FILLED)
        pass
    
    def median(self, source):
        """img_set_median"""
        self.img_set_median = cv2.medianBlur(source, self.prop_median_filter_size)
        pass
    
    def weighted_mean(self, source, overwrite=False):
        """img_set_mean"""
        if (overwrite):
            # overwrite mean to current blurred image
            self.img_set_mean = np.float32( source )
        else:
            # weighted accumulation
            cv2.accumulateWeighted( source, self.img_set_mean, \
                                    self.prop_mean_weight_alpha)
        pass
    
    def difference_from_mean(self, source):
        """img_set_diff"""
        # self.img_set_diff = np.int16(self.img_set_mean) - np.int16(self.img_set_resize)
        self.img_set_diff = np.int16(self.img_set_mean) - np.int16(source)
        pass
    
    def threshold_diff(self, source):
        """img_set_threshold"""
        # cut off negative values
        _,img = cv2.threshold(source,0,255,cv2.THRESH_TOZERO)
        
        temp = cv2.GaussianBlur(np.uint8(img),(5,5),0)
        self.img_blurr_max = np.max(temp)
        
        mean,std = cv2.meanStdDev(temp)
        self.img_blurr_thres = int(mean + std*1.2)
        # _,self.img_set_threshold = cv2.threshold(self.img_set_blurr,self.img_blurr_thres,255,cv2.THRESH_TOZERO)
        _,self.img_set_threshold = \
            cv2.threshold(temp, self.img_blurr_thres, 255, cv2.THRESH_BINARY)
        pass
    
    def gauss_blurr_reduce(self, source):
        """img_set_reduced"""
        temp = cv2.GaussianBlur(source, self.prop_gauss_reduce_kernel, 0 ) # blurr the images
        _,self.img_set_reduced = \
            cv2.threshold(temp, self.prop_gauss_reduce_threshold, 255, \
                          cv2.THRESH_BINARY)
        pass
    
    def get_contours_reduced(self, source):
        img = source
        _,contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.cont_reduced = contours
        
        self.cont_good=[]
        # go through contours - throw all small ones out - remove all black islands
        for i in range( len(self.cont_reduced) ):
            c=self.cont_reduced[i]
            
            area = int( cv2.contourArea(c) )
            if area < self.prop_min_pixel_area: continue
            
            mask = np.zeros(source.shape, np.uint8)
            cv2.drawContours(mask, c, -1, 255, -1)
            mean = cv2.mean(source, mask=mask)
            if mean[0] <= 127: continue
            self.cont_good.append(c)
            continue
        pass
    
    def extract_from_contours(self,source_contours):
        cs = source_contours
        # TODO
        pass
    
    def draw_contours_reduced(self, target):
        img = target
        fontFace = cv2.FONT_HERSHEY_PLAIN
        color=(255,255,255)
        dim,_=cv2.getTextSize("A", fontFace, 2, 2)
        y0=dim[1]
        y = y0
        for i in range( len(self.cont_reduced) ):
            c=self.cont_reduced[i]
            area = int( cv2.contourArea(c) )
            text = str(i)+" "+str(area)
            dim,_ = cv2.getTextSize(str(i), fontFace, 1.5, 2)
            
            M = cv2.moments(c)
            cx = (M['m10']/M['m00'])
            cy = (M['m01']/M['m00'])
            
            org=( int(cx-(dim[0]/2)), int(cy+(dim[1]/2)) )
            cv2.putText(img, str(i), org, fontFace, 1.5, color,2)
            
            cv2.putText(img, text, (0,y), fontFace, 1.5, color,2)
            y +=y0 # increment to next line
        pass
    
    def dilation(self, source):
        ks=31
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        self.img_set_dilation = cv2.dilate(source,kernel,iterations = 1)
        pass
    
    def overlay(self,source):
        """img_set_mask_RGB"""
        mask_g = self.img_set_dilation - self.img_set_reduced
        mask_r = self.img_set_reduced
        img = np.float32( cv2.cvtColor(source, cv2.COLOR_GRAY2BGR) )
        maskImg = np.zeros(img.shape, np.uint8)
        maskImg[:,:,1] = mask_g
        maskImg[:,:,2] = mask_r
        cv2.accumulateWeighted(maskImg, img, 0.2)
        self.img_set_mask_RGB = np.uint8( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
        pass
    
    def restart(self, prepare_time=10):
        print()
        print("-- Restarting Handler")
        
        # reset (create) the img index
        self.img_index = 0
        
        # clear (create) the list of image names and fill them
        self.img_name_list = []
        GIL.get_image_list(self.img_name_list, self.prop_img_path, "jpg")
        print("Image path: {}".format(self.prop_img_path))
        
        # load the first image
        self.load_img(self.img_index)
        print("Images are resized to: {}".format(self.prop_reduced_img_dim))
        
        # perform median filter
        # self.median(self.img_set_resize)
        
        # perform weighted mean (set mean to current image)
        self.weighted_mean(source=self.img_set_resize, overwrite=True)
        # self.weighted_mean(source=self.img_set_median, overwrite=True)
        
        print("Weighted Mean Alpha: {}".format(self.prop_mean_weight_alpha))
        
        
        self.prepare(prepare_time)  # prepare the mean image
        
        pass
    
    def prepare(self, times=10):
        times = int(times)
        
        self.img_index = 0
        if (times >= len(self.img_name_list)):
            print("Not enough items in 'self.img_name_list'.")
            return
        
        for i in range(times):
            # load next image
            self.load_img(self.img_index)
            
            # get median
            # self.median(source=self.img_set_resize)
            
            # update weighted mean (BEFORE loading a new image)
            self.weighted_mean(source=self.img_set_resize)
            # self.weighted_mean(source=self.img_set_median)
            
            self.img_index += 1     #increment index
        
        self.img_index = 0      # reset index
        
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
            self.weighted_mean(source=self.img_set_resize)
            
            # load the current image
            self.load_img(self.img_index)
            
            # self.median(source=self.img_set_resize)
            
            # calc difference from blurr to mean
            self.difference_from_mean(source=self.img_set_resize)
            
            
            # threshold
            self.threshold_diff(source=self.img_set_diff)
            
            # gaussian blurr
            self.gauss_blurr_reduce(source=self.img_set_threshold)
            
            # selection and seperation
            self.get_contours_reduced(source=self.img_set_reduced)
            
            # dilate
            self.dilation(source=self.img_set_reduced)
            
            #overlay
            self.overlay(source=self.img_set_resize)
            self.draw_contours_reduced(target=self.img_set_mask_RGB)
            
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
        
        #adding buttons
        plt.subplots_adjust(top=0.9)
        plt.subplots_adjust(bottom=0,left=0,right=1)
        ax_iter_1 =  plt.axes([0.1, 0.94, 0.2, 0.05]) #posx, posy, width, height
        ax_iter_10 = plt.axes([0.4, 0.94, 0.2, 0.05])
        ax_iter_20 = plt.axes([0.7, 0.94, 0.2, 0.05])
        self.but_iter_1 =  Button(ax_iter_1,  "Iter(1)")
        self.but_iter_10 = Button(ax_iter_10, "Iter(10)")
        self.but_iter_20 = Button(ax_iter_20, "Iter(20)")
        self.but_iter_1.on_clicked(self.on_click_iter_1)
        self.but_iter_10.on_clicked(self.on_click_iter_10)
        self.but_iter_20.on_clicked(self.on_click_iter_20)
        pass
    
    def on_click_iter_1(self,event):
        print("on_click_iter_1")
        self.iter_and_plot_update(1)
        pass
    def on_click_iter_10(self,event):
        print("on_click_iter_10")
        self.iter_and_plot_update(10)
        pass
    def on_click_iter_20(self,event):
        print("on_click_iter_20")
        self.iter_and_plot_update(20)
        pass
    
    
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
        
        plt.draw()
        pass


# %%









# %%
if __name__== "__main__":
    print("Calling main function.)\n")
    # main()
    cv2.destroyAllWindows()
    plt.close('all')
    # %%
    
    myHandler = img_handler(mean_weight_alpha=0.05)
    myHandler.restart(30)
    # %%
    # myHandler.iterate(times=8)
    # %%
    myHandler.iter_and_plot(times=1)
    # %%
    # myHandler.iter_and_plot_update(times=1)
    # %%
    
    
    
    
    
    