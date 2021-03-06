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
print("cv2.version = {}".format(cv2.__version__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

import GetImageList as GIL

# img_path = "../images/"
img_path = "C:/Users/Admin/0_FH_Joanneum/ECM_S3/PROJECT/bee_images/01_8_2020/5"
print("img_path =",img_path)


# %%

class img_handler:
    def __init__(self, path_img=img_path, path_extracted="extracted/", \
                 reduced_img_dim=(400,300), \
                 median_filter_size=5, mean_weight_alpha=0.1, \
                 gauss_reduce_kernel=41, gauss_reduce_threshold=160, \
                 min_pixel_area=1000, dilate_kernel_size=32):
        self.set_path_img(path_img)
        self.set_path_extracted(path_extracted)
        self.set_reduced_img_dim(reduced_img_dim)
        self.set_median_filter_size(median_filter_size)
        self.set_mean_weight_alpha(mean_weight_alpha)
        self.set_gauss_reduce_kernel(gauss_reduce_kernel)
        self.set_gauss_reduce_threshold(gauss_reduce_threshold)
        self.set_min_pixel_area(min_pixel_area)
        self.set_dilate_kernel_size(dilate_kernel_size)
        
        self.init_df(list_cols = ["img_extr","img_overlay","index_overlay"])
        
        
        print("-- Handler Object created")
        self.restart()
        pass
    
    def init_df(self,list_cols):
        assert type(list_cols) in [list, tuple]
        self.df = pd.DataFrame(columns=list_cols)
        pass
    def set_path_img(self,path_img):
        if not(path_img[-1] in ['/','\\']): 
            path_img = path_img+"/"
        self.prop_path_img = path_img
        pass
    def set_path_extracted(self,path_extracted):
        assert (type(path_extracted) == str)
        if not(path_extracted[-1] in ['/','\\']): 
            path_extracted = path_extracted+"/"
        self.prop_path_extracted = path_extracted
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
        imgpath_current = self.prop_path_img + self.img_name_list[index]
        
        # set original image and resized version
        self.img_set_original = cv2.imread(imgpath_current, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(self.img_set_original, cv2.COLOR_BGR2GRAY)
        self.img_set_resize = cv2.resize(img_gray, \
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
        
        # for safety reasons, we also apply "opening" = dilate(erode(img)) - to avoi 1px spots
        self.img_set_reduced = \
            cv2.morphologyEx( self.img_set_reduced, cv2.MORPH_OPEN, np.ones((5,5),np.uint8) )
        pass
    
    def get_contours_reduced(self, source):
        img = source
        _,contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.cont_reduced = contours
        
        self.cont_good=[]
        # go through contours - throw all small ones out - remove all black islands
        for c in self.cont_reduced:
            area = int( cv2.contourArea(c) )    # get area of contour
            if area < self.prop_min_pixel_area: 
                continue    # skip, if area is too small
            
            mask = np.zeros(source.shape, np.uint8)
            cv2.drawContours(mask, c, -1, 255, -1)
            
            # get avg color in source img, that is in the contour area
            mean = cv2.mean(source, mask=mask)
            if mean[0] <= 127: 
                continue    # skip, if the avg color is too low (sign of an encaspulated black area)
            
            # if all checks are ok, then save as "good" contour
            self.cont_good.append(c)
            continue
        pass
    
    def extract_from_contours(self, source_contours, source_img, orig_img):
        cs = source_contours
        
        im_overlay_name = self.img_name_list[self.img_index]
        im_overlay_name = (im_overlay_name.split("."))[0]
        im_overlay_name = im_overlay_name + "_show_ROI.png"
        
        
        # get scaling
        dim_orig = orig_img.shape
        dim_targ = source_img.shape
        dim_scale = (dim_orig[0]/dim_targ[0], dim_orig[1]/dim_targ[1])
        
        # init masks for the overlay
        mask_core = np.zeros(source_img.shape, np.uint8)
        mask_dila = np.zeros(source_img.shape, np.uint8)
        
        i=0
        l_coords = []
        for c in cs:
            # get center of core contour
            M = cv2.moments(c)
            cx = (M['m10']/M['m00'])
            cy = (M['m01']/M['m00'])
            l_coords.append( (cx,cy) )
            
            i_core =    np.zeros(source_img.shape, np.uint8)
            
            c2=[c[:,0,:],] # change format to fit for fillPoly
            
            # draw the reduced contour
            cv2.fillPoly(i_core, c2, 255)
            
            # dilate the core shape
            i_dilate = cv2.dilate(i_core, self.prop_dilate_kernel, \
                                  iterations=self.prop_dilate_iterations)
            # also perform "closing" to get rid of tiny holes in BLOB
            i_dilate = cv2.morphologyEx(i_dilate, cv2.MORPH_CLOSE, \
                                        np.ones((5,5),np.uint8) )
                
            # store for overlay
            mask_core = np.bitwise_or(i_core,   mask_core)
            mask_dila = np.bitwise_or(i_dilate, mask_dila)  
            
            # get coontours of BLOB (ideally only 1)
            _,c_dilate, _ = cv2.findContours(i_dilate, cv2.RETR_LIST, \
                                             cv2.CHAIN_APPROX_SIMPLE)
            
            # in case of additional encased contours
            if len(c_dilate)==1:                    #
                c_dilate = c_dilate[0][:,0,:]       #
            else:                                   #
                temp = np.array(c_dilate)           #
                pts = [len(item) for item in temp]  #
                idx = np.argmax(pts)                #
                c_dilate = c_dilate[idx][:,0,:]     #
            
            # scale up for contours on orig image
            c_big = [ (pair[0]*dim_scale[0], pair[1]*dim_scale[1]) \
                     for pair in c_dilate ]
            c_big = np.array(c_big, dtype=np.int32)
            
            # get bounding rect to cut out ROI
            x,y,w,h = cv2.boundingRect(c_big)
            
            # get new contour coords for cut out ROI
            c_local = np.array( [ (pair[0]-x, pair[1]-y) for pair in c_big ], np.int32)
            
            # copy roi
            img_roi = orig_img[y:y+h,x:x+w,:].copy()
            
            # split color channels in ROI for adding alpha channel
            roi_b,roi_g,roi_r = cv2.split(img_roi)
            
            # draw the big contour in alpha
            roi_a = np.zeros(roi_b.shape, np.uint8)
            cv2.fillPoly(roi_a, (c_local,), 255)
            
            # remerge channels for BGRA-png
            img_roi = cv2.merge((roi_b, roi_g, roi_r, roi_a))
            
            # get original image name and store extracted ROI with suffix
            im_roi_name = self.img_name_list[self.img_index]
            im_roi_name = (im_roi_name.split("."))[0]
            im_roi_name = im_roi_name + "_ex_{:02d}.png".format(i)
            cv2.imwrite(self.prop_path_extracted +"imgs/" + im_roi_name, img_roi)
            
            cols=[col for col in self.df.columns]
            df2 = pd.DataFrame([[im_roi_name, im_overlay_name, i]], columns=cols)
            self.df = self.df.append(df2, ignore_index=True)
            
            i += 1
            continue
        
        # TODO! make overlay with core/dilated shade + index label-text
        
        self.img_set_dilation = mask_dila.copy()
        # dilated mask should not overlay with core mask
        mask_dila = cv2.bitwise_xor(mask_core, mask_dila)
        
        # image to add weighted accum to + emtpy overlay
        img = np.float32( cv2.cvtColor(source_img, cv2.COLOR_GRAY2BGR) )
        overlay = np.zeros(img.shape, np.uint8)
        
        overlay[:,:,2] = mask_core # write core mask to red channel
        overlay[:,:,0] = mask_dila # write dilated mask to blu channel
        
        # add overlay to img
        cv2.accumulateWeighted(overlay, img, 0.2)
        img = np.array(img,np.uint8)
        
        #add text
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        color=(255,255,255)
        size = 1
        dim,_=cv2.getTextSize("A", fontFace, size, 2)
        
        i=0
        for coord in l_coords:
            org=( int(coord[0]-(dim[0]/2)), int(coord[1]+(dim[1]/2)) )
            cv2.putText(img, str(i), org, fontFace, size, color,2)
            i += 1
        
        self.img_set_mask_RGB = img
        
        # if(self.img_index >=10):
        #     cv2.imshow("img overlay", img)
        #     cv2.waitKey(1000)
        #     print()
            
        # save overlay img
        cv2.imwrite(self.prop_path_extracted +"showROI/" + im_overlay_name, self.img_set_mask_RGB)
        
        pass
    
    # def draw_contours_reduced(self, target):
    #     img = target
    #     fontFace = cv2.FONT_HERSHEY_PLAIN
    #     color=(255,255,255)
    #     dim,_=cv2.getTextSize("A", fontFace, 2, 2)
    #     y0=dim[1]
    #     y = y0
    #     for i in range( len(self.cont_reduced) ):
    #         c=self.cont_reduced[i]
    #         area = int( cv2.contourArea(c) )
    #         text = str(i)+" "+str(area)
    #         dim,_ = cv2.getTextSize(str(i), fontFace, 1.5, 2)
            
            
    #         M = cv2.moments(c)
    #         cx = (M['m10']/M['m00'])
    #         cy = (M['m01']/M['m00'])
            
    #         org=( int(cx-(dim[0]/2)), int(cy+(dim[1]/2)) )
    #         cv2.putText(img, str(i), org, fontFace, 1.5, color,2)
            
    #         cv2.putText(img, text, (0,y), fontFace, 1.5, color,2)
    #         y +=y0 # increment to next line
    #     pass
    
    # def dilation(self, source):
    #     self.img_set_dilation = cv2.dilate(source, self.prop_dilate_kernel, \
    #                               iterations=self.prop_dilate_iterations)
    #     pass
    
    # def overlay(self,source):
    #     """img_set_mask_RGB"""
    #     mask_g = self.img_set_dilation - self.img_set_reduced
    #     mask_r = self.img_set_reduced
    #     img = np.float32( cv2.cvtColor(source, cv2.COLOR_GRAY2BGR) )
    #     maskImg = np.zeros(img.shape, np.uint8)
    #     maskImg[:,:,1] = mask_g
    #     maskImg[:,:,2] = mask_r
    #     cv2.accumulateWeighted(maskImg, img, 0.2)
    #     self.img_set_mask_RGB = np.uint8( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
    #     pass
    
    def restart(self, prepare_time=5):
        print()
        print("-- Restarting Handler")
        
        # reset (create) the img index
        self.img_index = 0
        
        # clear (create) the list of image names and fill them
        self.img_name_list = []
        GIL.get_image_list(self.img_name_list, self.prop_path_img, "jpg")
        self.img_name_list_length = len(self.img_name_list)
        print("Image path: {}".format(self.prop_path_img))
        
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
        # try:
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
            
            
            self.extract_from_contours(source_contours=self.cont_good, \
                                       source_img=self.img_set_resize, \
                                       orig_img=self.img_set_original)
            
            
            # dilate
            # self.dilation(source=self.img_set_reduced)
            
            # #overlay
            # self.overlay(source=self.img_set_resize)
            # self.draw_contours_reduced(target=self.img_set_mask_RGB)
            
            pass
        path = self.prop_path_extracted + "Extracted.csv"
        self.df.to_csv(path,sep=";")
        pass
    
    def iter_and_plot(self, times = 1):
        self.iterate(times)        
        plt.close('all')
        
        self.fig, self.ax = plt.subplots( 3,2 )
        
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        row,col = 0,0
        title = "({}) {}".format(self.img_index, self.img_name_list[self.img_index])
        self.ax[row,col].imshow(cv2.cvtColor(self.img_set_mask_RGB, cv2.COLOR_BGR2RGB ) )
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
        self.but_iter_1.color = 'cyan'
        self.but_iter_10.color = 'cyan'
        self.but_iter_20.color = 'cyan'
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
        title = "({}) {}".format(self.img_index, self.img_name_list[self.img_index])
        self.ax[row,col].imshow(cv2.cvtColor(self.img_set_mask_RGB, cv2.COLOR_BGR2RGB ) )
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
    print("Number of imgs available:",myHandler.img_name_list_length)
    # myHandler.restart(30)
    # %%
    # myHandler.iterate(times=8)
    # %%
    myHandler.iter_and_plot(times=1)
    # %%
    # myHandler.iter_and_plot_update(times=1)
    # %%
    
    
    
    
    
    