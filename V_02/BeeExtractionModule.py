# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:55:24 2021

@author: Admin

PURPOSE: Find appropriate filters that allow for finding regions containing Bees. And Extract them.



# Based on V_01/test_12.py

Switch between branches with 'git checkout <branch>'



"""
#%%

version = "004" 


#%% IMPORTS

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import time
import math

from tqdm import tqdm

# Own Modules
import ImageHandlerModule as IHM
import PlotHelperModule as PHM


# %% CLASS DEFINES

class ParentImageClass:
    def __init__(self, ILO: IHM.ImageLoaderClass, 
                 ref_bg_img, 
                 alpha_bg,
                 index:int,
                 path_extr:str,
                 gauss_blurr_size=5, otsu_min_threshold=10,
                 open_close_kernel_size = 7,
                 pixel_area_min=1000, pixel_area_max=6000, 
                 focus_img_size=(128,128),
                 focus_bg_gauss_kernel_size=11,
                 focus_dilate_kernel_size=32,
                 DEBUG=False):
        self._ILO = ILO
        self._path_ILO = self._ILO._IFC_path
        
        self._ref_bg_img = ref_bg_img
        self._alpha_bg = alpha_bg
        
        self._index = index
        self.set_path_extracted(path_extr)
        
        self.set_parent_gauss_blurr_size(gauss_blurr_size)
        self.set_parent_otsu_min_threshold(otsu_min_threshold)
        self.set_parent_open_close_kernel(open_close_kernel_size)
        self.set_parent_pixel_area_min_max(pixel_area_min, pixel_area_max)
        
        self.set_focus_img_size(focus_img_size)
        self.set_focus_bg_gauss_kernel_size(focus_bg_gauss_kernel_size)
        self.set_focus_dilate_kernel_size(focus_dilate_kernel_size)
        
        # init some child objects/vars
        self.contour_list_valid = []
        self.contour_list_raw = []
        self.child_list = []
        
        # init the img dictionary
        self.img = dict()
        
        
        self.process_1(DEBUG=DEBUG)
        self.process_2(DEBUG=DEBUG)
        self.process_3()
        
        pass
    
    ### -----------------------------------------------------------------------
    ### PROCESS FUNCTIONS
    ### -----------------------------------------------------------------------
    
    def process_1(self,DEBUG=False) -> int:
        """Get gray source image, get difference from bg, get otsu_th, get contours (unless otsu th is too low)"""
        self.p00_fetch_src_gray()   # get the grayscale source image
        self.p10_diff_from_bg()     # get the difference from the BG
        otsu_th = self.p20_threshold_diff()     # get otsu_th value from thresholding
        
        
        self.contour_list_raw = []  # make empty list for contours
        # --------------------------------------
        if otsu_th < self.parent_otsu_min_th:
            # This image has no bees
            pass
        else:# ---------------------------------
            # This image has bees, maybe
            self.p25_openclose_threshold()
            self.p30_contours_extract_raw()
        # --------------------------------------
        
        if DEBUG:
            imgs = [self.img["00 gray"], self.img["10 diff"], self.img["20 threshold"]]
            labels=["00 gray","10 diff","20 threshold"]
            
            if len(self.contour_list_raw) > 0:
                imgs.append(self.img["25 reduced"])
                labels.append("25 reduced")
            mySIV = PHM.SimpleImageViewer(imgs,None,labels, "process_1 {}".format(self._index))
            pass
        
        return len(self.contour_list_raw)
    
    def process_2(self, DEBUG=False, debug_img=False):
        """(Generate debug images,) Check for valid area sizes of contours,
        Generate bee_focus objects."""
        # if debug_img:   self.p31_contours_debug(DEBUG=DEBUG);
        # else:           self.img["31 debug"] = None; pass
        
        self.p40_contours_check(DEBUG=DEBUG)
        
        self.p50_generate_focus_imgs()
        self.p51_generate_roi_img()
        pass
    
    def process_3(self, DEBUG=False):
        self.p60_save_imgs()
        self.p70_prepare_panda()
        pass
    
    ### -----------------------------------------------------------------------
    
    
    def p00_fetch_src_gray(self, DEBUG=False):
        """fetch the first image (grayscale of original)"""
        self.img["01 orig"] = self._ILO.get_img_orig(self._index)
        self.img["00 gray"] = self._ILO.get_img(self._index)
        
        self._path_parent = self._ILO.f_path
        self._fname_parent = self._ILO.f_name
        
        if DEBUG:
            imgs = [self.img["00 gray"]]
            labels=["00 gray"]
            mySIV = PHM.SimpleImageViewer(imgs,None,labels, "p00_fetch_src_gray")
        pass
    
    def p10_diff_from_bg(self, DEBUG=False):
        """get the int16 difference image, convert it to uint8 (adn updates the BG img)"""
        s1=self._ref_bg_img.shape
        s2=self.img["00 gray"].shape
        if s1 != s2: raise Exception("BG image and GRAY image do not have the same shape!")
        
        # calc diff
        diff_int16 = img_diff = np.int16(self._ref_bg_img) - np.int16(self.img["00 gray"])
        # update bg img
        cv2.accumulateWeighted( self.img["00 gray"], self._ref_bg_img, self._alpha_bg)
        
        # clean up difference image
        # This ignores artefacts from bees leaving the image (which would be negative)
        _,diff_int16 = cv2.threshold(diff_int16,0,255,cv2.THRESH_TOZERO)
        
        self.img["10 diff"] = np.uint8(diff_int16)
        
        if DEBUG:
            imgs = [self.img["00 gray"], self.img["10 diff"]]
            labels=["00 gray","10 diff"]
            mySIV = PHM.SimpleImageViewer(imgs,None,labels, "p10_diff_from_bg")
        pass
    
    def p20_threshold_diff(self, DEBUG=False):
        """Performs gaussian blurr on difference image.
        
        Thresholds the blurred difference image with OTSU algorithm. 
        (If the OTSU threshold is below the min_th_value, then the output image 
        MUST be viewed as 'empty'.)"""
        
        # perform gaussian blurr 
        diff_blurred = cv2.GaussianBlur(self.img["10 diff"],self.parent_gauss_blurr_kernel,0)
        
        # perform threshold with OTSU, but be careful about its threshold value!
        otsu_threshold, img_otsu = cv2.threshold(diff_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        self.img["20 threshold"] = img_otsu
        
        if DEBUG:
            imgs = [self.img["00 gray"], self.img["10 diff"], self.img["20 threshold"]]
            labels=["00 gray","10 diff","otsu {}".format(otsu_threshold)]
            mySIV = PHM.SimpleImageViewer(imgs,None,labels, "p20_threshold_diff")
        
        return otsu_threshold
    
    def p25_openclose_threshold(self, DEBUG=False):
        """Performs Closing (remove black holes) and then Opening 
        (remove lone white pixels) on the Threshold image (source)."""
        img = self.img["20 threshold"]
        # TODO: Test, if it works better with one additional erosion and dilation!
        img = cv2.erode(img, self.parent_open_close_kernel)
        
        img_closed = cv2.morphologyEx( img, cv2.MORPH_CLOSE, 
                                      kernel=self.parent_open_close_kernel, 
                                      iterations=self.parent_open_close_iter )
        
        img_opened = cv2.morphologyEx( img_closed, cv2.MORPH_OPEN,  
                                      kernel=self.parent_open_close_kernel, 
                                      iterations=self.parent_open_close_iter )
        
        img = cv2.dilate(img_opened, self.parent_open_close_kernel)
        
        self.img["25 reduced"] = img
        
        if DEBUG:
            imgs = [self.img["00 gray"],self.img["20 threshold"], img_closed, img_opened]
            labels = ["00 gray", "20 threshold", "img_closed", "img_opened"]
            mySIV = PHM.SimpleImageViewer(imgs, None, labels, "p25_openclose_threshold")
        pass
    
    def p30_contours_extract_raw(self, DEBUG=False):
        """Extract outermost contours to detect possible bees."""
        # We are only interested in the outermost contours (EXTERNAL), 
        #   because everything else does not make sense to handle (bees inside 
        #   a different detected object)
        _,contours, _ = cv2.findContours(self.img["25 reduced"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contour_list_raw = contours
        pass
    
    def p31_contours_debug(self, DEBUG=False):
        """Creates a debug image after getting the contours"""
        raise Exception("Do not use this anymore!")
    
        # fetch gray source image
        img = cv2.cvtColor( self.img["00 gray"].copy(), cv2.COLOR_GRAY2BGR )
        
        # if there are no contours, just make GRAY teh DEBUG image and stop
        if len(self.contour_list_raw) > 0: 
            img = np.float32( img ) #cast to float for weighted addition
            
            # make overlay for the weighted addition with the discovered blobs
            img_overlay = np.zeros(img.shape,dtype=np.uint8)
            img_overlay[:,:,2] = self.img["25 reduced"].copy() # write the reduced img to the red channel
            # perform weighted add of overlay
            cv2.accumulateWeighted(img_overlay, img, 0.333, mask=img_overlay[:,:,2])
            img = np.uint8(img)
            
            # reset overlay
            img_overlay = np.zeros(img.shape,dtype=np.uint8)
            
            info_list=[]    # get a list full of the center coordinates and the area
            for c in self.contour_list_raw:
                M = cv2.moments(c)
                A = M["m00"]
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                info_list.append((cx,cy,A))
            
            #add text
            fontFace = cv2.FONT_HERSHEY_PLAIN 
            color=(255,127,0) #blue
            size = 1
            (tx,ty),_=cv2.getTextSize("A", fontFace, size, 2)
            
            a_min,a_max = self.parent_area_min_max
            for i in range(len(info_list)):
                cx,cy,A=info_list[i]
                if A<a_min:     check = "-"
                elif (A>a_max): check = "X"
                else:           check = "+"
                txt = "{}: {} {}".format( i, int(A), check)
                # write index and area to top left
                cv2.putText(img_overlay, txt, (1,(i+1)*(ty+1)), fontFace, size, (255,255,255)) #blue
                # write index to center
                cv2.putText(img_overlay, str(i), (cx,cy), fontFace, size, (0,255,255)) #yellow
            img = cv2.bitwise_or(img, img_overlay) # hard burning of overly into img
            pass
        
        self.img["31 debug"] = img
        # REEEEEEALLY low quality saving
        # path = os.path.join(self.path_extracted, "DEBUG/{}_debug.jpg".format())
        # cv2.imwrite("0.jpg",myPar.img["31 debug"],[cv2.IMWRITE_JPEG_QUALITY,20])
        
        if DEBUG:
            imgs = [self.img["31 debug"],]
            labels = ["31 debug",]
            mySIV = PHM.SimpleImageViewer(imgs, None, labels, "p31_contours_debug",posX=900)
        pass
    
    def p40_contours_check(self, DEBUG=False) -> int:
        """Check the extracted contours if they are within min/max ares sizes."""
        self.contour_list_valid = []
        if len(self.contour_list_raw)==0: return
        
        contours = self.contour_list_raw
        
        # calc all areas
        contour_areas = [int( cv2.contourArea(c) ) for c in contours]
        
        # check all areas for their size
        (min_a, max_a) = self.parent_area_min_max
        checklist = [(a>=min_a and a<=max_a) for a in contour_areas]

        # based on this True/False list, the final contour_list is filled
        for i in range(len(contours)):
            if checklist[i]:
                self.contour_list_valid.append( (i,contours[i]) )
                
        return len(self.contour_list_valid)
    
    def p50_generate_focus_imgs(self, DEBUG=False) -> int:
        """Generate BeeFocusImage objects from valid contours"""
        self.child_list = []
        if len(self.contour_list_valid)==0: return
        
        contours = self.contour_list_valid
        
        for ID,c in contours:
            self.child_list.append( BeeFocusImage(self,ID, c,
                                                  self._focus_bg_gauss_kernel_size,
                                                  self._focus_dilate_kernel_size))
        return len(self.child_list)
    
    def p51_generate_roi_img(self, DEBUG=False):
        """Adds all detected regeins from 'reduced' img with index and are information. 
        Adds box around 'focus 'imgs."""
        # fetch gray source image
        img_bg = cv2.cvtColor( self.img["00 gray"].copy(), cv2.COLOR_GRAY2BGR )
        img_bg_f = np.float32(img_bg)
        
        # img for foreground overlay, which will be weighted added
        img_overlay = np.zeros(img_bg.shape,dtype=np.uint8)
        if len(self.contour_list_raw) == 0: 
            # write the threshold img to the red channel
            img_overlay[:,:,2] = self.img["20 threshold"].copy() 
            # and thats it - no more information
        else:
            # write the reduced img to the red channel
            img_overlay[:,:,2] = self.img["25 reduced"].copy() 
            
            # draw dilated region and box for every bee in child list
            for bee in self.child_list:
                # Draw dilated contour to GREEN channel and remove all parts where there is something in RED channel
                t1 = img_overlay[:,:,1].copy()
                img_overlay[:,:,1] = cv2.drawContours(t1, [bee.contour_dilate], -1, 255, -1)
                
                _,temp = cv2.threshold( np.int16(img_overlay[:,:,1]) - np.int16(img_overlay[:,:,2]) , 0, 255, cv2.THRESH_BINARY)
                img_overlay[:,:,1] = np.uint8(temp)
    
                # draw the rect of the focus img in BLUE channel
                a1 = bee.pos_anchor
                a2 = (a1[0] + bee.focus_size[0] - 1, a1[1] + bee.focus_size[1] - 1)
                t2 = img_overlay[:,:,0].copy()
                img_overlay[:,:,0] = cv2.rectangle(t2, a1, a2, 255, 2)
                pass
        
        # generate overlay mask
        img_overlay_mask = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2GRAY)
         
        # Add overlay to bg img
        cv2.accumulateWeighted(img_overlay, img_bg_f, 0.333, mask=img_overlay_mask)
        
        if DEBUG:
            imgs = [img_bg,img_overlay]
            labels = ["img","img_overlay"]
            # mySIV = PHM.SimpleImageViewer(imgs, None, labels, "p51_generate_roi_img",posX=1000)
        # raise Exception()
        
        
        
        # if there are no contours, just make GRAY with THRESHOLD overlay the ROI image
        if len(self.contour_list_raw) > 0: 
            # img for foregrpund text, which will be OR-ed to the final image
            img_txt = np.zeros(img_bg.shape,dtype=np.uint8)
            # img for textbox background, which will darken the img behind the are text
            img_txtbox = np.zeros(img_bg.shape,dtype=np.uint8)
            
            # Define TEXT properties
            fontFace = cv2.FONT_HERSHEY_PLAIN 
            txt_size = 1
            (tx,ty),_=cv2.getTextSize("A", fontFace, txt_size, 2)
            
            # get min/max area sizes
            a_min,a_max = self.parent_area_min_max
            
            # go though all contours
            for i in range(len(self.contour_list_raw)):
                c = self.contour_list_raw[i]
                M = cv2.moments(c)
                A = M["m00"]
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                
                # check-symbol for area size
                if A<a_min:     check = "-" # too small
                elif (A>a_max): check = "X" # ok
                else:           check = "+" # too big
                txt_w = "{}: {} {}".format( i, int(A), check)
                pos_w = (1,(i+1)*(ty+1)) # position moves down by (ty+1) for every new line
                
                # white text: write index and area to top left
                a1,a2 = self.cv2_putText_box(img_txt, txt_w, pos_w, fontFace, txt_size, (255,255,255))
                # immediately color in the txtbox area
                cv2.rectangle(img_txtbox, a1, a2, (255,255,255), -1)
                
                # yellow text: write index to center of contour
                cv2.putText(img_txt, str(i), (cx,cy), fontFace, txt_size, (0,255,255)) #yellow
                pass
            
            # put the overlay on the BG
            
            # reduce img intensity where the textbox is
            img_txtbox = cv2.dilate(img_txtbox, np.ones((5,5), dtype=np.uint8)) #widen the text box a bot
            img_txtbox_inv = cv2.bitwise_not(img_txtbox)
            # write inverse of textbox (black) to bg img (weighted) with the positive as a mask -> should only darken the textbox area
            cv2.accumulateWeighted(img_txtbox_inv, img_bg_f, 0.5, mask=img_txtbox[:,:,0])
            
            img_bg = np.uint8(img_bg_f)
            
            # burn text into bg
            img_bg = cv2.bitwise_or(img_bg, img_txt)
            
            if DEBUG:
                imgs.append( img_bg )
                labels.append( "img_bg" )
                # mySIV = PHM.SimpleImageViewer(imgs, None, labels, "p51_generate_roi_img",posX=1000)
        
        self.img["51 roi"] = img_bg
        
        if DEBUG:
            mySIV = PHM.SimpleImageViewer(imgs, None, labels, "p51_generate_roi_img",posX=1000)
        pass
    
    def p60_save_imgs(self):
        img_roi = self.img["51 roi"]
        folder_roi = "roi"
        jpg_quality = 50
        self.path_img_roi = np.NaN    # default assignment
        
        # save the debug img, if exists (in really low quality, ofc)
        if type(img_roi) != type(None):
            self.path_img_roi = os.path.join(self.path_extracted, folder_roi, "{:06d}_roi.jpg".format(self._index))
            cv2.imwrite(self.path_img_roi, img_roi, [cv2.IMWRITE_JPEG_QUALITY,jpg_quality])
            pass
        
        self.path_bee_focus=[]
        for bee in self.child_list:
            f_name = "focus/{:06d}_{:02d}.png".format(self._index,bee.bee_ID)
            path = os.path.join(self.path_extracted, f_name)
            
            self.path_bee_focus.append( (f_name, path) )
            cv2.imwrite(path, bee.img_focus)
            pass
        pass
    
    def p70_prepare_panda(self):
        # prepate parent
        parent_series = { "src_index":  self._index,
                          "src_fname":  self._fname_parent, 
                          "src_fpath":  self._path_parent, 
                          "roi_fpath":   self.path_img_roi, 
                          "contours_raw":   len(self.contour_list_raw),
                          "contours_valid": len(self.contour_list_valid), 
                          "children_names": [n for n,p in self.path_bee_focus]
                          }
        self.ds_parent = pd.Series(parent_series)
        
        # prepare child list
        self.ds_child_list=[]
        for i in range(len(self.child_list)):
            bee = self.child_list[i]
            fname,fpath = self.path_bee_focus[i]
            ID = bee.bee_ID
            pos_center = bee.get_coords_center_bee()
            pos_anchor = bee.get_coords_anchor()
            minAreaRect = bee.get_minAreaRect()
            
            bee_series = {"index": ID,
                          "fname": fname,
                          "fpath": fpath,
                          "parent_index": self._index,
                          "parent_fname": self._fname_parent,
                          "parent_fpath": self._path_parent, 
                          "roi_fpath":   self.path_img_roi, 
                          "pos_center": pos_center,
                          "pos_anchor": pos_anchor,
                          "minAreaRect": minAreaRect
                          }
            self.ds_child_list.append(bee_series)
            pass
        pass
    
    
    ### -----------------------------------------------------------------------
    ### cv2 text functinos
    ### -----------------------------------------------------------------------
    def cv2_putText_box(self,img, text, pos, fontFace, fontScale, color, thickness=1, lineType=None, bottomLeftOrigin=None) -> ((int,int),(int,int)):
        (tx,ty),_ = cv2.getTextSize(text, fontFace, fontScale, thickness)
        cv2.putText(img, text, pos, fontFace, fontScale, color, thickness)
        a1 = pos
        a2 = (a1[0]+tx, a1[1]-ty)
        return (a1,a2)
    
    
    ### -----------------------------------------------------------------------
    ### PARENT config FUNCTIONS
    ### -----------------------------------------------------------------------
    def get_dataseries_dict(self) -> dict:
        """Returns a dict containing 'ds_parent' and 'ds_child_list'."""
        ret = {"parent": self.ds_parent,
               "children": self.ds_child_list}
        return ret
    
    def set_path_extracted(self,path_extracted):
        """Sets the directory path to save all extracted information to."""
        assert (type(path_extracted) == str)    # ensure that path is a string
        
        # Stop object creation, if no valid file path is given
        if os.path.isdir(path_extracted) == False:
            raise Exception("Requires a legal directory path!")
        
        self.path_extracted = os.path.abspath(path_extracted)
        pass
    
    def get_path_extracted(self):
        """Returns the directory path to where all extracted information shall be stored at."""
        return str(self.path_extracted)
    
    def get_path_source(self):
        """Returns the directory path to the bee images, which shall be investigated."""
        return str(self._path_ILO)
    
    def set_parent_gauss_blurr_size(self,kernel_size):
        """Creates the tuple for the gaussian blurr of the difference image. 
        Will convert to an odd number, if necessary. (minimum size = 3)"""
        temp = int(kernel_size)
        if (temp % 2) != 1: temp += 1 # make an odd number
        if temp < 3: temp = 3
        self.parent_gauss_blurr_kernel = (temp,temp)
        pass
    
    def set_parent_otsu_min_threshold(self,otsu_min_th):
        """Sets the minimum accepted threshold value generated from the OTSU algorithm."""
        if otsu_min_th < 0:
            self.parent_otsu_min_th = 0
        else:
            self.parent_otsu_min_th = int(otsu_min_th)
        pass
    
    def set_parent_open_close_kernel(self, open_close_kernel_size):
        """Creates a circular kernel of size 5 and determines the necessary 
        number of iterations to have effectively the same kernel size.
        
        (Having a big kernel makes for slow operations. Performing several 
        iterations instead is quicker and leads to nearly the same result)"""
        # create circular kernel
        ks = 5
        self.parent_open_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        # calculate the number of necessary iterations
        self.parent_open_close_iter = int( open_close_kernel_size/(ks-1) )
        if self.parent_open_close_iter < 1:
            self.parent_open_close_iter = 1
        pass
    
    def set_parent_pixel_area_min_max(self, min_a:int, max_a:int):
        """Sets the minimum and maximum area size for a detected blob to be considered to be a Bee."""
        assert min_a > 0
        assert max_a > min_a
        self.parent_area_min_max = (min_a, max_a)
        pass
    
    ### -----------------------------------------------------------------------
    ### FOCUS config FUNCTIONS
    ### -----------------------------------------------------------------------
    def set_focus_img_size(self,dim):
        """Defines the size of the focus image (which shall contain only the extracted bee).
        
        dim: tuple of two ints."""
        assert type(dim) in [list,tuple]            # ensure, that it is a tuple
        assert len(dim) == 2                        # ensure, that it has 2 items
        assert all( [type(v)==int for v in dim] )   # ensure, that it only contains ints
        
        #check if focus size is smaller than parent image size
        fw,fh = dim # focus dimensions
        pw,ph = self._ILO._scale_dim # parent dimensions
        if not fw <= pw: raise Exception("Error: set_focus_img_size: fw <= pw")
        if not fh <= ph: raise Exception("Error: set_focus_img_size: fh <= ph")
        
        self._focus_size = tuple(dim)
        pass
    
    def get_focus_img_size(self):
        """Returns the currently defined size of the focus image (which shall contain only the extracted bee).
        
        return: tuple of two ints"""
        return self._focus_size
    
    def set_focus_bg_gauss_kernel_size(self, gauss_kernel_size:int):
        self._focus_bg_gauss_kernel_size = gauss_kernel_size
        pass
    def set_focus_dilate_kernel_size(self, dilate_kernel_size:int):
        self._focus_dilate_kernel_size = dilate_kernel_size
        pass
    
#%%

class BeeFocusImage:
    def __init__(self, parent: ParentImageClass, 
                 bee_ID: int, contour, 
                 bg_gauss_kernel_size=11, dilate_kernel_size=32):
        self.parent = parent
        self.parent_orig = self.parent.img["01 orig"]
        self.parent_dim = self.parent._ILO._scale_dim
        self.bee_ID = bee_ID
        self.focus_size = self.parent._focus_size
        
        self.set_contour(contour)
        
        self.set_bg_gauss_kernel_size(bg_gauss_kernel_size)
        self.set_dilate_kernel_size(dilate_kernel_size)
        
        #perform making the focus image
        self.fetch_roi_img()
        self.generate_focus_img()
        
        pass
    
    def set_bg_gauss_kernel_size(self, kernel_size=3):
        """Creates the tuple for the gaussian blurr of the BG image. Will 
        convert to an odd number, if necessary. (minimum size = 3)"""
        temp = int(kernel_size)
        if (temp % 2) != 1: temp += 1 # make an odd number
        if temp < 3: temp = 3
        self.bg_gauss_kernel = (temp,temp)
        pass
    
    def set_dilate_kernel_size(self, dilate_kernel_size):
        """Creates a circular kernel of size 5 and determines the necessary 
        number of iterations to have effectively the same kernel size.
        
        (Having a big kernel makes for slow operations. Performing several 
        iterations instead is quicker and leads to nearly the same result)"""
        # create circular kernel
        ks = 5
        self.dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        # calculate the number of necessary iterations
        self.dilate_iter = int( dilate_kernel_size/(ks-1) )
        if self.dilate_iter < 1:
            self.dilate_iter = 1
        pass
    
    def set_contour(self,contour):
        assert type(contour) == np.ndarray # ensure, that is is a contour
        assert len(contour.shape) == 3
        self.contour = contour
        
        M = cv2.moments(self.contour)
        self.area = M["m00"]
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        self.pos_center_parent = (cx,cy)
        
        # get the minAreaRect
        self.minAreaRect = cv2.minAreaRect(self.contour)
        pass
    
    def fetch_roi_img(self):
        fw,fh = self.focus_size             # w,h of focus img
        pw,ph = self.parent_dim             # w,h of parent img
        cx,cy = self.pos_center_parent      # center position of contour
        
        # get theoretical anchor points, based on cx,cy alone
        a1 = [cx-fw//2, cy-fh//2] # top left
        a2 = [a1[0]+fw, a1[1]+fh] # bottom right
        
        conflict = 0
        # check if conflict LEFT
        if a1[0] < 0:
            conflict +=1        # inc counter
            a1[0] = 0           # set left anchor to left img border
            a2[0] = a1[0] + fw  # anjust right anchor
        # check if conflict TOP
        if a1[1] < 0:
            conflict +=1        # inc counter
            a1[1] = 0           # set top anchor to top img border
            a2[1] = a1[1] + fh  # anjust bottom anchor
        # check if conflict RIGHT
        if a2[0] > pw:
            conflict +=1        # inc counter
            a2[0] = pw          # set right anchor to right img border
            a1[0] = a2[0] - fw  # anjust left anchor
        # check if conflict BOTTOM
        if a2[1] > ph:
            conflict +=1        # inc counter
            a2[1] = ph          # set bottom anchor to bottom img border
            a1[1] = a2[1] - fh  # anjust top anchor
        
        if conflict > 2:
            raise Exception("conflict={}. This should not happen! The focus image cannot have a conflict with more than 2 borders!".format(conflict))
        
        # now, that we have the anchor points, we can fetch the ROI
        self.pos_anchor = tuple(a1)
        # ROI = img[row1:row2, col1:col2] = img[y1:y2, x1:x2]
        self.img_roi = self.parent_orig[a1[1]:a2[1], a1[0]:a2[0]].copy()
        
        # # get the offset for the contour line (for use in the focus image) based on our anchor position
        # ax,ay = self.pos_anchor
        # self.contour_roi =  np.array([ [[v[0][0]-ax,v[0][1]-ay]] for v in self.contour ])
        
        # get the offset center pos of the contour as well
        ax,ay = self.pos_anchor
        self.pos_center_roi = (self.pos_center_parent[0]-ax, self.pos_center_parent[1]-ay)
        pass
    
    def generate_focus_img(self):
        img_focus = self.img_roi.copy() # copy the ROI for use as the background
        img_focus = cv2.GaussianBlur(img_focus, self.bg_gauss_kernel,0) # blurr the background
        
        ax,ay = self.pos_anchor
        mask = np.zeros(np.flip(self.focus_size),dtype=np.uint8) # empty mask image
        self.mask_core = cv2.drawContours(mask, [self.contour], -1, 255, -1, offset=(-ax,-ay)) #draw core roi
        
        # dilate the core ROI
        self.mask_dil = cv2.dilate(self.mask_core, self.dilate_k, \
                              iterations=self.dilate_iter)
        # also perform "closing" to get rid of tiny holes in BLOB
        self.mask_dil = cv2.morphologyEx(self.mask_dil, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8) )
        self.mask_inv = cv2.bitwise_not(self.mask_dil)
        
        # Now black-out the area of Bee in ROI
        img_bg = cv2.bitwise_and(img_focus,img_focus, mask = self.mask_inv)
        
        # Take only region of logo from BG image.
        img_fg = cv2.bitwise_and(self.img_roi,self.img_roi, mask = self.mask_dil)
        
        # Put FG in BG
        self.img_focus = cv2.add(img_bg,img_fg)
        
        # we also generate the dilated contours (in parent coords: with anchor offset), 
        #   since the parent might need them for drawing
        _,c_outer, _ = cv2.findContours(self.mask_dil, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE, offset=(ax,ay))
        self.contour_dilate = c_outer[0] #there SHOULD only be ONE outer contour
        pass
    
    def get_coords_anchor(self):
        return self.pos_anchor
    
    def get_coords_center_bee(self):
        return self.pos_center_parent
    
    def get_contour_core(self):
        return self.contour
    
    def get_contour_dilate(self):
        return self.contour_dilate
    
    def get_img_focus(self):
        return self.img_focus
    
    def get_minAreaRect(self):
        return self.minAreaRect
    

#%%
class BeeExtractionHandler:
    def __init__(self, path_extraction, 
                 src_img_path, src_max_files=0, src_extension_list=("png",),
                 img_dim=(400,300), img_mask_rel=(0,0,1,1), img_resize=False,
                 bg_alpha_weight=0.1):
        self._index = 0
        
        self.set_path_extracted(path_extraction)
        self.set_IFC_properties(src_img_path, src_max_files, src_extension_list)
        self.set_ILC_properties(img_dim, img_mask_rel, img_resize)
        self.set_bg_alpha_weight(bg_alpha_weight)
        
        self.df_startup()
        
        pass
    
    # -------------------------------------------------------------------------
    # STARTUP functions 
    # -------------------------------------------------------------------------
    def set_path_extracted(self,path_extracted):
        """Sets the directory path to save all extracted information to."""
        assert (type(path_extracted) == str)    # ensure that path is a string
        
        # Stop object creation, if no valid file path is given
        if os.path.isdir(path_extracted) == False:
            raise Exception("Requires a legal directory path!")
        
        self._path_extracted = os.path.abspath(path_extracted)
        pass
    
    def get_path_extracted(self):
        """Returns the directory path to where all extracted information shall be stored at."""
        return str(self._path_extracted)
    
    def set_IFC_properties(self, src_img_path, src_max_files=0, src_extension_list=("png",)):
        """Sets the ImageFinderClass."""
        self._IFC = IHM.ImageFinderClass(src_img_path, src_extension_list, src_max_files)
        self.src_len = self._IFC.size # number of imgs in ILC
        pass
    
    def set_ILC_properties(self, img_dim=(400,300), img_mask_rel=(0,0,1,1), img_resize=False):
        """Sets the ImageLoaderClass."""
        self._ILC = IHM.ImageLoaderClass(self._IFC, dim=img_dim, mask_rel=img_mask_rel, 
                                         resize_en=img_resize, grayscale_en=True)
        pass
    
    def set_bg_alpha_weight(self, alpha=0.1):
        """Sets the BackgroundImageClass."""
        if alpha >= 0 and alpha <=1:
            self._bg_alpha_weight = alpha
        else:
            raise Exception("'bg_alpha_weight' must  be a number between 0 and 1.")
        pass
    
    def set_bg_new(self, index=0, prepare=30):
        """Will overwrite the background image data with the image at 
        'index' in the ImageLoaderClass object."""
        # find the start index for making a few bg iterations
        if prepare < 1: prepare=1
        index_start = max([0, index-prepare])
        
        img_new = self._ILC.get_img(index_start)      # load new image
        self._ref_bg_img = np.float32( img_new )     # set new img as a float array (important for weighted addition!!!)
        
        for i in range(prepare-1):
            img_new = self._ILC.get_img(index_start+1+i)      # load new image
            cv2.accumulateWeighted( img_new, self._ref_bg_img, self._bg_alpha_weight)
            
        pass
    
    def df_startup(self):
        fname_parent = "Parent"
        fname_focus =  "Focus"
        
        path_dir = self.get_path_extracted()
        self._df_fname_parent_csv =  os.path.join(path_dir, "{}_csv.csv".format(fname_parent))
        self._df_fname_parent_scsv = os.path.join(path_dir, "{}_scsv.csv".format(fname_parent))
        self._df_fname_focus_csv =  os.path.join(path_dir, "{}_csv.csv".format(fname_focus))
        self._df_fname_focus_scsv = os.path.join(path_dir, "{}_scsv.csv".format(fname_focus))
        self._txt_last_index_fname = os.path.join(path_dir, "last_index_extract.txt")
        
        # We will load from the [comma] separated value files
        fparent_exists = os.path.isfile(self._df_fname_parent_csv)
        ffocus_exists =  os.path.isfile(self._df_fname_focus_csv)
        
        if fparent_exists:
            self._df_parent = pd.read_csv(self._df_fname_parent_csv, index_col=0)
        else:
            cols_parent =  ["src_index",
                            "src_fname",
                            "src_fpath",
                            "roi_fpath",
                            "contours_raw",
                            "contours_valid",
                            "children_names"]
            self._df_parent = pd.DataFrame(columns=cols_parent)
            pass
        
        if ffocus_exists:
            self._df_focus = pd.read_csv(self._df_fname_focus_csv, index_col=0)
        else:
            cols_focus =   ["index",
                            "fname",
                            "fpath",
                            "parent_index",
                            "parent_fname",
                            "parent_fpath",
                            "roi_fpath",
                            "pos_center",
                            "pos_anchor",
                            "minAreaRect"]
            self._df_focus = pd.DataFrame(columns=cols_focus)
        pass
    
    def df_store(self):
        
        self._df_parent.to_csv(self._df_fname_parent_csv,  sep=",")
        self._df_parent.to_csv(self._df_fname_parent_scsv, sep=";")
        self._df_focus.to_csv(self._df_fname_focus_csv,  sep=",")
        self._df_focus.to_csv(self._df_fname_focus_scsv, sep=";")
        pass
    
    def write_last_index_to_file(self):
        """writes current index to txt"""
        with open(self._txt_last_index_fname,"w") as f:
            f.write(str(self._index))
        pass
    
    def read_last_index_to_file(self) -> int:
        """read last index from txt"""
        index = -1 #default assignment
        try:
            with open(self._txt_last_index_fname,"r") as f:
                index = int(f.readline())
        except:
            pass
        return index
    
    
    # -------------------------------------------------------------------------
    # PROCESSING functions
    # -------------------------------------------------------------------------
    def p_process(self,start_index:int, times=1, prepare_len=10) -> int:
        # check if index possible
        if type(start_index)==type(None):
            start_index = self.read_last_index_to_file()
        
        if (start_index < 0) or (start_index >= self.src_len):
            raise Exception("Index is outside of list size.")
            return -1
        self._index = start_index
        
        self.set_bg_new(self._index, min([30, self.src_len-1]) ) # setup bg image, 30 prepare iterations, if possible
        
        if times < 1: times=1   # ignore iterations less than 1
        
        # iterate 
        ret = self.p_process_iterate(times)
        return self._index
    
    def p_process_iterate(self,times=1) -> int:
        if times < 1: times = 1
        
        time_size = 200
        
        ILC = self._ILC
        # BIC = self._BIC
        path_extr = self.get_path_extracted()
        
        if times <= time_size:
            for i in tqdm(range(times),desc="Iterating (from {})".format(self._index)):
                # create parent class object
                parent = ParentImageClass(ILC, 
                                          self._ref_bg_img,self._bg_alpha_weight, 
                                          self._index, path_extr)
                ds_dict = parent.get_dataseries_dict()
                ds_parent = ds_dict["parent"]
                ds_child_list = ds_dict["children"]
                
                self._df_parent.loc[self._index] = ds_parent
                
                for ds in ds_child_list:
                    # new_series = pd.Series(ds,name=ds["fname"])
                    # self._df_focus=self._df_focus.append(new_series)
                    self._df_focus.loc[ds["fname"]] = ds
    
                # temp = self._df_focus
                # print(self._df_parent.info())
                # print(self._df_focus.info())
                
                self._index += 1 #inc index
                
                # Stop, if the last img has been reached
                if (self._index + 1) >= self.src_len: break
                pass
            
            self.df_store()
            self.write_last_index_to_file()
        else:
            j=0
            supertime = math.ceil(times/time_size)
            for i in range(supertime):
                # print("\nIteration package {}/{} (from {})".format(i+1, supertime, self._index))
                
                for k in tqdm(range( min([time_size,times-j]) ),desc="Iteration package {}/{} (from {: 4})".format(i+1, supertime, self._index)):
                    # create parent class object
                    
                    parent = ParentImageClass(ILC, 
                                              self._ref_bg_img,self._bg_alpha_weight, 
                                              self._index, path_extr)
                    ds_dict = parent.get_dataseries_dict()
                    ds_parent = ds_dict["parent"]
                    ds_child_list = ds_dict["children"]
                    
                    self._df_parent.loc[self._index] = ds_parent
                    
                    for ds in ds_child_list:
                        # new_series = pd.Series(ds,name=ds["fname"])
                        # self._df_focus=self._df_focus.append(new_series)
                        self._df_focus.loc[ds["fname"]] = ds
        
                    # temp = self._df_focus
                    # print(self._df_parent.info())
                    # print(self._df_focus.info())
                    
                    self._index += 1 #inc index
                    
                    # Stop, if the last img has been reached
                    if (self._index + 1) >= self.src_len: break
                    pass
                    
                    j+=1
                    if j>=times: break
                
                self.df_store()
                self.write_last_index_to_file()
                if j>=times: break
            
        return self._index
    
    
    
    
    
    
    
    

# %%









# %%
if __name__== "__main__":
    print("## Calling main function.)\n")
    
    print("cv2.version = {}".format(cv2.__version__))
    print("numpy.version = {}".format(np.__version__))
    print("matplotlib.version = {}".format(mpl.__version__))
    print("pandas.version = {}".format(pd.__version__))
    
    
    # Window Cleanup
    cv2.destroyAllWindows()
    plt.close('all')
    time.sleep(0.1)
    
    TEST = 1
    # %%
    if TEST == 1:
        cv2.destroyAllWindows()
        plt.close('all')
        
        path_src = "D:\\ECM_PROJECT\\bee_images_small"
        path_extr = "extracted"
        
        myB = BeeExtractionHandler(path_extr, 
                                   path_src, src_max_files=0,
                                   img_mask_rel=(0.1,0,1,1), 
                                   bg_alpha_weight=0.1)
        # myB.set_BIC_properties()
        
        #%%
        number = 100
        import datetime
        t1 = datetime.datetime.now()
        
        myB.p_process(75,number)
        
        t2 = datetime.datetime.now()
        dt = (t2-t1).total_seconds()
        speed = number/dt
        
        pass
    
    
    