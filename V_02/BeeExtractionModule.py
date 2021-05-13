# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:55:24 2021

@author: Admin

PURPOSE: Find appropriate filters that allow for finding regions containing Bees. And Extract them.



# Based on V_01/test_12.py



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

from tqdm import tqdm

# Own Modules
import ImageHandlerModule as IHM
import PlotHelperModule as PHM


# %% CLASS DEFINES
class BackgroundImageClass:
    def __init__(self, ILO, start_index=0, alpha_weight=0.05):
        self._ILO = ILO
        self.set_alpha_weight(alpha_weight)
        
        self.reset(times=1)
        pass
    
    def set_alpha_weight(self,alpha_weight):
        """Defines with which weight new images are added to the background 
        in order to gain the running average.
        
        alpha_weight : float between 0 and 1. """
        assert alpha_weight >= 0
        assert alpha_weight <= 1
        self.alpha = alpha_weight
        pass
    
    def reset(self, start_index=0, times=1):
        """Will set the background image to the image at 'start_index' 
        in the ImageLoaderClass object and then update a number of 
        [times - 1] times. (overall: still uses [times] images)"""
        assert start_index >= 0     # start_index must be positive int
        assert times >= 1           # must be >= 1
        
        index = start_index
        self.set_bg_new(index)
        
        # perform [times] updates on the bg, if set to greater than '1'
        if times > 1:
            for i in tqdm(range(times-1),initial=1,total=times, desc="Reset BG (start_index={})".format(start_index)):
                index += 1
                self._update_bg_once(index)
        pass
    
    def set_bg_new(self, index=0):
        """Will overwrite the background image data with the image at 
        'index' in the ImageLoaderClass object."""
        assert index >= 0     # index must be positive int
        
        img_new = self._ILO.get_img(index)      # load new image
        self.img_bg = np.float32( img_new )     # set new img as a float array (important for weighted addition!!!)
        pass
    
    def update_bg(self,start_index:int, times=1):
        """Will update the background image a number of [times] times, from 
        the position of 'start_index' in the ImageLoaderClass object."""
        assert start_index >= 0     # start_index must be positive int
        assert times >= 1           # must be >= 1
        
        index = start_index
        
        # if only ONE time, then without progressbar
        if times == 1:
            self._update_bg_once(index)
        else: #otherwise with progress bar
            for i in tqdm(range(times), desc="Update BG (start_index={})".format(start_index)):
                self._update_bg_once(index)
                index += 1
        pass
    
    def _update_bg_once(self,index:int):
        """Will update the background image ONCE, from 
        the position of 'index' in the ImageLoaderClass object."""
        if index < 0: return # do nothing if we pass a negative value
    
        # load new image
        img_new = self._ILO.get_img(index)
        # weighted accumulation
        assert img_new.shape == self.img_bg.shape    # ensure that we can add them (same dimensions)
        cv2.accumulateWeighted( img_new, self.img_bg, self.alpha)
        pass
    
    def get_bg_diff(self,img,inverse=True):
        """
        Calculates the difference between the background image and the provided 'img'.
        
        Due to the bees being dark (lower values) compared to the light background 
        (higher values), we must subtract the 'img' from 'bg' (inverse=True), 
        to get positive values for the position of NEW bees.
        
        (positions were bees have left will have negative values, but those 
        can be ignored later on, by casting back to uint8)

        Parameters
        ----------
        img : : numpy.ndarray
            Grayscale uint8 image.
        inverse : : Bool, optional
            Defines whether diff=img-bg (False), or diff=bg-img (True). The default is True.

        Returns
        -------
        img_diff : : numpy.ndarray
            Grayscale int16 image.
        """
        if inverse:
            img_diff = np.int16(self.img_bg) - np.int16(img)
        else:
            img_diff = np.int16(img) - np.int16(self.img_bg)
        return img_diff
    
#%%

class ParentImageClass:
    def __init__(self, ILO: IHM.ImageLoaderClass, 
                 BGHandler: BackgroundImageClass, 
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
        self._BG_ref = BGHandler
        self._index = index
        self.set_path_extracted(path_extr)
        
        self.set_parent_gauss_blurr_size(gauss_blurr_size)
        self.set_parent_otsu_min_threshold(otsu_min_threshold)
        self.set_parent_open_close_kernel(open_close_kernel_size)
        self.set_parent_pixel_area_min_max(pixel_area_min, pixel_area_max)
        
        self.set_focus_img_size(focus_img_size)
        self.set_focus_bg_gauss_kernel_size(focus_bg_gauss_kernel_size)
        self.set_focus_dilate_kernel_size(focus_dilate_kernel_size)
        
        # deepcopy of image
        self._img = self._ILO.get_img(self._index).copy()
        self._dim = (self._img.shape[1],self._img.shape[0])
        # deepcopy of image (original)
        self._orig_img = self._ILO.get_img_orig(self._index).copy()
        self._orig_dim = (self._orig_img.shape[1],self._orig_img.shape[0])
        
        # init some child objects/vars
        self.contour_list_valid = []
        self.contour_list_raw = []
        self.child_list = []
        
        # init the img dictionary
        self.img = dict()
        
        
        self.process_1(DEBUG=True)
        self.process_2(DEBUG=True)
        
        
        # # Fill the child list with BeeFocusImage objects
        # for i in range(len(self.contour_list)):
        #     self.child_list.append(BeeFocusImage(self,i,self.contour_list[i],
        #                                          self._focus_bg_gauss_kernel_size,
        #                                          self._focus_dilate_kernel_size))
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
    
    def process_2(self, DEBUG=False):
        """Check for valid area sizes of contours"""
        self.p31_contours_debug(DEBUG=DEBUG)
        
        pass
    
    
    def p00_fetch_src_gray(self, DEBUG=False):
        """fetch the first image (grayscale of original)"""
        self.img["00 gray"] = self._ILO.get_img(self._index)
        
        if DEBUG:
            imgs = [self.img["00 gray"]]
            labels=["00 gray"]
            mySIV = PHM.SimpleImageViewer(imgs,None,labels, "p00_fetch_src_gray")
        pass
    
    def p10_diff_from_bg(self, DEBUG=False):
        """get the int16 difference image, convert it to uint8"""
        diff_int16 = self._BG_ref.get_bg_diff(self.img["00 gray"])
        
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
    
    def p40_contours_check(self, DEBUG=False):
        """Check the extracted contours if they are within min/max ares sizes."""
        # We are only interested in the outermost contours (EXTERNAL), 
        #   because everything else does not make sense to handle (bees inside 
        #   a different detected object)
        _,contours, _ = cv2.findContours(self.img["25 reduced"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contour_list_raw = contours
        
        # calc all areas
        self.contour_areas = [int( cv2.contourArea(c) ) for c in contours]
        
        # check all areas for their size
        (min_a, max_a) = self.parent_area_min_max
        checklist = [(a>=min_a and a<=max_a) for a in self.contour_areas]

        # based on this True/False list, the final contour_list is filled
        for i in range(len(self.contour_areas)):
            if checklist[i]:
                self.contour_list.append(contours[i])
        pass
    
    def p31_contours_debug(self, DEBUG=False):
        """Creates a debug image after getting the contours"""
    
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
    
    ### -----------------------------------------------------------------------
    ### PARENT config FUNCTIONS
    ### -----------------------------------------------------------------------
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
        self.parent_orig = self.parent._orig_img
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
    """
    Handles the extraction of images from a list of images (as defined in \
    ImageHandlerObject).
    Will place extracted images in specified directory.

    Parameters
    ----------
    ILObject : : ImageLoaderClass
        Must be an ImageLoaderClass object. Contains a list of images to be \
        used for extraction.
    path_extracted : : String (path), optional
        Path to directory for extracted images. The default is "extracted/".
    gauss_blurr_kernel : : Integer, optional
        Filter size for gaus blurr after difference from BG. The default is 5.
    otsu_min_threshold : : Integer, optional
        Minimum accepted threshold value resulting from the OTSU algorithm. \
        Difference images with a value below this will be considered to be \
        [empty]. The default is 10.
    mean_weight_alpha : : Float (0...1), optional
        The weight with which a new image is added to the 'average' of the \
        backgorund image. The default is 0.1.
    open_close_kernel_size : : INT, optional
        The Kernel size for Opening/Closing the Threshold image for better
        seperation and detection of blobs. The default is 7.
    min_pixel_area : : Number, optional
        The minimum pixel area (mind reduced_img_dim) which is required to \
        plausibly contain a bee. The default is 1000.
    dilate_kernel_size : : Integer, optional
        Size of the Kernel when opning the Image (to get rid of small dots). \
        The default is 32.
    """
    
    def __init__(self, ILObject, path_extracted="./extracted/", \
                 gauss_blurr_kernel=5, otsu_min_threshold=10, mean_weight_alpha=0.1, \
                 open_close_kernel_size = 7, \
                 min_pixel_area=1000, dilate_kernel_size=32):
        self.img_index = 0
        self.img = dict()
        self.img["bg"]=None
        self.img["00 src"]=None
        
        self.set_ImageLoaderObject(ILObject)
        self.set_path_extracted(path_extracted)
        
        self.set_gauss_blurr_kernel(gauss_blurr_kernel)
        self.set_otsu_min_threshold(otsu_min_threshold)
        self.set_mean_weight_alpha(mean_weight_alpha)
        self.set_open_close_kernel(open_close_kernel_size)
        # self.set_gauss_reduce_threshold(gauss_reduce_threshold)
        self.set_min_pixel_area(min_pixel_area)
        self.set_dilate_kernel_size(dilate_kernel_size)
        
        self.init_df(list_cols = ["img_extr","img_overlay","index_overlay"])
        
        
        # print("-- Handler Object created")
        self.restart(10)
        pass
    
    # TODO: Update if necessaray
    def init_df(self,list_cols):
        assert type(list_cols) in [list, tuple]
        self.df = pd.DataFrame(columns=list_cols)
        pass
    
    
    # SETTINGS for image import and export ------------------------------------
    
    # DONE
    def set_ImageLoaderObject(self, new_ILO):
        """Sets the 'ImageLoaderClass' from which the images to read are loaded."""
        # Make sure, you have a correct IHC Object type
        assert type(new_ILO) == IHM.ImageLoaderClass
        
        self.ILO = new_ILO
        pass
    
    # DONE: could be better
    def set_path_extracted(self,path_extracted):
        """Sets the path to save all extracted information to."""
        assert (type(path_extracted) == str)    # ensure that path is a string
        
        # Stop object creation, if no valid file path is given
        if os.path.isdir(path_extracted) == False:
            raise Exception("Requires a legal directory path!")
        
        self.prop_path_extracted = os.path.abspath(path_extracted)
        pass
    
    
    # PERFORMING of Background update and Loading of image --------------------
    
    # TODO: Update if necessaray
    def add_to_background_img(self, img_new, overwrite=False):
        """
        Either overwrites the "background" image with "img_new" (overwrite=True). \
        Or adds the "img_new" (weighted addition) to the "background" image.
        
        self.img["bg"] must be converted to np.uint8 before using imshow.

        Parameters
        ----------
        img_new : : cv2-grayscale image
            Image to be added to "background".
        overwrite : : BOOL, optional
            Determines wether the current "background" image is overwritten \
            instead of a weighted addition. The default is False.
        """
        if (overwrite):
            # overwrite "background" image with "img_new" 
            self.img["bg"] = np.float32( img_new )
        else:
            # weighted accumulation
            assert img_new.shape == self.img["bg"].shape    # ensure that we can add them
            cv2.accumulateWeighted( img_new, self.img["bg"], self.prop_mean_weight_alpha)
        
        pass
    
    # DONE
    def load_img(self, index):
        """Loads the image at 'index' position. \
        (Path information comes from List of ImageLoaderClass)
        """
        # Loads/Copies image(index) to local image variable
        self.img["00 src"] = self.ILO.get_img(index).copy()
        pass
    
    
    # SETTINGs for Image extraction -------------------------------------------
    
    # TODO: Update if necessaray
    def set_gauss_blurr_kernel(self, gauss_blurr_kernel):
        # gauss kernel size must be an odd integer
        temp = int(gauss_blurr_kernel)
        if (temp % 2) != 1: temp += 1
        self.prop_gauss_blurr_kernel = (temp, temp)
        pass
    
    # TODO: Update if necessaray
    def set_mean_weight_alpha(self, mean_weight_alpha):
        self.prop_mean_weight_alpha = float(mean_weight_alpha)
        pass
    
    # Done
    def set_open_close_kernel(self, open_close_kernel_size):
        """Creates a circular (elliptic) kernel based on the specified kernel 
        size. Will convert to an odd number, if necessary."""
        #  kernel size must be an odd integer
        temp = int(open_close_kernel_size)
        
        # make an odd number
        if (temp % 2) != 1: temp += 1
        
        # create circular (elliptic) kernel
        self.open_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(temp,temp))
        pass
    
    # DONE
    def set_otsu_min_threshold(self, otsu_min_threshold):
        """Sets the minimum accepted threshold value generated from the OTSU \
        algorithm."""
        self.otsu_min_threshold = int(otsu_min_threshold)
        pass
    
    # # TODO: Update if necessaray
    # def set_gauss_reduce_threshold(self, gauss_reduce_threshold):
    #     self.prop_gauss_reduce_threshold= int(gauss_reduce_threshold)
    #     pass
    
    # TODO: Update if necessaray
    def set_min_pixel_area(self, min_pixel_area):
        self.prop_min_pixel_area = int(min_pixel_area)
        pass
    
    # TODO: Update if necessaray
    def set_dilate_kernel_size(self, dilate_kernel_size):
        ks = 5
        self.prop_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks))
        self.prop_dilate_iterations = int( dilate_kernel_size/(ks-1) )
        pass
    
    
    # PERFORMING of Image extraction ------------------------------------------
    
    # # TODO: Update if necessaray
    # def median(self, source):
    #     """img_set_median"""
    #     self.img_set_median = cv2.medianBlur(source, self.prop_median_filter_size)
    #     pass
    
    # Done
    def difference_from_BG(self, source):
        """img_set_diff"""
        # self.img_set_diff = np.int16(self.img_set_mean) - np.int16(self.img_set_resize)
        self.img["10 diff"] = np.int16(self.img["bg"]) - np.int16(source)
        
        """Due to the bees being dark (lower values) compared to the light 
        background (higher values), we must subtract the current image from 
        the BG, to get positive values for the position of new bees.
        (and negative values for the positions were bees have left).
        (the negative values will be ignored later on)"""
        pass
    
    # Done
    def threshold_diff(self, source, DEBUG=False):
        """Performs gaussian blurr on difference image (source).
        
        Thresholds the blurred difference image with OTSU algorithm. 
        (If the OTSU threshold is below the min_th_value, then the output image 
        MUST be viewed as 'empty'.)
        
        I DEBUG is True, then the results will be shown"""
        
        # 10 : cut off negative values
        # This ignores artefacts from bees leaving the image (which would be negative)
        _,img = cv2.threshold(source,0,255,cv2.THRESH_TOZERO)
        
        # 20 : perform gaussian blurr (kernel size defined in constructor)
        k = self.prop_gauss_blurr_kernel
        diff_blurred = cv2.GaussianBlur(np.uint8(img),k,0)
        
        # myHist = cv2.calcHist([temp],[0],None,[256],[0,256])
        
        # 50 : perform threshold with OTSU, but be careful about its threshold value!
        otsu_threshold, img_otsu = cv2.threshold(diff_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        self.img["20 threshold"] = img_otsu
        
        
        if DEBUG:
            imgs = [self.img["00 src"], self.img["bg"], diff_blurred, img_otsu]
            labels=["image","BG","diff_blurred","otsu {}".format(otsu_threshold)]
            mySIV = PHM.SimpleImageViewer(imgs, None,labels, "threshold_diff")
        
        return otsu_threshold
    
    # -------------------------------------------------------------------------
    # DONE: TESTING `FUNCTION FOR THRESHOLDING
    def threshold_diff_TEST_inc(self,iterations=1):
        raise Exception("Do not call this function anymore, unless debugging!")
        
        if iterations>1:
            for i in tqdm(range(iterations), desc="BG iterations before threshold"):
                # 10 : increment the index and check if index is still inside the list
                self.img_index += 1
                if (self.img_index >= self.ILO._size):
                    print("No more Iterations possible. Index has reached end of img_name_list.")
                    return
                # 20 : update weighted mean with last image (BEFORE loading a new image)
                self.add_to_background_img(img_new=self.img["00 src"])
                # 30 : load the current image
                self.load_img(self.img_index)
        else:
            # 10 : increment the index and check if index is still inside the list
            self.img_index += 1
            if (self.img_index >= self.ILO._size):
                print("No more Iterations possible. Index has reached end of img_name_list.")
                return
            # 20 : update weighted mean with last image (BEFORE loading a new image)
            self.add_to_background_img(img_new=self.img["00 src"])
            # 30 : load the current image
            self.load_img(self.img_index)
            
        # 40 : calc difference from blurr to mean
        self.difference_from_BG(source=self.img["00 src"]) #self.img["10 diff"]
        #-------------------
        # 10 : cut off negative values
        _,diff =  cv2.threshold(self.img["10 diff"],0,255,cv2.THRESH_TOZERO) 
        diff_uint8 = np.uint8(diff)
        # 20 : perform gaussian blurr (kernel size defined in constructor)
        k = self.prop_gauss_blurr_kernel
        diff_blurred = cv2.GaussianBlur(diff_uint8,k,0)
        
        myHist = cv2.calcHist([diff_blurred],[0],None,[256],[0,256])
        
        plt.hist(myHist,256,[0,255])
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(900,100,640, 545)
        plt.draw()
        
        # 30 : determine mean value and standard-deviation of blurred image
        mean,std = cv2.meanStdDev(diff_blurred)
        print(mean,std)
        
        #otsu testing
        otsu_threshold, img_otsu = cv2.threshold(diff_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _,img_th_10 = cv2.threshold(diff_blurred, 10, 255, cv2.THRESH_BINARY)
        
        #hybrid threshold
        th3 = max([10,int(0.9*otsu_threshold)])
        _,img_th_hybrid = cv2.threshold(diff_blurred, th3, 255, cv2.THRESH_BINARY)
        
        imgs = [self.img["00 src"], self.img["bg"], diff_blurred, img_otsu,img_th_10,img_th_hybrid]
        labels=["image","BG","diff_blurred","otsu {}".format(otsu_threshold),"threshold 10","hybrid th {}".format(th3)]
        mySIV = PHM.SimpleImageViewer(imgs, None,labels, "threshold_diff_TEST_inc")
        
        
        return myHist,otsu_threshold,diff_blurred
    # -------------------------------------------------------------------------
    
    
    # Done
    def reduce_open_close(self, source, DEBUG=False):
        """Performs Closing (remove black holes) and then Opening 
        (remove lone white pixels) on the Threshold image (source).
        
        Will plot results, if DEBUG=True."""
        img_closed = cv2.morphologyEx( source, cv2.MORPH_CLOSE, self.open_close_kernel )
        
        img_opened = cv2.morphologyEx( img_closed, cv2.MORPH_OPEN, self.open_close_kernel )
        
        self.img["30 reduced"] = img_opened
        
        if DEBUG:
            imgs = [self.img["00 src"],self.img["20 threshold"], img_closed, img_opened]
            labels = ["src", "threshold", "img_closed", "img_opened"]
            mySIV = PHM.SimpleImageViewer(imgs, None, labels)
        pass
    
    # Done
    def get_contours_reduced(self, img):
        """Detects ONLY the contours found in the reduced threshold image"""
        _,contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # check all the contours
        self.contours=[]

        for c in self.cont_reduced:
            # Check for area size - throw out if too small
            area = int( cv2.contourArea(c) )    # get area of contour
            if area < self.prop_min_pixel_area: 
                continue    # skip, if area is too small
            
            
            # create a mask based on current contour
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask, c, -1, 255, -1)
            
            # get avg color in source img, that is in the contour area - throw out, if too dark (black hole)
            mean = cv2.mean(img, mask=mask)
            if mean[0] <= 127: 
                continue    # skip, if the avg color is too low (sign of an encaspulated black area)
            
            # if all checks are ok, then save as "good" contour
            self.contours.append(c)
            continue
        pass
    
    # TODO: Update if necessaray
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
    
    
    # EXECUTION of Image Extraction -------------------------------------------
    
    # Done
    def restart(self, prepare_time=5):
        """
        Resets index. Sets first image as the Background image. \
        Loads the next x images to generate a more 'realistic' background image. \
        Resets Index.

        Parameters
        ----------
        prepare_time : : INTEGER, optional
            How many images to load for generating the initial background image. \
            The default is 5.
        """
        print()
        print("-- Restarting Handler")
        
        # 1 : load the first image
        self.img_index = 0
        self.load_img(self.img_index)
        
        # 2 : perform weighted mean (set current image as background)
        self.add_to_background_img(self.img["00 src"], overwrite=True)
        
        # 3 : use the first 'prepare_time' number of images to get a usaable background
        assert prepare_time >= 0 # We need a positive number of times to repeat this
        
        # usage of loading bar indicator (#tqdm)
        for i in tqdm(range(prepare_time), desc="Restart: Preparing Background image"):
            # Repeated loading of images to generate a better BG image for the beginning
            self.load_img(self.img_index)
            self.add_to_background_img(self.img["00 src"])
            self.img_index += 1 
        
        self.img_index = 0      # reset index
        pass
    
    
    # TODO: Update if necessaray
    def iterate(self, times = 1):
        """
        This function will perform ALL extraction steps for the set number \
        of images. (incremental increase from the current index) """
        assert times>0  # MUST be a number >= 1 !!!
        
        # try:
        for i in range(times):
            # 10 : increment the index and check if index is still inside the list
            self.img_index += 1
            if (self.img_index >= self.ILO._size):
                print("No more Iterations possible. \
                      Index has reached end of img_name_list.")
                return
            
            # 20 : update weighted mean with last image (BEFORE loading a new image)
            self.add_to_background_img(img_new=self.img["00 src"])
            
            # 30 : load the current image
            self.load_img(self.img_index)
            
            # 40 : calc difference from blurr to mean
            self.difference_from_BG(source=self.img["00 src"]) # => self.img["10 diff"]
            
            
            # 50 : threshold of the difference
            otsu_th = self.threshold_diff(source=self.img["10 diff"]) # => self.img["20 threshold"]
            
            # 60 : Check, if otsu threshold may be empty
            if otsu_th < self.otsu_min_threshold:
                # THIS IMAGE IS EMPTY!!! STOP HERE!
                print("empty image")
                
            else:
            
                # gaussian blurr
                self.reduce_open_close(source=self.img["20 threshold"], DEBUG=True) # => self.img["30 reduced"]
                
                raise Exception("DEBUG stop")
                
                
                # selection and seperation
                self.get_contours_reduced(source=self.img["30 reduced"])
                
                
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
    
    
    # ADDITION of Plotting ----------------------------------------------------
    
    # TODO: Update if necessaray
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
        
        ax_textbox = plt.axes([0.05, 0.94, 0.1, 0.05])
        ax_iter_x =  plt.axes([0.25, 0.94, 0.2, 0.05]) #posx, posy, width, height
        ax_iter_10 = plt.axes([0.50, 0.94, 0.2, 0.05])
        ax_iter_20 = plt.axes([0.75, 0.94, 0.2, 0.05])
        
        self.text_box_iter = mpl.widgets.TextBox(ax_textbox, "Iter","1")
        self.text_box_iter.on_text_change(self.txt_change)
        
        self.but_iter_x_value = 1 ##########
        self.but_iter_x =  mpl.widgets.Button(ax_iter_x,  "Iter({})".format(str(self.but_iter_x_value)) )
        
        self.but_iter_10 = mpl.widgets.Button(ax_iter_10, "Iter(10)")
        self.but_iter_20 = mpl.widgets.Button(ax_iter_20, "Iter(20)")
        self.but_iter_x.on_clicked(self.on_click_iter_x)
        self.but_iter_10.on_clicked(self.on_click_iter_10)
        self.but_iter_20.on_clicked(self.on_click_iter_20)
        self.but_iter_x.color = 'cyan'
        self.but_iter_10.color = 'cyan'
        self.but_iter_20.color = 'cyan'
        pass
    
    # TODO: Update if necessaray
    def txt_change(self,event):
        print("txt_change:",str(event))
        print("max:",str(self.img_name_list_length-self.img_index-1))
        self.txt_change_event = event
        
        if event.isnumeric():
            temp = int(event)
            # print("temp:",temp)
            
            if temp < 1:
                temp=1
            if temp > (self.img_name_list_length-self.img_index-1):
                temp = self.img_name_list_length-self.img_index-1
            # print("temp:",temp)
            self.but_iter_x_value = temp
        self.text_box_iter.set_val(str(self.but_iter_x_value))
        
        self.but_iter_x.label.set_text( "Iter({})".format(str(self.but_iter_x_value)) )
        self.fig.canvas.draw()
        # print("done txt box")
        pass
    
    # TODO: Update if necessaray
    def on_click_iter_x(self,event):
        print("on_click_iter_x:",str(self.but_iter_x_value))
        self.iter_and_plot_update(self.but_iter_x_value)
        pass
    # TODO: Update if necessaray
    def on_click_iter_10(self,event):
        print("on_click_iter_10")
        self.iter_and_plot_update(10)
        pass
    # TODO: Update if necessaray
    def on_click_iter_20(self,event):
        print("on_click_iter_20")
        self.iter_and_plot_update(20)
        pass
    
    
    # TODO: Update if necessaray
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
    print("## Calling main function.)\n")
    
    print("cv2.version = {}".format(cv2.__version__))
    print("numpy.version = {}".format(np.__version__))
    print("matplotlib.version = {}".format(mpl.__version__))
    print("pandas.version = {}".format(pd.__version__))
    
    
    # Window Cleanup
    cv2.destroyAllWindows()
    plt.close('all')
    
    TEST = 1
    # %%
    if TEST == 1:
        cv2.destroyAllWindows()
        plt.close('all')
        
        myPath = "D:\\ECM_PROJECT\\bee_images_small"
        path_extr = "./extracted/"
        
        myIFC = IHM.ImageFinderClass(myPath,maxFiles=0,acceptedExtensionList=("png",))
        myILC = IHM.ImageLoaderClass(myIFC, dim=(400,300),mask_rel=(0.1,0,1,1))
        
        myBGH = BackgroundImageClass(myILC,0,alpha_weight=0.1)
        myBGH.reset(0,20)
        index=5020
        myBGH.update_bg(index-20,20)
        
        myPar = ParentImageClass(myILC,myBGH,index=index+1,path_extr=path_extr,
                                 open_close_kernel_size=11,DEBUG=True)
        #%%
        plt.close('all')
        index+=1
        
        myBGH.update_bg(index)
        
        myPar = ParentImageClass(myILC,myBGH,index=index+1,path_extr=path_extr,
                                 open_close_kernel_size=8,
                                 focus_dilate_kernel_size=20,
                                 pixel_area_min=1000,
                                 pixel_area_max=5000,
                                 DEBUG=True)
        pass
    
    
    
    