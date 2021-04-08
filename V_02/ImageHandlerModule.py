# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:10:41 2021

@author: Admin

PURPOSE: Make an Object, that handles getting all the images

VERSION: 001
"""

import os, glob
import numpy as np
import cv2


#%%
class ImageFinderClass:
    """
    Creates an object which contains a list of all files of specified extension in the specified directory.

    Parameters
    ----------
    filePathString : STRING
        Location of the directory from which the images will be taken.
    acceptedExtensionList : LIST or TUPLE (or STRING of a single extension), optional
        List of STRINGS which define which file extensions are looked for. The default is ("jpg",).
    maxFiles : INTEGER, optional
        A value of greater than Zero will limit the number of files which are added to the list, otherwise all valid files are added. The default is 0.
    DebugPrint : BOOLEAN, optional
        If set to TRUE, will print additional information when the Constructor is Called. The default is False.

    """
    
    def __init__(self, filePathString, acceptedExtensionList=("jpg",),maxFiles=0, DebugPrint=False):
        # Ensure correct parameter type
        assert type(maxFiles) == int;
        
        self.dir_path = None    # Reduntant initialization
        self.ext_list = None
        self.file_list = None
        self.size = None
        
        self.set_dir_path(filePathString, DebugPrint)
        self.set_ext_list(acceptedExtensionList, DebugPrint)
        self.find_extension_files(maxFiles, DebugPrint)
        
        if DebugPrint: print("### Init complete.")
        pass


    def set_dir_path(self, fpath, DebugPrint=False):
        if DebugPrint: print("## Function Call: set_dir_path")
        
        # Checks if path is a directory
        isDirectory = os.path.isdir(fpath)
        # MEMO: dont forget about [os.path.join(path1, path2)] !!!
        
        # Stop object creation, if no valid file path is given
        if isDirectory == False:
            raise Exception("Requires a legal directory path!"); pass
        # No else-branch should be required at this point, 
        # since the exception should catch this.
        
        self.dir_path = fpath;
        if DebugPrint: print("# Set path to: "+str(self.dir_path))
        pass
    
    
    def set_ext_list(self, ext_list, DebugPrint=False):
        if DebugPrint: print("## Function Call: set_ext_list")
        
        # Quick fix for just passing a string as accepted extension
        if type(ext_list)==str:
            ext_list = (ext_list,)
            if DebugPrint: print("# Detected string as ext_list")
        
        # Check if we are dealing with a list here
        if type(ext_list) not in [list, tuple]:
            raise Exception("Invalid Extension List parameter: Not a List or Tuple!")
        
        # Check if our ext_list is a list of strings (which is desired)
        if not all(isinstance(s, str) for s in ext_list):
            raise Exception("Invalid Extension List parameter: List must contain only Strings!")
        
        self.ext_list = ext_list;
        if DebugPrint: print("# Set ext_list to: "+str(self.ext_list))
        pass
    
    
    def find_extension_files(self, maxFiles, DebugPrint=False):
        if DebugPrint: print("## Function Call: find_extension_files")
        
        self.file_list = []
        
        # Iterate through all extensions
        for extn in self.ext_list:
            
            # Make a pathmask for every extension
            pathmask = os.path.join(self.dir_path, ("*." + extn) )
            if DebugPrint: print("# Current mask extension: "+str("*." + extn))
            
            for maskedFile in glob.glob(pathmask):
                self.file_list.append( os.path.basename(maskedFile) )
                # Stop, if we have the desired number of files
                if (maxFiles>0 and len(self.file_list)>=maxFiles): break
            
            # Stop, if we have the desired number of files
            if (maxFiles>0 and len(self.file_list)>=maxFiles): break
        
        # Check, if we have more than 0 files
        if len(self.file_list) == 0:
            raise Exception("No files for specified extensions found!")
        
        self.size = len(self.file_list)
        
        if DebugPrint: print("# Number of files in list: "+str(self.size))
        pass

#%%

class ImageLoaderClass:
    """
    Creates an 'ImageLoaderClass' object.
    Can be used to automatically load, grayscale and resize images as defines by an 'ImageFinderClass' object.

    Parameters
    ----------
    base_IFC : ImageFinderClass Object
        The 'ImageFinderClass' Object defines which images can be loaded.
    grayscale_en : BOOL, optional
        Defines if the loaded image will be converted to grayscale. The default is True.
    new_dim : TUPLE(INT,INT), optional
        This tuple defines to which dimensions the image shall be resized. The default is None.
    mask_rel : TUPLE(FLOAT,FLOAT,FLOAT,FLOAT), optional
        Defines a rectangular mask in relative coordinates (x1, x2, y1, y2). All pixels outside the defined rectangle will be set to 0. The default is (0.0, 0.0, 1.0, 1.0).

    Returns
    -------
    None.
    """
        
    def __init__(self, base_IFC, new_dim=(400,300), mask_rel=(0.0,0.0,1.0,1.0), grayscale_en=True):
        self._img = None
        
        self._set_base_IFC(base_IFC)
        self._set_grayscale(grayscale_en)
        self._set_dim(new_dim)
        self._set_mask_rel(mask_rel)
        pass
    
    # Constructor Functions ---------------------------------------------------
    
    def _set_base_IFC(self, new_IFC):
        # Ensure correct parameter type
        assert type(new_IFC) == ImageFinderClass;
        self._IFC = new_IFC
        self._IFC_path = self._IFC.dir_path
        self._IFC_list = self._IFC.file_list
        self._size = self._IFC.size
        pass
    
    def _set_grayscale(self, grayscale_en):
        # Ensure correct parameter type
        if not type(grayscale_en) == bool:
            raise Exception("grayscale_en must be a bool!")
        self._grayscale = grayscale_en
        pass
    
    def _set_dim(self, dim):
        # Check if we are dealing with a list here
        if type(dim) not in [list, tuple]:
            raise Exception("Not a List or Tuple!")
        
        # Check if list is only ints
        if not all(isinstance(x, int) for x in dim):
            raise Exception("List must contain only two positive INTEGERS!")
        
        # Check if length of 2 elements
        if not len(dim)==2:
            raise Exception("List must contain only TWO positive integers!")
        
        # Check if positive values
        if not all(x>0 for x in dim):
            raise Exception("List must contain only two POSITIVE (> 0) integers!")
        
        self._scale_dim = tuple(dim)
        pass
    
    def _set_mask_rel(self,m):
        import numbers
        
        # Check if we are dealing with a list here
        if type(m) not in [list, tuple]:
            raise Exception("Not a List or Tuple!")
        
        # Check if our dim is a list of two ints (which is desired)
        if not all(isinstance(x, numbers.Number) for x in m):
            raise Exception("List must contain four positive NUMBERS [0,1]!")
        
        # Check if our dim is a list of two ints (which is desired)
        if not len(m)==4:
            raise Exception("List must contain FOUR positive numbers [0,1]!")
        
        # Check if our dim is a list of two ints (which is desired)
        # REMEMBER: mask = (x1,y1, x2,y2)
        if not all([0 <= m[0], m[0] < m[2], m[2] <= 1, 0 <= m[1], m[1] < m[3], m[3] <= 1]):
            raise Exception("List must contain four positive numbers in the RANGE [0,1]!")
        
        self._mask_rel = [np.float32(x) for x in m]
        
        self._gen_mask_abs()
        pass
    
    def _gen_mask_abs(self):
        mr = self._mask_rel         # get mask (relative)
        sx,sy = self._scale_dim     # get reduced dimensions
        
        # mask for image of reduced size (x1,y1, x2,y2)
        ma = np.around([mr[0]*sx, mr[1]*sy, mr[2]*sx, mr[3]*sy])
        pt1 = ( int(ma[0]), int(ma[1]) )
        pt2 = ( int(ma[2]), int(ma[3]) )
        
        # create emtpy (black) grayscale image
        img_mask = np.zeros((sy,sx),dtype=np.uint8) 
        # Draw mask as white (255) rectangle
        img_mask = cv2.rectangle(img_mask, pt1, pt2, 255, cv2.FILLED)
        
        if not self._grayscale:
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        
        # cv2.imshow("img_mask", img_mask)
        
        self._mask_img = img_mask
        pass
    
    
    # Work Functions ----------------------------------------------------------
    
    
    def _load_img(self, index):
        # Check if index is possible
        if index not in range(self._size):
            raise Exception("Index ({}) out of bounds (length={})".format(index,self._size))
        
        
        # get path to currently indexed image (use os.path.join!)
        f_path = os.path.join(self._IFC_path, self._IFC_list[index])
        
        
        # Load original image
        img_0 = cv2.imread(f_path, cv2.IMREAD_COLOR)
        
        
        if (self._grayscale):
            # Get grayscale version
            img_1 = cv2.cvtColor(img_0, cv2.COLOR_BGR2GRAY)
        else:
            img_1 = img_0
        
        
        # Scale to new dimensions
        img_2 = cv2.resize(img_1, self._scale_dim, interpolation = cv2.INTER_AREA )
        
        
        # Apply Mask
        img_3 = cv2.bitwise_and( img_2, self._mask_img )
        
        self._img = img_3
        pass
    
    
    def get_img(self, index):
        self._load_img(index)
        return self._img
        pass







#%%

if __name__ == "__main__":
    # main function test code area.
    TEST = 2
    
    
    if(TEST==1):
        myPath = "C:\\Users\\Admin\\0_FH_Joanneum\\ECM_S3\\PROJECT\\bee_images\\01_8_2020\\5"
        
        myIFC = ImageFinderClass(myPath,maxFiles=20,DebugPrint=True)
        # myIHC = ImageFinderClass(myPath,"png",maxFiles=20,DebugPrint=True)
        
        print(myIFC.file_list)
    
    
    if(TEST==2):
        myPath = "C:\\Users\\Admin\\0_FH_Joanneum\\ECM_S3\\PROJECT\\bee_images\\01_8_2020\\5"
        myIFC = ImageFinderClass(myPath,maxFiles=100)
        
        myILC = ImageLoaderClass(myIFC, new_dim=(400,300),mask_rel=(0.0,0.0,.85,.95),grayscale_en=False)
        
        im = myILC.get_img(1)
        cv2.imshow("im",im)
        
        
        