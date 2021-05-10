# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:10:41 2021

@author: Admin

PURPOSE: Make an Object, that handles getting all the images

VERSION: 002
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
    filePathString : : STRING
        Location of the directory from which the images will be taken.
    acceptedExtensionList : : LIST or TUPLE (or STRING of a single extension), optional
        List of STRINGS which define which file extensions are looked for. The default is ("jpg",).
    maxFiles : : INTEGER, optional
        A value of greater than Zero will limit the number of files which are added to the list, otherwise all valid files are added. The default is 0.
    DebugPrint : : BOOLEAN, optional
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
        """
        Checks if file path is valid. \
        Store filepath if valid, throw exception otherwise.
        """
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
        """Checks if the provided 'ext_list' is of the correct type/formating.
                
        Accepted formating
        ----------
        String : : Must be a single string of a file extension.
        
        Tuple or List : : Must be a tuple or list which ONLY contains strings \
        of file extensions.
        """
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
        """
        This function generates a list of all files with matching file extension \
        (or up to the max number), that can be found inside the defined directory.
        """
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
    base_IFC : : ImageFinderClass Object
        The 'ImageFinderClass' Object defines which images can be loaded.
    grayscale_en : : BOOL, optional
        Defines if the loaded image will be converted to grayscale. The default is True.
    new_dim : : TUPLE(INT,INT), optional
        This tuple defines to which dimensions the image shall be resized (w,h). \
        The default is (400,300).
    mask_rel : : TUPLE(FLOAT,FLOAT,FLOAT,FLOAT), optional
        Defines a rectangular mask in relative coordinates (x1,y1, x2, y2). \
        All pixels outside the defined rectangle will be set to 0. \
        (0,0) is topleft and (1,1) is bottomright in CV2 formating. \
        The default is (0.0, 0.0, 1.0, 1.0).

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
        """
        Defines whether the loaded image will be converted to grayscale or \
        left unchanged.
        """
        # Ensure correct parameter type
        if not type(grayscale_en) == bool:
            raise Exception("grayscale_en must be a bool!")
        self._grayscale = grayscale_en
        pass
    
    def _set_dim(self, dim):
        """
        Sets the image dimension to which all images shall be resized \
        before they are processed

        Parameters
        ----------
        dim : : LIST or TUPLE of two integers
            Defines the dimensions (w,h) to which all loaded images shall be resized to.
        """
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
        """
        This function defines a positive mask in relative coordinates to \
        ignore all image information outside the defined rectangle.
        
        The function checks, if the input tuple of 4 floats is in the correct \
        format for marking the 2 defining vertices of a rectangle \
        (x1,y1, x2,y2) in a relative range from [0,1]. \
        (with the limitation 0<x1<x2<1 and 0<y1<y2<1) 
        
        (0,0) is topleft and (1,1) is bottomright in CV2 formating.
        
        This will later be upscaled to the real image dimensions. \
        (The realtive dimensions are used, so that the code \
        does not break, if the size of the images change.)
            
        If all checks are OK, the function for generating a masking image \
        is called, which generates an image of the same absolute size as the target \
        image in which all pixel in the rectangle are set to '1'. (positive mask)

        Parameters
        ----------
        m : : TUPLE or LIST of 4 Float values
            Defines a rectangle with two vertices (x1,y1, x2,y2) in a relative \
            coordinate system of range from [0,1]. 
        """
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
            raise Exception("List must contain four positive numbers in the RANGE [0,1]! \
                            (x1,y1, x2,y2) with the limitation 0<x1<x2<1 and 0<y1<y2<1.")
        
        self._mask_rel = [np.float32(x) for x in m]
        
        self._gen_mask_abs()
        pass
    
    def _gen_mask_abs(self):
        """
        This function takes the (previously validated) relative mask \
        coordinates and the absolute size of the target image and generates \
        an image of the same size as the target image, in which all pixels \
        inside the defined rectangle are set to '1' (positive mask).
        
        This mask is later used to bitwise-AND the loaded image. \
        All pixels outside the rectangle will be set to '0' (black).
        """
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
        
        # If we dont work in grayscale, then we convert to BGR
        if not self._grayscale:
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)
        
        # cv2.imshow("img_mask", img_mask)
        
        self._mask_img = img_mask
        pass
    
    
    # Work Functions ----------------------------------------------------------
    
    
    def _load_img(self, index):
        """
        Based on the list of images from the ImageFinderClass, \
        this function loads the image of specified index in that list.
        
        The image may be converted to grayscale. 
        
        The image is resized to the defined target dimensions.
        
        The mask is applied via a bitwise-AND to ignore regions that are \
        unimportant or that might cause interference.
        """
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
        """Returns the loaded (and prepared) image of the defined index"""
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
        
        myILC = ImageLoaderClass(myIFC, new_dim=(300,300),mask_rel=(0.1,0.2,.7,.99),grayscale_en=False)
        
        im = myILC.get_img(1)
        cv2.imshow("im",im)
        
        
        