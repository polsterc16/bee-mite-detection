# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:10:41 2021

@author: Admin

PURPOSE: Make an Object, that handles getting all the images

VERSION: 001
"""

import os, glob
import numpy as np


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
        self.file_list = []
        
        self.set_dir_path(filePathString, DebugPrint)
        self.set_ext_list(acceptedExtensionList, DebugPrint)
        self.find_extension_files(self.file_list, maxFiles, DebugPrint)
        
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
    
    
    def find_extension_files(self, dest_list, maxFiles, DebugPrint=False):
        if DebugPrint: print("## Function Call: find_extension_files")
        
        # Iterate through all extensions
        for extn in self.ext_list:
            
            # Make a pathmask for every extension
            pathmask = os.path.join(self.dir_path, ("*." + extn) )
            if DebugPrint: print("# Current mask extension: "+str("*." + extn))
            
            for maskedFile in glob.glob(pathmask):
                dest_list.append( os.path.basename(maskedFile) )
                # Stop, if we have the desired number of files
                if (maxFiles>0 and len(dest_list)>=maxFiles): break
            
            # Stop, if we have the desired number of files
            if (maxFiles>0 and len(dest_list)>=maxFiles): break
        
        # Check, if we have more than 0 files
        if len(dest_list) == 0:
            raise Exception("No files for specified extensions found!")
        
        if DebugPrint: print("# Number of files in list: "+str(len(dest_list)))
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
    grayscale : BOOL, optional
        Defines if the loaded image will be converted to grayscale. The default is True.
    new_dim : TUPLE(INT,INT), optional
        This tuple defines to which dimensions the image shall be resized. The default is None.
    mask_rel : TUPLE(FLOAT,FLOAT,FLOAT,FLOAT), optional
        Defines a rectangular mask in relative coordinates (x1, x2, y1, y2). All pixels outside the defined rectangle will be set to 0. The default is (0.0, 0.0, 1.0, 1.0).

    Returns
    -------
    None.
    """
        
    def __init__(self, base_IFC, grayscale=True, new_dim=None, mask_rel=(0.0,0.0,1.0,1.0)):
        self._set_base_IFC(base_IFC)
        self._set_dim(new_dim)
        self._set_mask_rel(mask_rel)
        
        pass
    
    # Constructor Functions ---------------------------------------------------
    
    def _set_base_IFC(self, new_IFC):
        # Ensure correct parameter type
        assert type(new_IFC) == ImageFinderClass;
        self.IFC = new_IFC
        pass
    
    def _set_dim(self, dim):
        if dim == None:
            # If none given, then we save 'None' and no scaling will be applied later
            self._scale_dim = dim
        else:
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
        pass
    
    
    # Work Functions ----------------------------------------------------------
    
    def _load_img(self, index):
        
        pass
    
    
    def get_img(self, index=0):
        
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
        
        myILC = ImageLoaderClass(myIFC, new_dim=(400,300),mask_rel=(0.3,0.1,0.95,0.8))
        
        
        