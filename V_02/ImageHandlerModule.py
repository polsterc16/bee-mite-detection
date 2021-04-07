# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:10:41 2021

@author: Admin

PURPOSE: Make an Object, that handles getting all the images

VERSION: 001
"""

import os, glob


#%%
class ImageHandlerClass:
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
def _main_():
    # main function test code area.
    myPath = "C:\\Users\\Admin\\0_FH_Joanneum\\ECM_S3\\PROJECT\\bee_images\\01_8_2020\\5"
    
    myIHC = ImageHandlerClass(myPath,maxFiles=-20,DebugPrint=True)
    # myIHC = ImageHandlerClass(myPath,"png",maxFiles=20,DebugPrint=True)
    
    print(myIHC.file_list)
    pass


if __name__ == "__main__":
    _main_()
