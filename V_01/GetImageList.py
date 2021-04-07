# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:14:13 2020

@author: Admin
"""


def get_image_list(list_dst,path,extension):
    """Sets list_dst to be a list of files with given extension in path of interest."""
    list_dst.clear()
    get_image_list_append(list_dst,path,extension)
    
def get_image_list_append(list_dst,path,extension):
    """Appends to list_dst a list of files with given extension in path of interest."""
    import os, glob
    
    while(path.endswith("/") or path.endswith("\\")):
        path = path[:-1]
    path = path + "\\"
    
    temp = extension.split(".") # split if "*.extension" format is used
    extn = temp[-1] # get only the extension
    
    pathmask = path + "*." +extn
    
    # get list of file names in given path
    for x in glob.glob(pathmask):
        list_dst.append( os.path.basename(x) )
    # list_jpg = [os.path.basename(x) for x in glob.glob(pathmask)]
    


def get_image_dict(path,extension,seperator):
    """Returns a dict of files with given extension in path of interest. \
    With prefixes as key, depending on defined seperator."""
    import os, glob
    
    while(path.endswith("/") or path.endswith("\\")):
        path = path[:-1]
    path = path + "\\"
    
    temp = extension.split(".") # split if "*.extension" format is used
    extn = temp[-1] # get only the extension
    
    pathmask = path + "*." +extn
    
    # get list of file names in given path
    list_jpg = [os.path.basename(x) for x in glob.glob(pathmask)]
    
    
    # create a dict sorted by prefixes seperated by "seperator"
    return_dict = {}
    for item in list_jpg:
        temp = item.split(seperator)
        key = temp[0]
        
        if key not in return_dict:
            return_dict[key] = []
        
        return_dict[key].append(item)
    
    return return_dict

# %%
    
def _main_():
    import os, glob
    
    path_img = "D:/ECM_PROJECT/images/"
    
    list_jpg = glob.glob(path_img + "*.jpg")
    # print(list_jpg)
    
    prefix = "D:/ECM_PROJECT/images\\"
    list_jpg2 = []
    for item in list_jpg:
        list_jpg2.append(item[len(prefix):])
    # print(list_jpg2)
    
    
    
    buch = {}
    
    for item in list_jpg2:
        temp = item.split("_image")
        key = temp[0]
        
        if key not in buch:
            buch[key] = []
        
        buch[key].append(item)
    
    for key in buch:
        print(key)
        print(buch[key])




if __name__ == "__main__":
    _main_()
