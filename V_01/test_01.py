# -*- coding: utf-8 -*-
"""
Spyder Editor

This Test involves taking the running average of a handful of images 
to then calculate the difference to the individual images.

- Results are lackluster
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

path_img = "D:/ECM_PROJECT/images/"

list_img_name = []
list_img_name.append("12_30_image0000_0.jpg")
list_img_name.append("12_30_image0002_0.jpg")
list_img_name.append("12_30_image0003_0.jpg")
list_img_name.append("12_30_image0005_0.jpg")
list_img_name.append("12_30_image0006_0.jpg")

cv2.destroyAllWindows()


# %%

def load_img_small(imgpath, div_factor=4):
    img = cv2.imread(imgpath, 0) # load img as grayscale
    
    width =     int(img.shape[1] / div_factor)
    height =    int(img.shape[0] / div_factor)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized



# %%

list_img = []

for img_name in list_img_name:
    list_img.append(load_img_small(path_img + img_name,10))

# %%
ksize = 21
list_blurr=[]
for img in list_img:
    list_blurr.append( cv2.GaussianBlur(img,(ksize,ksize),0) ) # blurr the images

# i=0
# for img in list_img:
#     i=i+1
#     cv2.imshow('blur_'+str(i),img)

# %%
avg = np.float32(list_blurr[0])

for img in list_blurr:
    cv2.accumulateWeighted(img,avg,0.1)
avg = np.uint8(avg)

cv2.imshow('image - avg',avg)

# %%
list_diff = []
for img in list_blurr:
    list_diff.append( np.uint8( np.abs(np.int16(img) - np.int16(avg)) ) )

index = 4
cv2.imshow('image - avg',avg)
cv2.imshow('blurr',list_blurr[index])
cv2.imshow('diff',list_diff[index])


# %%












# print("Hit any key to exit...")
# cv2.waitKey(0)
# cv2.destroyAllWindows()




def main():
    pass


# ----------------------------------------------------------------------------
if __name__== "__main__":
	print("Calling main function.)\n")
	main()