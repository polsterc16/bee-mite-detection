# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:31:50 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys


#%%
cv2.destroyAllWindows()

im1 = cv2.imread("colorimg_small.png")
dim = np.flip( im1.shape[0:2] )
im1 = cv2.resize(im1, (dim[0]*20,dim[1]*20))
cv2.imshow("1", im1)

im2 = np.zeros(im1.shape, dtype=np.uint8)
im2 = cv2.circle(im2, (150,100), 60, (255,127,255), -1)
cv2.imshow("2", im2)

# im1 = np.float32(im1)

# cv2.addWeighted(src1, alpha, src2, beta, gamma)(im2, im1, -0.25,beta)#,mask=im2[:,:,0])
# temp=im1.copy()
# im1 = np.int16(im1)
# _,im1 = cv2.threshold(im1, 0, 255, cv2.THRESH_TOZERO)
# im1 = np.uint8(im1)

# cv2.imshow("3", im1)

# didnt work

#%%
im = cv2.dilate(im2, np.ones((15,5),dtype=np.uint8))
cv2.imshow("3", im)
#dilation funktioniert auch in farbe

im_inv = cv2.bitwise_not(im)
cv2.imshow("4", im_inv)