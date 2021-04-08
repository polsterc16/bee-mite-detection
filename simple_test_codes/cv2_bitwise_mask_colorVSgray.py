# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:19:36 2021

@author: Admin

Test if we need a 3channel mask for color --> yes we do
"""

import numpy as np
import cv2


img_c = cv2.imread("colorimg_small.png")
img_g = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)

print("shape ",img_c.shape)
cv2.imshow("w",img_c)

scale = 10

dim=(int(img_c.shape[1]*scale),int(img_c.shape[0]*scale))

img_2 = cv2.resize(img_c, dim, interpolation = cv2.INTER_AREA )
cv2.imshow("w",img_2)