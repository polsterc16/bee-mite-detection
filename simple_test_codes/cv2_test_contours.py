# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:37:12 2021

@author: Admin

https://www.docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

cv2.destroyAllWindows()

#%%
img = cv2.imread("colorimg_small.png")
img = cv2.resize(img, (200,100), interpolation = cv2.INTER_AREA )

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img[:,:]=0
cv2.circle(img, (50,50), 25, 255, 10)
cv2.circle(img, (150,50), 15, 255, -1)
cv2.imshow("win",img)

#%%
ret, thresh = cv2.threshold(img, 127, 255, 0)
_,contours,_= cv2.findContours(thresh, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)

img3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

cv2.drawContours(img3, contours, -1, (0,127,255), 3)
cv2.imshow("win2",img3)

c0=contours[0]
c0=c0[:,0,:]