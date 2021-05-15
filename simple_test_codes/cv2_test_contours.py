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
cv2.rectangle(img, (25,25), (75,50), 255, -1)
cv2.circle(img, (150,50), 15, 255, -1)
cv2.imshow("win",img)

#%%
ret, thresh = cv2.threshold(img, 127, 255, 0)
_,contours,_= cv2.findContours(thresh, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE)

img3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

cv2.drawContours(img3, contours, -1, (0,127,255), 3)
cv2.imshow("win2",img3)

c0=contours[0]
c1=contours[1]
# c0=c0[:,0,:]
#%%
# zeichne contouren, die über das bild hinausgehen (es muss mindestens ein punkt innerhalb sein)


c2 = [[[-10, -10]],

       [[30, -10]],

       [[30, 10]],
       [[60, 10]],
       [[60, 300]],

       [[60, 300]]]

c2 = np.array(c2)

#cv2.drawContours(img3, [c2], -1, (255,0,120), -1)
cv2.imshow("win2",img3)

#%%
# testen, wie man am besten einen offset in die contour reinbringt
cx=-20
cy=-10

    
c_off = np.array([ [[v[0][0]-cx,v[0][1]-cy]] for v in c1 ])
print(c1)
print(c_off)


cv2.drawContours(img3, [c_off], -1, (255,255,0), 1)
cv2.imshow("win2",img3)

#%%
# contours
areas = [int( cv2.contourArea(c) ) for c in contours]
ell = cv2.fitEllipse(c0)
minRect = cv2.minAreaRect(c_off)

# ((315.8999938964844, 163.79998779296875), (45.520748138427734, 73.4285659790039), -82.87498474121094)
# ((318.6860046386719, 183.79737854003906), (89.6900405883789, 70.09237670898438), -71.20011138916016)
# ((315.6593017578125, 221.61062622070312), (59.27708053588867, 115.2726821899414), -14.620874404907227)
# ((300.5, 271.0), (81.0, 56.0), -0.0)

#%%
# rect = cv2.minAreaRect(cnt)
# box = cv2.cv.BoxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
# box = np.int0(box)
# cv2.drawContours(im,[box],0,(0,0,255),2)

rect = (minRect[0], minRect[1], minRect[2]-45)
box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
box = np.int0(box)
cv2.drawContours(img3,[box],0,(0,0,255),2)

cv2.imshow("win3",img3)









