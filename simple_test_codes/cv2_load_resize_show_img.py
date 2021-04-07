# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 17:52:26 2021

@author: Admin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

dirpath = "C:\\Users\\Admin\\0_FH_Joanneum\\ECM_S3\\PROJECT\\bee_images\\01_8_2020\\5"
file = "53_46_image0003_0.jpg"

fpath = os.path.join(dirpath, file)

resize = (400,300)


img_1 = cv2.imread(fpath, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2 = cv2.resize(gray, resize)

cv2.imshow("resized", img_2)