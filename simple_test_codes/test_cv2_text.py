# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:58:12 2021

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


fontFace = cv2.FONT_HERSHEY_PLAIN 
color=(0,255,255)
size = 1
(tx,ty),_=cv2.getTextSize("A", fontFace, size, 2)