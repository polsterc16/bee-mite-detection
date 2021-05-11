# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:34:20 2021

@author: Admin


https://build-system.fman.io/pyqt5-tutorial
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys

myPath = "C:\\Users\\Admin\\0_FH_Joanneum\\ECM_S3\\PROJECT\\bee_images\\01_8_2020\\5"
myFile1 = "0_0_image0000_0.jpg"
myFile2 = "0_0_image0002_0.jpg"
myFile3 = "0_0_image0003_0.jpg"
myFile4 = "0_0_image0004_0.jpg"
path = os.path.join(myPath,myFile1)

version = "002"

#%%

class MainWindow(qtw.QWidget):
    def __init__(self):
        """MainWindow constructor"""
        super().__init__()
        # Main UI code goes here
        
        
        # End main UI code
        self.show()


#%%
if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    sys.exit(app.exec())
    
#%%
a=cv2.imread(path)