# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:34:20 2021

@author: Admin


https://build-system.fman.io/pyqt5-tutorial
"""

import numpy as np
import matplotlib.pyplot as plt



"""First, we tell Python to load PyQt via the import statement: """
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc


"""Next, we create a QApplication with the command: """
app = qtw.QApplication([])"
"""This is a requirement of Qt: Every GUI app must have exactly one instance \
of QApplication. Many parts of Qt don't work until you have executed the \
above line. You will therefore need it in virtually every (Py)Qt app you write.

The brackets [] in the above line represent the command line arguments \
passed to the application. Because our app doesn't use any parameters, \
we leave the brackets empty."""


"""Now, to actually see something, we create a simple label:""" 
label = qtw.QLabel("hellow world")


"""Then, we tell Qt to show the label on the screen: """
label.show()


"""The last step is to hand control over to Qt and ask it to \
    "run the application until the user closes it". \
    This is done via the command: """
app.exec_()