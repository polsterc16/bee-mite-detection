# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:28:56 2021

@author: Admin
"""

import numpy as np


class FOO:
    def __init__(self, number):
        self.a = np.ones((number,number*2), np.uint8)
        pass
    def get_a(self):
        return self.a

class BAR:
    def __init__(self):
        self.b=None
        pass
    def set_b(self,new_b):
        self.b=new_b
        pass
    def get_b(self):
        return self.b


 
#%%
if __name__== "__main__":
    
    foo = FOO(2)
    bar = BAR()
    
    print(bar.get_b())
    
    bar.set_b(foo.get_a())
    
    print(bar.get_b())
    
    bar.b[0][0]=4
    
    print(bar.get_b())
    print(foo.get_a())
    
    pass