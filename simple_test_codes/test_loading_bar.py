# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:54:06 2021

@author: Admin
"""

#Importing tqdm function of tqdm module 
from tqdm import tqdm  
from time import sleep 

#%%
for i in tqdm(range(200)):  
# Waiting for 0.01 sec before next execution 
   sleep(.01) 
   pass
   
#%%
from tqdm import tqdm 
for i in tqdm(range(int(5000000)), desc="Progress"): 
   pass
   
#%%
for i in tqdm(range(0, 100), disable=True): 
    sleep(.01) 
print("Done") 
#%%


for i in tqdm(range(10), desc="Super Progress"): 
    for j in tqdm(range(100), desc="Progress"): 
        sleep(.01) 
print("Done") 