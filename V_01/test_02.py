# -*- coding: utf-8 -*-
"""
Spyder Editor

This Test involves gradient for ROI detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import GetImageList as GIL


path_img = "../images/"
num_img_sets,list_imges = GIL.get_image_list(path_img, "jpg", seperator="_image")
print("Got {} image sets in dict!".format(num_img_sets))


# get first set
list_img_name = list_imges[ list(list_imges.keys())[0] ]


cv2.destroyAllWindows()
plt.close('all')



def load_img_scale(imgpath, div_factor=4):
    img = cv2.imread(imgpath, 0) # load img as grayscale
    
    width =     int(img.shape[1] / div_factor)
    height =    int(img.shape[0] / div_factor)
    dim = (width, height)
    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized

def load_img_dim(imgpath, dim=(400,300)):
    # check formatting
    if not type(dim) in [list, tuple]: return None
    if len(dim) != 2: return None
    
    img = cv2.imread(imgpath, 0) # load img as grayscale
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    return resized


list_img = []
for img_name in list_img_name:
    list_img.append(load_img_dim(path_img + img_name))

# %%
ksize = 31
list_blurr=[]
for img in list_img:
    list_blurr.append( cv2.GaussianBlur(img,(ksize,ksize),0) ) # blurr the images
    
# list_threshold=[]
# for img in list_blurr:
#     _,thresh = cv2.threshold(img,31,255,cv2.THRESH_TOZERO)
#     list_threshold.append( thresh ) 

#------------------------------------------------------------
# %%
index=2
# threshold = 1

img = list_blurr[index]

kernelsize=5
laplacian = cv2.Laplacian(img,cv2.CV_64F,ksize=kernelsize)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernelsize)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernelsize)

laplacian = np.abs(laplacian)
# laplacian //= threshold
# laplacian *= threshold
# _,laplacian_t = cv2.threshold(laplacian,0,255,cv2.THRESH_TOZERO)

sobel = ( np.square(sobelx) + np.square(sobely) )
# sobel //= 64
# sobel *= 16
# _,sobel_t = cv2.threshold(sobel,0,255,cv2.THRESH_TOZERO)

im = [img, laplacian, sobel]
tiles = ['Original '+str(index), 'laplacian', 'sobel']
# plt.close('all')
fig,ax = plt.subplots(3)
i=0
for item in im:
    ax[i].imshow(im[i],cmap = 'gray',vmin=0) 
    ax[i].title.set_text(tiles[i])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    i+=1

plt.tight_layout()
fig.show()
print(np.max(sobel))





# %%
plt.close('all')
# %%












# %%

# print("Hit any key to exit...")
# cv2.waitKey(0)
# cv2.destroyAllWindows()




def main():
    pass


# ----------------------------------------------------------------------------
if __name__== "__main__":
	print("Calling main function.)\n")
	main()