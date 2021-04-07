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
list_imges = GIL.get_image_dict(path_img, "jpg", seperator="_image")



# get first set
list_img_name = []
for item in list_imges[ list(list_imges.keys())[0] ]:
    list_img_name.append(item)
for item in list_imges[ list(list_imges.keys())[1] ]:
    list_img_name.append(item)
# list_img_name = list_imges[ list(list_imges.keys())[0] ]


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

def get_img_list_loaded():
    pass

# %% get small images

list_img = []
for img_name in list_img_name:
    list_img.append(load_img_dim(path_img + img_name))

# %% get blurred images

ksize = 9
list_blurr=[]
for img in list_img:
    list_blurr.append( cv2.GaussianBlur(img,(ksize,ksize),0) ) # blurr the images

# %% get sobel edge images

list_sobel = []

i_list = list(range(len(list_blurr)))
for i in i_list:
    img = list_blurr[i]
    kernelsize=5
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernelsize)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernelsize)
    sobel = np.sqrt( np.square(sobelx) + np.square(sobely) )
    list_sobel.append(sobel)

# %% get average mean
im_avg = list_sobel[0]

for i in i_list[1:]:
    im_avg = np.add(im_avg, list_sobel[i])

im_avg /= len(list_sobel)
im_avg_max = np.max(im_avg)
print(im_avg_max)
im_avg_max_255 = np.max(im_avg)/255

# %% get diff
list_diff = []
for img in list_sobel:
    list_diff.append( np.uint8( np.abs(img - im_avg)/im_avg_max_255 ) )
for img in list_diff:
    print(np.max(img))

# %% adaptive threshold in diff
list_thres = []
for img in list_diff:
    # th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,51,0)
    _,th = cv2.threshold(img, 63, 255, cv2.THRESH_BINARY)
    list_thres.append(th)

# %% matplot images

vmax = np.max(im_avg)
plt.close('all')

fig,ax = plt.subplots( 4, len(list_sobel)+1 )

i=0
row=0
for item in list_blurr:
    i+=1
    ax[row,i].imshow(item, cmap = 'gray',vmin=0, vmax=255) 
    ax[row,i].title.set_text("blurr {}".format(i)), ax[row,i].axis("off")
    # ax[row,i].title.set_text("diff {}".format(i)), ax[row,i].set_xticks([]), ax[row,i].set_yticks([])

i=0
row=1
ax[row,i].imshow(im_avg, cmap = 'gray',vmin=0, vmax=vmax) 
ax[row,i].title.set_text("avg"), ax[row,i].set_xticks([]), ax[row,i].set_yticks([])
for item in list_sobel:
    i+=1
    ax[row,i].imshow(item, cmap = 'gray',vmin=0, vmax=vmax) 
    ax[row,i].title.set_text("sobel {}".format(i)), ax[row,i].axis("off")
    
i=0
row=2
for item in list_diff:
    i+=1
    ax[row,i].imshow(item, cmap = 'gray',vmin=0, vmax=255) 
    ax[row,i].title.set_text("diff {}".format(i)), ax[row,i].axis("off")

i=0
row=3
for item in list_thres:
    i+=1
    ax[row,i].imshow(item, cmap = 'gray',vmin=0, vmax=255) 
    ax[row,i].title.set_text("thres {}".format(i)), ax[row,i].axis("off")


plt.tight_layout()
plt.axis("off")
fig.show()





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