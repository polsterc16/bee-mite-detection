# -*- coding: utf-8 -*-
"""
Spyder Editor

(MEAN on resize)
RESIZE
DIFF
SOBEL
(THRES)

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import GetImageList as GIL


path_img = "../images/"

cv2.destroyAllWindows()
plt.close('all')


# %%



# %%

class img_handler_mean_diff_sobel:
    def __init__(self,img_path="../images/", img_dim=(400,300), \
                 mean_alpha=0.1):
        self.set_img_path(img_path)
        self.set_img_dim(img_dim)
        # self.set_blurr_gaussian_size(blurr)
        self.set_mean_alpha(mean_alpha)
        
        
        print("-- Handler Object created")
        self.restart()
        pass
    
    def set_img_path(self,img_path):
        self.img_path = img_path
        pass
    def set_img_dim(self, img_dim):
        # image dimension must be an int tuple
        assert type(img_dim) in [list,tuple]
        assert len(img_dim) == 2
        self.img_dim = ( int(img_dim[0]), int(img_dim[1]) )
        pass
    # def set_blurr_gaussian_size(self, blurr_gaussian_size):
    #     # gauss kernel size msut be an odd integer
    #     self.blurr_gaussian_size = int(blurr_gaussian_size)
    #     if (self.blurr_gaussian_size % 2) != 1:
    #         self.blurr_gaussian_size += 1
    #     pass
    def set_mean_alpha(self, mean_alpha):
        self.mean_alpha = float(mean_alpha)
        pass
    
    
    def load_img(self, index):
        # get image path to indexed element in img_name_list
        imgpath_current = self.img_path + self.img_name_list[index]
        
        # set original image and resized version
        self.img_set_original = cv2.imread(imgpath_current, 0)
        self.img_set_resize = cv2.resize(self.img_set_original, \
                                     self.img_dim, \
                                     interpolation = cv2.INTER_AREA )
        pass
    # def blurr_img(self):
    #     ksize = (self.blurr_gaussian_size,self.blurr_gaussian_size)
    #     self.img_set_blurred = cv2.GaussianBlur(self.img_set_diff, ksize, 0)
    #     pass
    def weighted_mean(self, overwrite=False):
        if (overwrite):
            # overwrite mean to current blurred image
            self.img_set_mean = np.float32( self.img_set_resize )
        else:
            # weighted accumulation
            cv2.accumulateWeighted(self.img_set_resize, \
                                    self.img_set_mean, \
                                    self.mean_alpha)
        pass
    def difference(self):
        # self.img_set_diff = np.int16(self.img_set_mean) - np.int16(self.img_set_resize)
        self.img_set_diff = np.int16(self.img_set_mean) - np.int16(self.img_set_resize)
        pass
    
    def sobel(self):
        self.img_set_sobelx = cv2.Sobel(self.img_set_diff, cv2.CV_64F,1,0,ksize=7)
        self.img_set_sobely = cv2.Sobel(self.img_set_diff, cv2.CV_64F,0,1,ksize=7)
        self.img_set_sobel = np.sqrt( np.square(self.img_set_sobelx) + np.square(self.img_set_sobely) )
        pass
    
    def restart(self):
        print()
        print("-- Restarting Handler")
        
        # reset (create) the img index
        self.img_index = 0
        
        # clear (create) the list of image names and fill them
        self.img_name_list = []
        GIL.get_image_list(self.img_name_list, self.img_path, "jpg")
        print("Image path: {}".format(self.img_path))
        
        # load the first image
        self.load_img(self.img_index)
        print("Images are resized to: {}".format(self.img_dim))
        
        # perform weighted mean (set mean to current image)
        self.weighted_mean(overwrite=True)
        print("Weighted Mean Alpha: {}".format(self.mean_alpha))
        
        
        self.img_set_diff = None
        self.img_set_sobel = None
        pass
    
    def iterate(self, times = 1):
        for i in range(times):
            # increment the index and check if index is still inside the list
            self.img_index += 1
            if (self.img_index >= len(self.img_name_list)):
                print("No more Iterations possible. \
                      Index has reached end of img_name_list.")
                return
            
            # update weighted mean (BEFORE loading a new image)
            self.weighted_mean()
            
            # load the current image
            self.load_img(self.img_index)
            
            # calc difference from blurr to mean
            self.difference()
            
            
            # sobel of difference
            self.sobel()
        pass
    
    def iter_and_plot(self, times = 1):
        self.iterate(times)        
        plt.close('all')
        
        self.fig, self.ax = plt.subplots( 2,2 )
        
        row,col = 0,0
        title = "resize: {}".format(self.img_index)
        self.ax[row,col].imshow(self.img_set_resize, cmap = 'gray',vmin=0, vmax=255) 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        
        row,col = 0,1
        title = "mean"
        self.ax[row,col].imshow(self.img_set_mean, cmap = 'gray',vmin=0, vmax=255) 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        row,col = 1,0
        title = "diff"
        self.ax[row,col].imshow(self.img_set_diff, cmap = 'gray',vmin=-255, vmax=255) 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        row,col = 1,1
        title = "sobel"
        self.ax[row,col].imshow(self.img_set_sobel, cmap = 'gray') 
        self.ax[row,col].title.set_text(title), self.ax[row,col].axis("off")
        
        
    def iter_and_plot_update(self, times = 1):
        self.iterate(times)        
        # plt.close('all')
        
        # fig,ax = plt.subplots( 2,2 )
        
        
        row,col = 0,0
        title = "resize: {}".format(self.img_index)
        self.ax[row,col].imshow(self.img_set_resize, cmap = 'gray',vmin=0, vmax=255) 
        self.ax[row,col].title.set_text(title)
        
        row,col = 0,1
        self.ax[row,col].imshow(self.img_set_mean, cmap = 'gray',vmin=0, vmax=255) 
        
        row,col = 1,0
        self.ax[row,col].imshow(self.img_set_diff, cmap = 'gray',vmin=-255, vmax=255) 
        
        row,col = 1,1
        self.ax[row,col].imshow(self.img_set_sobel, cmap = 'gray') 
        



# %% get small images

"""
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

"""



# %%
plt.close('all')
# %%





def main():
    myHandler = img_handler_mean_diff_sobel()
    # %%
    myHandler.iter_and_plot()
    # %%
    pass


# %%
if __name__== "__main__":
    print("Calling main function.)\n")
    # main()
    
    myHandler = img_handler_mean_diff_sobel(mean_alpha=0.05)
    # %%
    myHandler.iter_and_plot(times=39)
    # %%
    myHandler.iter_and_plot_update(times=10)
    # %%
    
    
    
    
    
    
    
    
    
    
    
    