# -*- coding: utf-8 -*-
"""
Created on Fri May 14 21:09:14 2021

@author: Admin

Based on V_01/manual_label_02.py
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import random

from ast import literal_eval

# import ImageHandlerModule as IHM

# %% FUNCTIONS

class ManualLabelHelper:
    def __init__(self, dir_extraction = "extracted",
                 file_focus =       "Focus_csv.csv", 
                 dir_focus_img =    "focus", 
                 dir_roi_img =      "roi",
                 DEBUG=False):
        self.DEBUG=DEBUG
        self.index = 0
        self.index_goto = 0
        self.index_max = 0
        
        self.set_dir_extracted(dir_extraction)
        self.set_path_file_focus(file_focus)
        self.set_dir_focus_img(dir_focus_img)
        
        # check if labels file exists. otherwise create it.
        self.df_startup()
        self.df_generate_setUnlabeled()
        
        # self.nav_goto_by_index()
        
        
        self.print_keybinds()
        self._new_fig_()
        self.nav_goto_rndm()
        pass
    
    # -------------------------------------------------------------------------
    def check_isdir(self,path):
        """Stop object creation, if no valid directory path is given. Returns the absolute path."""
        if (os.path.isdir(path) == False):
            raise Exception("Requires a legal directory path!")
        return os.path.abspath(path)
    def check_isfile(self,path):
        """Stop object creation, if no valid file path is given. Returns the absolute path."""
        if (os.path.isfile(path) == False):
            raise Exception("Requires a legal file path!")
        return os.path.abspath(path)
    
    def set_dir_extracted(self,path):
        """Sets the dir path to the extraction folder. 
        
        All other files are assumed to be relative to this folder, unless an abs_path is specified."""
        self._dir_extracted = self.check_isdir(path)
        pass
    def get_dir_extracted(self):
        return self._dir_extracted
    
    def set_dir_focus_img(self,path):
        """Sets the dir path to the focus_img folder. 
        
        Is assumed to be relative to the Extraction folder, unless an abs_path is specified."""
        if os.path.isabs(path):
            self._dir_focus_img = self.check_isdir( path )
        else:
            self._dir_focus_img = self.check_isdir( os.path.join(self._dir_extracted, path) )
        pass
    def get_dir_focus_img(self):
        return self._dir_focus_img
    
    def set_dir_roi_img(self,path):
        """Sets the dir path to the roi_img folder. 
        
        Is assumed to be relative to the Extraction folder, unless an abs_path is specified."""
        if os.path.isabs(path):
            self._dir_roi_img = self.check_isdir( path )
        else:
            self._dir_roi_img = self.check_isdir( os.path.join(self._dir_extracted, path) )
        pass
    def get_dir_roi_img(self):
        return self._dir_roi_img
    
    def set_path_file_focus(self,path:str):
        m = path.lower()
        if not m.endswith(".csv"):
            raise Exception("'path_file_focus' must end in '.csv'!")
        
        if os.path.isabs(path):
            self._path_file_focus = self.check_isfile( path )
        else:
            self._path_file_focus = self.check_isfile( os.path.join(self._dir_extracted, path) )
        pass
    
    
    def df_startup(self):
        """Setup for the dataframe:
            
        Defines file names, trys to load the label file or creats it, if not yet exists."""
        fname_labels = "Labels"
        
        dir_extracted = self._dir_extracted
        self._df_fname_labels_csv =  os.path.join(dir_extracted, "{}_csv.csv".format(fname_labels))
        self._df_fname_labels_scsv = os.path.join(dir_extracted, "{}_scsv.csv".format(fname_labels))
        self._txt_last_index_fname = os.path.join(dir_extracted, "last_index_labels.txt")
        
        # Check if [comma] separated value files exists
        f_labels_exists = os.path.isfile(self._df_fname_labels_csv)
        
        if f_labels_exists: # if exists: load
            self._df_labels = pd.read_csv(self._df_fname_labels_csv, index_col=0)
        else: # otherwise make new from focus file
            cols_label_default = [("has_bee",np.nan),
                                  ("img_sharp",np.nan),
                                  ("rel_pos_abdomen"," ")]
            # load the focus file as basis
            self._df_labels = pd.read_csv(self._path_file_focus, index_col=0)
            # append the columns with NaN as default values
            for item in cols_label_default:
                self._df_labels[item[0]] = item[1]
            
            # sort df
            self._df_labels.sort_values(by=["fname"], inplace=True)
            # save the new file
            self.df_store()
        
        self._df_size = len(self._df_labels)
        pass
    
    def df_store(self):
        """Saves the dataframe"""
        self._df_labels.to_csv(self._df_fname_labels_csv,  sep=",")
        self._df_labels.to_csv(self._df_fname_labels_scsv, sep=";")
        pass
    
    def df_generate_setUnlabeled(self):
        """Will search through 'has_bee' and 'img_sharp' columns for empty cells 
        and generate a set from this."""
        df = self._df_labels
        
        # check both 'has_bee' and 'img_sharp' columns for empty cells
        index1 = df['has_bee'].index[df['has_bee'].apply(np.isnan)]
        index2 = df['img_sharp'].index[df['img_sharp'].apply(np.isnan)]
        
        self._set_unlabeled_rowIndex = set(index1) | set(index2)
        pass
    
    def df_get_rndm_idx_from_setUnlabeled_rowIndex(self):
        """Get a random rowIndex from the set of unlabled_rowIndex."""
        return random.choice(tuple(self._set_unlabeled_rowIndex))
    def df_delete_idx_from_setUnlabeled_rowIndex(self,idx):
        """Removes the specified rowIndex from the set of unlabled_rowIndex."""
        self._set_unlabeled_rowIndex.discard(idx)
        pass
    
    # -------------------------------------------------------------------------
    
    def load_imgs(self,focus_name):
        self._img_focus = None
        self._img_roi = None
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def print_keybinds(self):
        print("[ESCAPE] Save to CSV")
        print("[SPACE]  Write & Next")
        pass
    
    
    
    

    
    # FIGURE functions
    def _new_fig_(self):
        plt.close('all')
        
        # make new figure
        self.fig = plt.figure()
        
        # # axes on left side for the image
        # self.ax_img = plt.axes([0, 0, 0.5, 0.95])
        # self.ax_img.axis("off")
        # self.ax_img.title.set_text("title")
        
        #int(self.fig.get_figwidth() * self.fig.dpi), int(self.fig.get_figheight() * self.fig.dpi)
        self.fig_shape = ( 600 , 600  )
        
        # self.figim = self.fig.figimage( np.random.random((100,100)) )
        # self.img_extr_temp = np.random.random((100,100))
        
        self.widget_figtext_left = \
            plt.figtext(0.01, 0.99, "title_top", 
                        va ="top", ha ="left", wrap = True, fontsize = 10) 
            
        
        # axes on top to show the ROI image
        self.ax_roi = plt.axes([0, 0.45, 1, 0.5]) #[left,bottom,width,height]
        # self.ax_roi.axis("off")
        self.ax_roi.title.set_text("")
        
        # axes on bottom left to show the focus image
        self.ax_foc = plt.axes([0, 0, 0.45, 0.45]) #[left,bottom,width,height]
        # self.ax_foc.axis("off")
        self.ax_foc.title.set_text("")
        
        
        # store informaiton about the remaining USER SPACE BOX
        self._fig_user_box_xywh = {"px":0.45, "py":0, "w":0.55, "h":0.45}
        uBox = self._fig_user_box_xywh
        # Helps with resizing/shaping a relative box to the real ubox (px,py,w,h)
        rshBx = lambda b,ub:(b[0]*ub["w"]+ub["px"], b[1]*ub["h"]+ub["py"], b[2]*ub["w"], b[3]*ub["h"])
        
        # axes for checkbox
        box = (0, 0, 0.4, 0.5)    # coordinates relative inside ubox
        box = rshBx(box,uBox)
        self.ax_checkbox = plt.axes(box) #[left,bottom,width,height]
        self.ax_checkbox_labels = ["+ BEE","- bee","+ SHARP","- sharp"]
        self.widget_checkbox = mpl.widgets.CheckButtons(self.ax_checkbox, 
                                                        self.ax_checkbox_labels)
        self.widget_checkbox.on_clicked(self.on_click_widget_checkbox)
        
        #axes for "Reload" button
        box = [0*0.3+0.4,   0*0.15,     0.3,    0.15]
        self.ax_button_Reload = plt.axes( rshBx(box,uBox) )
        self.widget_button_Reload = mpl.widgets.Button(self.ax_button_Reload, "Reload")
        self.widget_button_Reload.on_clicked(self.on_click_restore_from_df)
        
        #axes for "Save" button
        box = [1*0.3+0.4,   0*0.15,     0.3,    0.15]
        self.ax_button_save = plt.axes( rshBx(box,uBox) )
        self.widget_button_Save = mpl.widgets.Button(self.ax_button_save, "to DF")
        self.widget_button_Save.on_clicked(self.on_click_write_to_df)
        
        #axes for "Write and Next" button
        box = [0.4, 1*0.15, 0.6, 0.15]
        self.ax_button_WnN = plt.axes( rshBx(box,uBox) )
        self.widget_button_WnN = mpl.widgets.Button(self.ax_button_WnN, "> to DF & Next(rnd) >")
        self.widget_button_WnN.on_clicked(self.on_click_WnN)
        
        #axes for "Save CSV" button
        box = [1*0.3+0.4,   1-0.15,     0.3,    0.15]
        self.ax_button_saveCSV = plt.axes( rshBx(box,uBox) )
        self.widget_button_saveCSV = mpl.widgets.Button(self.ax_button_saveCSV, "Save CSV")
        self.widget_button_saveCSV.on_clicked(self.on_click_saveCSV)
        
        
        #axes for "Reset img position" button
        box = [0,   1-0.15,     0.3,    0.15]
        self.ax_button_RstImgPos = plt.axes( rshBx(box,uBox) )
        self.widget_button_RstImgPos = mpl.widgets.Button(self.ax_button_RstImgPos, "Clear Pos")
        self.widget_button_RstImgPos.on_clicked(self.on_click_RstImgPos)
        
        
        #axes for klick in image text
        box = [0.025,0.825,0,0];  box=rshBx(box,uBox)
        self.widget_figtext_pos_text = "Abdomen Position:\nclick into image to set."
        self.widget_figtext_pos = plt.figtext(box[0], box[1], 
                        self.widget_figtext_pos_text, 
                        va ="top", ha ="left", wrap = True, fontsize = 10, 
                        bbox ={'facecolor':'grey', 'alpha':0.1, 'pad':5}) 
        
        
        
        
        # axes for navigation buttons
        box = [0*0.2+0.4,   2*0.15+0.05,    0.2,    0.15]
        self.ax_button_j2_prev = plt.axes( rshBx(box,uBox) )
        self.widget_button_j2_prev = mpl.widgets.Button(self.ax_button_j2_prev, "< Prev")
        self.widget_button_j2_prev.on_clicked(self.on_click_j2_prev)
        
        box = [1*0.2+0.4,   2*0.15+0.05,    0.2,    0.15]
        self.ax_button_j2_rndm = plt.axes( rshBx(box,uBox) )
        self.widget_button_j2_rndm = mpl.widgets.Button(self.ax_button_j2_rndm, "Rndm")
        self.widget_button_j2_rndm.on_clicked(self.on_click_j2_rndm)
        
        box = [2*0.2+0.4,   2*0.15+0.05,    0.2,    0.15]
        self.ax_button_j2_next = plt.axes( rshBx(box,uBox) )
        self.widget_button_j2_next = mpl.widgets.Button(self.ax_button_j2_next, "Next >")
        self.widget_button_j2_next.on_clicked(self.on_click_j2_next)
        
        box = [2*0.2+0.4,   0.5+0.05,     0.2,    0.15]
        self.ax_button_j2_goto = plt.axes( rshBx(box,uBox) )
        self.widget_button_j2_goto = mpl.widgets.Button(self.ax_button_j2_goto, "Goto")
        self.widget_button_j2_goto.on_clicked(self.on_click_j2_goto)
        
        
        # make textbox widget
        box = [1*0.2+0.4,   0.55,     0.2,    0.15]
        self.ax_textbox_j2_goto = plt.axes( rshBx(box,uBox) )
        self.widget_textbox_j2_goto = mpl.widgets.TextBox(self.ax_textbox_j2_goto, "To:","0")
        self.widget_textbox_j2_goto.on_submit(self.txt_submit)
        
        
        
        thismanager = plt.get_current_fig_manager()
        thismanager.resize(600,600)
        # thismanager.window.setGeometry(self.posX, self.posY, self.w, self.h)
        
        
        
        # save to csv on figure closing
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # add key event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        pass
    
    
    
    def on_close(self,event):
        # save to csv on figure closing
        # self.func_save_CSV()
        self.fig.canvas.mpl_disconnect(self.cid)
        print("Figure closed.")
        pass
    
    def on_click_widget_checkbox(self, event):
        print("on_click_widget_checkbox")
        # print(type(event))
        # print(self.widget_checkbox.get_status())
        # print(event)
        
        if event in self.ax_checkbox_labels:
            old = self.ax_checkbox_status_old
            states = self.widget_checkbox.get_status()
            labels = list(self.ax_checkbox_labels)
            
            # detect where to toggle
            if event == self.ax_checkbox_labels[0]:     # if 1st button
                if states[0] and states[1]:                 # check if 2nd button was active
                    self.widget_checkbox.set_active(1)      # togge it (to off)
            elif event == self.ax_checkbox_labels[1]:   # if 2nd button
                if states[1] and states[0]:                 # check if 1st button was active
                    self.widget_checkbox.set_active(0)      # togge it (to off)
                    
            elif event == self.ax_checkbox_labels[2]:   # if 3rd button
                if states[2] and states[3]:                 # check if 4th button was active
                    self.widget_checkbox.set_active(3)      # togge it (to off)
            elif event == self.ax_checkbox_labels[3]:   # if 4th button
                if states[3] and states[2]:                 # check if 3rd button was active
                    self.widget_checkbox.set_active(2)      # togge it (to off)
            
            states = self.widget_checkbox.get_status()
            diff = np.bitwise_xor(states, old)
            # print(diff)
            for i in range(len(diff)):
                if diff[i]:
                    self.widget_checkbox.labels[i].set_color("r")
                else:
                    self.widget_checkbox.labels[i].set_color("k")
            pass
        
        self.fig.canvas.draw()  # update the display manually
        pass
    
    def on_click_saveCSV(self, event):
        print("on_click_saveCSV")
        # self.func_save_CSV()
        pass
    
    def on_click_RstImgPos(self, event):
        # TODO
        print("on_click_RstImgPos")
        
        pos_xy = None
        self.widget_figtext_pos_coords = pos_xy
        self.draw_imgFocus_abdomenPos(pos_xy)
        
        self.ax_foc.clear()
        self.ax_foc.axis("off")
        self.ax_foc.imshow(self._img_focus_show)
        
        self.widget_figtext_pos_text = "Abdomen Position:\n{}.".format(self.widget_figtext_pos_coords)
        
        self.widget_figtext_pos.set_text(self.widget_figtext_pos_text)
        if self.widget_figtext_pos_coords_old == self.widget_figtext_pos_coords:
            self.widget_figtext_pos.set_c(color="k")
        else:
            self.widget_figtext_pos.set_c(color="r")
        
        self.fig.canvas.draw()
        pass
    
    
    def on_click_write_to_df(self, event):
        print("on_click_write_to_df")
        # state = self.widget_checkbox.get_status()
        # self.df.loc[self.index,"isBee"] = state[0]
        # self.df.loc[self.index,"mostlyVisible"] = state[1]
        # self.df.loc[self.index,"hasMites"] = state[2]
        
        # self.index_goto = self.index
        # self.nav_goto_by_index()
        pass
    
    def on_click_WnN(self, event):
        print("on_click_WnN")
        # state = self.widget_checkbox.get_status()
        # self.df.loc[self.index,"isBee"] = state[0]
        # self.df.loc[self.index,"mostlyVisible"] = state[1]
        # self.df.loc[self.index,"hasMites"] = state[2]
        
        # self.index_goto = min( self.index+1, self.index_max )
        # self.nav_goto_by_index()
        pass
    
    def on_click_restore_from_df(self, event):
        print("on_click_restore_from_df")
        # self.index_goto = self.index
        # self.nav_goto_by_index()
        pass
    
    def on_click_j2_rndm(self, event):
        print("on_click_j2_rndm")
        # TODO
        # self.index_goto = 0
        # self.nav_goto_by_index()
        pass
    
    def on_click_j2_prev(self, event):
        print("on_click_j2_prev")
        # self.index_goto = max( 0, self.index-1 )
        # self.nav_goto_by_index()
        pass
    
    def on_click_j2_next(self, event):
        print("on_click_j2_next")
        # self.index_goto = min( self.index+1, self.index_max )
        # self.nav_goto_by_index()
        pass
    
    def on_click_j2_goto(self, event):
        print("on_click_j2_goto")
        # TODO
        # self.nav_goto_by_index()
        pass
    
    def on_click_canvas(self, event):
        print("on_click_canvas")
        # print(event)
        # TODO
        if event.inaxes == self.ax_foc:
            print("Clicked on focus image")
            # print(event)
            pos_xy = (int(event.xdata), int(event.ydata))
            self.widget_figtext_pos_coords = pos_xy
            self.draw_imgFocus_abdomenPos(pos_xy)
            
            self.ax_foc.clear()
            self.ax_foc.axis("off")
            self.ax_foc.imshow(self._img_focus_show)
            
            self.widget_figtext_pos_text = "Abdomen Position:\n{}.".format(self.widget_figtext_pos_coords)
            self.widget_figtext_pos.set_text(self.widget_figtext_pos_text)
            if self.widget_figtext_pos_coords_old == self.widget_figtext_pos_coords:
                self.widget_figtext_pos.set_c(color="k")
            else:
                self.widget_figtext_pos.set_c(color="r")
            
            self.fig.canvas.draw()
        pass
    
    
    
    def txt_submit(self, event):
        if event=="":
            # if tectbox left empty, then just show the current index
            self.index_goto = self.index
        elif event.isnumeric():
            # else: if the string is a number, we will set for as the jump-to index.
            # (additionally: only allow to jump to the max possible index)
            self.index_goto = min( int(event), self.index_max )
        
        # if self.index_goto is unchanged (no valid input), then the display resets.
        # otherwise, the display updates to the entered value.
        self.widget_textbox_j2_goto.set_val( str(self.index_goto) )
        
        self.fig.canvas.draw()  # update the display manually
        
        # print("DEBUG index_goto:", self.index_goto)
        pass
    
    def on_key(self,event):
        self.keypressed = event.key
        print('you pressed', event.key)
        
        if event.key == "escape":
            self.on_click_saveCSV(None)
        elif event.key == " ":
            self.on_click_WnN(None)
        pass
    
    def nav_goto_rndm(self):
        rowIdx = self.df_get_rndm_idx_from_setUnlabeled_rowIndex()
        self.nav_goto_rowIdx(rowIdx)
        pass
    
    def nav_goto_position(self,position:int):
        # only if the index is inside the possible range of positions
        if position in [0, self._df_size-1]:
            row = self._df_labels.iloc[position]
            rowIdx = row.name
            self.nav_goto_rowIdx(rowIdx)
        # Else: nothing
        pass
    
    def nav_goto_rowIdx(self,rowIdx):
        self._rowIdx = rowIdx
        self._df_row = self._df_labels.loc[self._rowIdx]         # fetch the row
        self._df_position = self._df_labels.index.get_loc(self._rowIdx)   # fetch the position as integer
        
        img_roi = cv2.imread(self._df_row["roi_fpath"])          # fetch roi image
        self._img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)    # make to RGB!!!
        
        img_focus = cv2.imread(self._df_row["fpath"])            # fetch focus image
        self._img_focus_backup = cv2.cvtColor(img_focus, cv2.COLOR_BGR2RGB)
        self._img_focus_over = np.zeros(img_focus.shape, dtype=np.uint8) # make empty overlay of focus img
        self.draw_imgFocus_centerContour()      # draw the center contour
        
        
        
        # check status of saved data in DF
        index_list = ["has_bee","img_sharp","rel_pos_abdomen"]
        _has_bee =    self._df_row["has_bee"] 
        if np.isnan(_has_bee): # we must differentiate between NaN and T/F data!
            _has_bee_Y = False
            _has_bee_N = False
        else:
            _has_bee_Y = _has_bee > 0
            _has_bee_N = not _has_bee_Y
            
        _img_sharp =  self._df_row["img_sharp"] 
        if np.isnan(_img_sharp):
            _img_sharp_Y = False
            _img_sharp_N = False
        else:
            _img_sharp_Y = _has_bee > 0
            _img_sharp_N = not _img_sharp_Y
            
        _rel_pos_abdomen = self._df_row["rel_pos_abdomen"]
        if _rel_pos_abdomen in [" ",""]: #special empty case for _rel_pos_abdomen
            _rel_pos_abdomen = tuple()
        else:
            _rel_pos_abdomen = tuple( literal_eval(_rel_pos_abdomen) )
        
        if len(_rel_pos_abdomen)==2:
            if type(_rel_pos_abdomen[0])==int and type(_rel_pos_abdomen[1])==int:
                coords = _rel_pos_abdomen
            else:
                coords = None
        else:
            coords = None
        
        self.draw_imgFocus_abdomenPos(coords)
        self.widget_figtext_pos_coords_old = coords
        self.widget_figtext_pos_coords = coords
                
            
        self.ax_checkbox_status_old = (_has_bee_Y, _has_bee_N, _img_sharp_Y, _img_sharp_N)
        self.ax_checkbox_status = list(self.ax_checkbox_status_old)
        
        if self.widget_figtext_pos_coords==None:
            self.widget_figtext_pos_text = "Abdomen Position:\nclick into image to set."
        else:
            self.widget_figtext_pos_text = "Abdomen Position:\n{}.".format(self.widget_figtext_pos_coords)
        
        
        
        self.txt_submit(str(self._df_position))
        self.update_fig()
        pass
    
    def draw_imgFocus_centerContour(self):
        """Write contour-center-cross into the overlay (blue channel)."""
        shape = self._img_focus_backup.shape
        dim = np.flip(shape[0:2])
        img = np.zeros(shape[0:2], dtype=np.uint8) # clear blue channel
        
        # Fetch coors of contour center
        posC = literal_eval( self._df_row["pos_center"] )
        posA = literal_eval( self._df_row["pos_anchor"] )
        pX = posC[0] - posA[0] # calc pos in focus image
        pY = posC[1] - posA[1]
        cv2.line(img, (pX,0), (pX,dim[1]), 255,2) #draw cross goring through center
        cv2.line(img, (0,pY), (dim[0],pY), 255,2)
        
        # Fetch the info of the minAreaRect
        minAreaRect = literal_eval( self._df_row["minAreaRect"] )
        box = cv2.boxPoints(minAreaRect) # make a box
        box = np.int0(box)
        cv2.drawContours(img,[box],0,255,1,cv2.LINE_AA ,offset=(- posA[0],- posA[1])) # draw box
        
        # write this img to blue channel of overlay
        self._img_focus_over[:,:,2] = img 
        self._draw_imgFocus_overlay()
        pass
    
    def draw_imgFocus_abdomenPos(self, pos_xy, radius=10):
        """Write user bee pos into the overlay (red channel)."""
        shape = self._img_focus_backup.shape
        dim = np.flip(shape[0:2])
        img = np.zeros(shape[0:2], dtype=np.uint8) # clear red channel
        
        # if the coords are int, then we draw the circle - otherwise we leave the red channel empty
        if type(pos_xy) in [list,tuple]:
            if len(pos_xy)==2:
                cv2.circle(img, (int(pos_xy[0]), int(pos_xy[1])), radius, 255, -1)
        
        # write this img to red channel of overlay
        self._img_focus_over[:,:,0] = img 
        self._draw_imgFocus_overlay()
        pass
    
    def _draw_imgFocus_overlay(self):
        """Write the overlay into '_img_focus_show'."""
        self._img_focus_show = np.float32( self._img_focus_backup )     # prepare bg img for weighted addition
        
        mask = cv2.bitwise_or(self._img_focus_over[:,:,0], self._img_focus_over[:,:,2])   # make mask just from OR-ing the R and B channels together
        cv2.accumulateWeighted(self._img_focus_over, self._img_focus_show, 0.25, mask=mask)
        
        self._img_focus_show = np.uint8( self._img_focus_show ) # back to uint8
        pass
    
    
    def update_fig(self):
        
        title = "{} ({})".format(self._rowIdx, self._df_position) 
        # self.ax_img.title.set_text(title)
        self.widget_figtext_left.set_text(title)
        
        self.ax_roi.clear()
        self.ax_roi.axis("off")
        self.ax_roi.imshow(self._img_roi)
        
        self.ax_foc.clear()
        self.ax_foc.axis("off")
        self.ax_foc.imshow(self._img_focus_show)
        
        
        self.ax_checkbox.clear()
        self.widget_checkbox = mpl.widgets.CheckButtons(self.ax_checkbox, 
                                                        self.ax_checkbox_labels,
                                                        self.ax_checkbox_status_old)
        self.widget_checkbox.on_clicked(self.on_click_widget_checkbox)
        
        
        self.widget_figtext_pos.set_text(self.widget_figtext_pos_text)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click_canvas)
        self.fig.canvas.draw()  # update the display manually
        pass

# %% 


# %% 


# %% 
myHLH = ManualLabelHelper()
my_df = myHLH._df_labels

# myHLH._new_fig_()
# myHLH.update_fig()
# myHLH.fig.show()

# b = myHLH.widget_checkbox
# a = myHLH._a