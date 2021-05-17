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

from tqdm import tqdm
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
        self._storing_counter = 0
        
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
        
        self._storing_counter = 0
        
        print("Stored {}".format(self._df_fname_labels_csv))
        pass
    
    def df_generate_setUnlabeled(self):
        """Will search through 'has_bee' and 'img_sharp' columns for empty cells 
        and generate a set from this."""
        df = self._df_labels
        
        # check both 'has_bee' and 'img_sharp' columns for empty cells
        index1 = df['has_bee'].index[df['has_bee'].apply(np.isnan)]
        index2 = df['img_sharp'].index[df['img_sharp'].apply(np.isnan)]
        
        self._set_unlabeled_rowIndex = set(index1) & set(index2)
        self._set_unlabeled_rowIndex_partly = set(index1) ^ set(index2)
        
        ul = len(self._set_unlabeled_rowIndex) + len(self._set_unlabeled_rowIndex_partly)
        print("Unlabled imgs: {} -> {}".format(ul, self._df_size-ul))
        pass
    
    def df_get_rndm_idx_from_setUnlabeled_rowIndex(self):
        """Get a random rowIndex from the set of unlabled_rowIndex."""
        # return partly labeled items first
        if len(self._set_unlabeled_rowIndex_partly)>0:
            return random.choice(tuple(self._set_unlabeled_rowIndex_partly))
        else:
            return random.choice(tuple(self._set_unlabeled_rowIndex))
    def df_delete_idx_from_setUnlabeled_rowIndex(self,idx):
        """Removes the specified rowIndex from the set of unlabled_rowIndex."""
        self._set_unlabeled_rowIndex.discard(idx)
        self._set_unlabeled_rowIndex_partly.discard(idx)
        pass
    
    # -------------------------------------------------------------------------
    
    
    def print_keybinds(self):
        print("[ESCAPE] Save to CSV")
        print("[SPACE]  Write & Next")
        pass
    
    
    
    # FIGURE functions
    def _new_fig_(self):
        plt.close('all')
        
        # make new figure
        self.fig = plt.figure()
        
        #int(self.fig.get_figwidth() * self.fig.dpi), int(self.fig.get_figheight() * self.fig.dpi)
        self.fig_shape = ( 600 , 600  )
        
        
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
        self.widget_button_Reload.on_clicked(self.on_click_reload)
        
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
        self.df_store()
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
            # print(states)
            
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
            
            # print(states)
            states = self.widget_checkbox.get_status()
            diff = np.bitwise_xor(states, old)
            # print(diff)
            for i in range(len(diff)):
                if diff[i]:
                    self.widget_checkbox.labels[i].set_color("r")
                else:
                    self.widget_checkbox.labels[i].set_color("k")
            
            self.ax_checkbox_status = states
            pass
        
        self.fig.canvas.draw()  # update the display manually
        pass
    
    def on_click_saveCSV(self, event):
        print("on_click_saveCSV")
        self.df_store()
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
        self.write_to_df()
        # reload itself
        self.nav_goto_rowIdx(self._rowIdx)
        pass
    
    def on_click_WnN(self, event):
        print("on_click_WnN")
        self.write_to_df()
        # goto rndm
        self.nav_goto_rndm()
        pass
    
    def on_click_reload(self, event):
        print("on_click_restore_from_df")
        
        self.nav_goto_rowIdx(self._rowIdx)
        pass
    
    def on_click_j2_rndm(self, event):
        print("on_click_j2_rndm")
        self.nav_goto_rndm()
        pass
    
    def on_click_j2_prev(self, event):
        print("on_click_j2_prev")
        # dont go farther than the minimum
        self.nav_goto_position( max([self._df_position-1, 0]) )
        pass
    
    def on_click_j2_next(self, event):
        print("on_click_j2_next")
        # dont go farther than the max size
        self.nav_goto_position( min([self._df_position+1, self._df_size]) )
        pass
    
    def on_click_j2_goto(self, event):
        print("on_click_j2_goto")
        self.nav_goto_position( self._df_position_goto )
        pass
    
    def on_click_canvas(self, event):
        # print("on_click_canvas")
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
            self._df_position_goto = self._df_position
        elif event.isnumeric():
            # else: if the string is a number, we will set for as the jump-to index.
            # (additionally: only allow to jump to the max possible index)
            self._df_position_goto = max( [min( [int(event), self._df_size-1] ), 0] )
        
        # if self.index_goto is unchanged (no valid input), then the display resets.
        # otherwise, the display updates to the entered value.
        self.widget_textbox_j2_goto.set_val( str(self._df_position_goto) )
        
        self.fig.canvas.draw()  # update the display manually
        
        # print("DEBUG index_goto:", self.index_goto)
        pass
    
    def on_key(self,event):
        self.keypressed = event.key
        
        
        if event.key == "escape":
            print('you pressed', event.key)
            self.on_click_saveCSV(None)
        elif event.key == " ":
            print('you pressed', event.key)
            self.on_click_WnN(None)
        pass
    
    
    
    def write_to_df(self):
        cbx = self.ax_checkbox_status
        pos_abd = self.widget_figtext_pos_coords
        # print(cbx)
        # print(pos_abd)
        # row = self._df_labels.at[self._rowIdx, 
        # print(cbx)
        if (cbx[0]==False and cbx[1]==False):
            self._df_labels.at[self._rowIdx, "has_bee"] = np.nan
        elif cbx[0]==True:
            self._df_labels.at[self._rowIdx, "has_bee"] = 1
        else:
            self._df_labels.at[self._rowIdx, "has_bee"] = 0
        
        if (cbx[2]==False and cbx[3]==False):
            self._df_labels.at[self._rowIdx, "img_sharp"] = np.nan
        elif cbx[2]==True:
            self._df_labels.at[self._rowIdx, "img_sharp"] = 1
        else:
            self._df_labels.at[self._rowIdx, "img_sharp"] = 0
        
        self._df_labels.at[self._rowIdx, "rel_pos_abdomen"] = str(pos_abd)
        
        print("wrote {} to DF.".format(self._rowIdx))
        
        self.df_delete_idx_from_setUnlabeled_rowIndex(self._rowIdx)
        
        
        self._storing_counter += 1 # inc counter
        if (self._storing_counter >= 10):
            self._storing_counter = 0
            self.df_store()
        
        pass
    
    def nav_goto_rndm(self):
        rowIdx = self.df_get_rndm_idx_from_setUnlabeled_rowIndex()
        self.nav_goto_rowIdx(rowIdx)
        pass
    
    def nav_goto_position(self,position:int):
        # only if the index is inside the possible range of positions
        if position in range(0, self._df_size):
            row = self._df_labels.iloc[position]
            rowIdx = row.name
            self.nav_goto_rowIdx(rowIdx)
        # Else: nothing
        pass
    
    def nav_goto_rowIdx(self,rowIdx):
        self._rowIdx = rowIdx
        self._df_row = self._df_labels.loc[self._rowIdx]         # fetch the row
        self._df_position = self._df_labels.index.get_loc(self._rowIdx)   # fetch the position as integer
        self._df_position_goto = self._df_position
        
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
            _img_sharp_Y = _img_sharp > 0
            _img_sharp_N = not _img_sharp_Y
            
        _rel_pos_abdomen = self._df_row["rel_pos_abdomen"]
        coords = None # default assignment
        if _rel_pos_abdomen not in [" ",""]: #special empty case for _rel_pos_abdomen
            _rel_pos_abdomen =  literal_eval(_rel_pos_abdomen) 
            if type(_rel_pos_abdomen) in [list,tuple]:
                if len(_rel_pos_abdomen)==2:
                    if type(_rel_pos_abdomen[0])==int and type(_rel_pos_abdomen[1])==int:
                        coords = _rel_pos_abdomen
        
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
        self.widget_figtext_pos.set_c(color="k")

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click_canvas)
        self.fig.canvas.draw()  # update the display manually
        pass

# %% 
class LabelInspectorClass:
    def __init__(self, dir_extraction = "extracted"):
        self.set_dir_extracted(dir_extraction)
        self.df_startup()
        self.df_analyze()
        self.show_info()
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
    
    
    
    def df_startup(self):
        """Setup for the dataframe:
            
        Defines file names, trys to load the label file or creats it, if not yet exists."""
        fname_labels = "Labels"
        
        dir_extracted = self._dir_extracted
        self._df_fname_labels_csv =  os.path.join(dir_extracted, "{}_csv.csv".format(fname_labels))
        self._df_fname_labels_scsv = os.path.join(dir_extracted, "{}_scsv.csv".format(fname_labels))
        
        # Check if [comma] separated value files exists : otherwise exception
        self.check_isfile(self._df_fname_labels_csv)
        
        # read csv file
        self._df_labels = pd.read_csv(self._df_fname_labels_csv, index_col=0)
        self._df_labels.dropna(inplace=True) # get rid of all NaN value Rows
        
        self._df_size = len(self._df_labels)
        print("df length: {}".format(self._df_size))
        pass
    
    def df_analyze(self):
        df = self._df_labels
        # self.df_beeYes = df.where(df["has_bee"] > 0)
        # self.df_beeYes.dropna(inplace=True)
        # self.df_beeNo =  df.where(df["has_bee"] <= 0)
        # self.df_beeNo.dropna(inplace=True)
        
        self.len_bee = {}
        
        df_bY = df.loc[df["has_bee"] > 0]           # bee yes
        df_bYS = df_bY.loc[df_bY["img_sharp"] > 0]  # sharp
        df_bYF = df_bY.loc[df_bY["img_sharp"] <= 0] # fuzzy
        
        self.len_bee["bY"] =    len(df_bY)  # number of bee imgs
        self.len_bee["bYS"] =   len(df_bYS) # bee yes: sharp
        self.len_bee["bYF"] =   len(df_bYF) # bee yes: fuzzy
        
        # all rows with abdomen pos definded have a "(" inside that column: Regex: "\("
        df_bYSaY = df_bYS.loc[df_bYS['rel_pos_abdomen'].str.contains('\(') ]
        df_bYFaY = df_bYF.loc[df_bYF['rel_pos_abdomen'].str.contains('\(') ]
        # df_bYSaY = df_bYS.loc[ ( df_bYS["rel_pos_abdomen"] ) not in [""," ",str(None)] ]  # sharp bee imgs WITH abdomen pos
        # df_bYFaY = df_bYF.loc[ ( df_bYF["rel_pos_abdomen"] ) not in [""," ",str(None)] ]  # fuzzy bee imgs WITH abdomen pos
        
        self.len_bee["bYSaY"] =   len(df_bYSaY)                 # number of sharp bee: with abdomen
        self.len_bee["bYSaN"] =   len(df_bYS) - len(df_bYSaY)   # number of sharp bee: withOUT abdomen
        self.len_bee["bYFaY"] =   len(df_bYFaY)                 # number of fuzzy bee: with abdomen
        self.len_bee["bYFaN"] =   len(df_bYF) - len(df_bYFaY)   # number of fuzzy bee: withOUT abdomen
        
        df_bN = df.loc[df["has_bee"] <= 0]          # bee noo
        df_bNS = df_bN.loc[df_bN["img_sharp"] > 0]  # sharp
        df_bNF = df_bN.loc[df_bN["img_sharp"] <= 0] # fuzzy
        
        self.len_bee["bN"] =    len(df_bN)  # number of empty imgs
        self.len_bee["bNS"] =   len(df_bNS) # bee no: sharp
        self.len_bee["bNF"] =   len(df_bNF) # bee no: fuzzy
        
        # df_label_match = df_labels.loc[df_labels["parent_fname"] == row_empty["src_fname"]]
        pass
    
    def show_info(self):
        dct = self.len_bee
        
        print("+ BEE: {} ({} vs {})".format(dct["bY"], dct["bYS"], dct["bYF"]) )
        print("- bee: {} ({} vs {})".format(dct["bN"], dct["bNS"], dct["bNF"]) )
        
        wedge_bYNSF = [dct["bYS"], dct["bYF"], dct["bNF"], dct["bNS"]]
        label_bYNSF = ["is bee\n(sharp)","is bee\n(fuzzy)","no bee\n(fuzzy)","no bee\n(sharp)"]
        color_bYNSF = ["dodgerblue", "royalblue","gold", "yellow"]
        title_bYNSF = "Labeling of Focus imgs:\nBee ({}) vs Empty ({})".format(dct["bY"], dct["bN"])
        
        wedge_bYSFaYN = [dct["bYSaY"], dct["bYSaN"], dct["bYFaN"], dct["bYFaY"]]
        label_bYSFaYN = ["sharp\n(pos:Y)","sharp\n(pos:N)","fuzzy\n(pos:N)","fuzzy\n(pos:Y)"]
        color_bYSFaYN = ["orangered", "coral","aquamarine", "turquoise"]
        title_bYSFaYN = "Abdomen position given in bee imgs:\nSharp ({}) vs Fuzzy({})".format(dct["bYS"], dct["bYF"])
        
        # plt.close('all')
        
        # make new figure
        self.fig, self.ax = plt.subplots(1,2)
        
        self.ax[0].pie(wedge_bYNSF, labels=label_bYNSF, autopct='%1.1f%%', colors=color_bYNSF, startangle=90)
        self.ax[0].set_title(title_bYNSF)
        self.ax[1].pie(wedge_bYSFaYN, labels=label_bYSFaYN, autopct='%1.1f%%', colors=color_bYSFaYN, startangle=90)
        self.ax[1].set_title(title_bYSFaYN)
        
        self.fig.tight_layout()
        self.fig.show()
        pass
    

# %% 
        
class FindEmptyClass:
    def __init__(self, dir_extraction, sort_list=["contours_raw","contours_valid"], over_write=False):
        self._storing_counter = 0
        self._sort_list=sort_list
        
        self.set_dir_extracted(dir_extraction)
        self.df_startup(over_write)
        
        # self._new_fig_()
        # self.df_FE_fetch_next()
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
        self._dir_extracted = self.check_isdir(path)
        pass
    
    def analyze(self):
        plt.close("all")
        df = self._df_FE
        
        df_empty = df.where(df["empty"] > 0)
        df_empty.dropna(inplace=True)
        df_popul =  df.where(df["empty"] <= 0)
        df_popul.dropna(inplace=True)
        
        wedge_sizes = [len(df_popul), len(df_empty), len(self.list_candidates)]
        labels = ["populated","empty","unknown"]
        colors = ["dodgerblue", "gold", "gray"]
        
        # plt.close('all')
        
        # make new figure
        self.fig_pie, self.ax_pie = plt.subplots()
        
        self.ax_pie.pie(wedge_sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        txt = "Population in Source images ({})\nPopulated: {}, Empty: {}, Unknown: {}".format(len(df),wedge_sizes[0],wedge_sizes[1],wedge_sizes[2])
        self.ax_pie.set_title(txt)
        
        self.fig_pie.tight_layout()
        self.fig_pie.show()
        
        pass
    
    def focus_label_assist(self):
        print("This will set the focus images of empty src images to contain no bees and being sharp!")
        fname_labels = "Labels"
        dir_extracted = self._dir_extracted
        df_fname_labels_csv =  os.path.join(dir_extracted, "{}_csv.csv".format(fname_labels))
        df_fname_labels_scsv = os.path.join(dir_extracted, "{}_scsv.csv".format(fname_labels))
        
        # Check if [comma] separated value files exists : otherwise exception
        self.check_isfile(df_fname_labels_csv)
        
        # read csv file
        df_labels = pd.read_csv(df_fname_labels_csv, index_col=0)
        
        # save a backup version of the labels
        df_fname_labels_backup_csv =  os.path.join(dir_extracted, "{}_backup_csv.csv".format(fname_labels))
        df_fname_labels_backup_scsv = os.path.join(dir_extracted, "{}_backup_scsv.csv".format(fname_labels))
        df_labels.to_csv(df_fname_labels_backup_csv,  sep=",")
        df_labels.to_csv(df_fname_labels_backup_scsv, sep=";")
        
        df_empty = self._df_FE.where(self._df_FE["empty"] > 0)
        df_empty.dropna(inplace=True)
        
        match_counter = 0
        focus_counter = 0
        # go through all rows of the empty df
        for i in tqdm(range( len(df_empty) ), desc="Help with empty images"):
            row_empty = df_empty.iloc[i]
            
            # find all matching rows in Label df for the index of an empty image
            df_label_match = df_labels.loc[df_labels["parent_fname"] == row_empty["src_fname"]]
            
            # df_label_match = df_labels.loc[df_labels["parent_fname"] == "20200803_052936_image0005_0_s.png"]
            
            matchFlag=0
            # go through all Indexes that matched and set their bee pos as empty
            for idx in df_label_match.index.to_list():
                matchFlag += 1
                # print(idx)
                df_labels.at[idx, "has_bee"] = 0
                df_labels.at[idx, "img_sharp"] = 1
                df_labels.at[idx, "rel_pos_abdomen"] = str(None)
                pass
            match_counter += (matchFlag>0)
            focus_counter += matchFlag
            pass #-------------------------------------------------------------
        print("{} empty src images had {} focus images.".format(match_counter, focus_counter))
        
        # write to a DIFFERENT output file
        # fname_labels = "Labels_helped"
        # df_fname_labels_csv =  os.path.join(dir_extracted, "{}_csv.csv".format(fname_labels))
        # df_fname_labels_scsv = os.path.join(dir_extracted, "{}_scsv.csv".format(fname_labels))
        
        df_labels.to_csv(df_fname_labels_csv,  sep=",")
        df_labels.to_csv(df_fname_labels_scsv, sep=";")
        
        print("Stored {}".format(df_fname_labels_csv))
        pass
    
    def labeling(self):
        if len(self.list_candidates) > 0:
            self._new_fig_()
            self.df_FE_fetch_next()
        else:
            print("No more unlabeled")
        pass
    
    def df_startup(self,over_write):
        fname_parent = "Parent"
        fname_FE = "Find_Empty"
        
        path_dir = self._dir_extracted
        self._df_fname_parent_csv =  os.path.join(path_dir, "{}_csv.csv".format(fname_parent))
        self._df_fname_FE_csv =  os.path.join(path_dir, "{}_csv.csv".format(fname_FE))
        self._df_fname_FE_scsv = os.path.join(path_dir, "{}_scsv.csv".format(fname_FE))
        
        # check if the [comma] separated value file exists
        self.check_isfile(self._df_fname_parent_csv) # parent file MUST exist!
        self._df_parent = pd.read_csv(self._df_fname_parent_csv, index_col=0)
        if type(self._sort_list) in [list,tuple]:
            self._df_parent.sort_values(self._sort_list, inplace=True)
            print("Sorted by: {}".format(self._sort_list))
        else:
            print("No additional Sorting.")
        
        
        # We will load from the [comma] separated value files
        f_FE_exists =  os.path.isfile(self._df_fname_FE_csv)
        
        if f_FE_exists and not over_write:
            self._df_FE = pd.read_csv(self._df_fname_FE_csv, index_col=0)
        else:
            cols_FE =  ["src_fname",
                        "src_fpath",
                        "roi_fpath",
                        "contours_raw",
                        "contours_valid",
                        "empty"]
            self._df_FE = pd.DataFrame(columns=cols_FE)
            pass
        
        
        list_candidates = self._df_parent.index.values.tolist()
        list_empty = self._df_FE.index.values.tolist()
        
        # get a list of all not yet check imgs
        self.list_candidates = [x for x in list_candidates if x not in list_empty]
        pass
    
    def df_FE_fetch_next(self):
        if len(self.list_candidates) > 0:
            self._rowIdx = self.list_candidates[0]
        else:
            self.df_store()
            self.fig.close()
            raise Exception("'self.list_candidates' is empty!")
        
        self._row_parent = self._df_parent.loc[self._rowIdx]
        print(self._rowIdx)
        # print(self._row_parent["src_fpath"])
        # print(self._row_parent["roi_fpath"])
        
        self.update_fig()
        pass
    
    def df_FE_set_empty(self,emptyFlag:int):
        # write result into FE dataframe
        data_slice = {"src_fname":       self._row_parent["src_fname"],
                      "src_fpath":       self._row_parent["src_fpath"],
                      "roi_fpath":       self._row_parent["roi_fpath"],
                      "contours_raw":    self._row_parent["contours_raw"],
                      "contours_valid":  self._row_parent["contours_valid"],
                      "empty":           emptyFlag}
        self._df_FE.at[self._rowIdx] = pd.Series(data_slice)
        
        # remove idx from candidate list
        self.list_candidates.remove(self._rowIdx)
        
        self._storing_counter += 1 # inc counter
        if (self._storing_counter >= 10): self.df_store()
        
        # fetch next idx
        self.df_FE_fetch_next()
        pass
    
    def df_FE_drop_last(self,num:int):
        # will drop the last num rows
        self._df_FE.drop(self._df_FE.tail(num).index,inplace=True)
        print("Dropped the last {} rows. Restarting...".format(num))
        
        # restart
        self.df_store()
        self.df_startup(False)
        
        # fetch next idx
        self.df_FE_fetch_next()
        pass
    
    def df_store(self):
        self._storing_counter = 0
        
        self._df_FE.to_csv(self._df_fname_FE_csv,  sep=",")
        self._df_FE.to_csv(self._df_fname_FE_scsv, sep=";")
        
        print("Stored {}".format(self._df_fname_FE_csv))
        pass
    
    
    # FIGURE functions
    def _new_fig_(self):
        
        # make new figure
        self.fig = plt.figure()
        
        
        self.widget_figtext_Title = \
            plt.figtext(0.5, 0.99, "[0]: Set to Empty,   [,],[1-9]: Set to Populated", 
                        va ="top", ha ="center", wrap = True, fontsize = 10) 
            
        # axes on top to show the ROI image
        self.ax_roi = plt.axes([0, 0*0.48, 1, 0.48]) #[left,bottom,width,height]
        self.ax_roi.axis("off")
        # self.ax_roi.title.set_text("")
        
        # axes on bottom left to show the focus image
        self.ax_orig = plt.axes([0, 1*0.48, 1, 0.48]) #[left,bottom,width,height]
        self.ax_orig.axis("off")
        # self.ax_orig.title.set_text("")
        
        
        thismanager = plt.get_current_fig_manager()
        thismanager.resize(400,630)
        # thismanager.window.setGeometry(self.posX, self.posY, self.w, self.h)
        
        
        
        # save to csv on figure closing
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # add key event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.draw()
        pass
    
    def update_fig(self):
        self.ax_roi.clear()
        self.ax_roi.axis("off")
        self.ax_roi.imread(self._row_parent.roi_fpath)
        
        self.ax_orig.clear()
        self.ax_orig.axis("off")
        self.ax_orig.imread(self._row_parent.src_fpath)
        
        self.fig.canvas.draw()  # update the display manually
        pass
    
    def on_close(self,event):
        # save to csv on figure closing
        self.df_store()
        print("Figure closed.")
        pass
    
    def on_key(self,event):
        self.keypressed = event.key
        
        if event.key == "escape":
            print('you pressed', event.key)
            self.df_store()
        elif event.key == "delete":
            print('you pressed', event.key)
            self.df_FE_drop_last(5)
        elif event.key in ["0"]:
            print('you pressed', event.key)
            self.df_FE_set_empty(1)
            
        elif event.key in ["1","2","3","4","5","6","7","8","9",","]:
            print('you pressed', event.key)
            self.df_FE_set_empty(0)
        
        pass
    
    def update_fig(self):
        img_roi =  cv2.cvtColor( cv2.imread(self._row_parent.roi_fpath), cv2.COLOR_BGR2RGB)
        img_orig = cv2.cvtColor( cv2.imread(self._row_parent.src_fpath), cv2.COLOR_BGR2RGB)
        self.ax_roi.clear()
        self.ax_roi.axis("off")
        self.ax_roi.imshow(img_roi)
        
        self.ax_orig.clear()
        self.ax_orig.axis("off")
        self.ax_orig.imshow(img_orig)
        
        self.fig.canvas.draw()  # update the display manually
        pass


# %% 


# %% 

if __name__== "__main__":
    print("## Calling main function.)\n")
    
    print("cv2.version = {}".format(cv2.__version__))
    print("numpy.version = {}".format(np.__version__))
    print("matplotlib.version = {}".format(mpl.__version__))
    print("pandas.version = {}".format(pd.__version__))
    print()
    
    
    # Window Cleanup
    cv2.destroyAllWindows()
    plt.close('all')
    
    TEST = 2
    
    # %%
    if TEST == 1:
        cv2.destroyAllWindows()
        plt.close('all')
        print("start labeling unlabeled focus imgs...")

        myHLH = ManualLabelHelper()
        my_df = myHLH._df_labels
        pass

    # %%
    if TEST == 2:
        print("Show which of you labeled images has bees or is empty")
        myLIC = LabelInspectorClass()
        # my_df = myLIC._df_labels
        # df_beeY = myLIC.df_beeYes
        # df_beeN = myLIC.df_beeNo
        
        pass

    # %%
    if TEST == 3:
        print("Label/Show which of the source images is empty/populated")
        plt.close('all')
        # path_src = "D:\\ECM_PROJECT\\bee_images_small"
        path_extr = "extracted"
        
        myFEC = FindEmptyClass(path_extr,sort_list=None, over_write=False)
        df_parent = myFEC._df_parent
        df_EF = myFEC._df_FE
        myList = myFEC.list_candidates
        
        # myFEC.labeling()
        myFEC.analyze()
        pass
    
    # %%
    if TEST==4:
        print("Write all focus images in the lavel-csv, that come from a confirmed empty src image, to have no bees")
        plt.close('all')
        # path_src = "D:\\ECM_PROJECT\\bee_images_small"
        path_extr = "extracted"
        
        myFEC = FindEmptyClass(path_extr,sort_list=None, over_write=False)
        
        myFEC.focus_label_assist()
    
    
    
    
    
    
    
    
    