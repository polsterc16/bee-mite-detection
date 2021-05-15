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

# %% FUNCTIONS

class ManualLabelHelper:
    def __init__(self, dir_extraction = "extracted",
                 file_focus =       "Focus_csv.csv", 
                 dir_focus_img =    "focus", 
                 dir_roi_img =      "roi"):
        self.index = 0
        self.index_goto = 0
        self.index_max = 0
        
        self.set_dir_extracted(dir_extraction)
        self.set_path_file_focus(file_focus)
        self.set_dir_focus_img(dir_focus_img)
        
        
        self._path_file_focus = self.check_path(file_focus)
        self._path_focus_img =  self.check_path(dir_focus_img)
        self._path_roi_img =    self.check_path(dir_roi_img)
        self._check_for_labels_file_(path_labels_file)
        self._load_from_csv_()
        
        
        self._new_fig_()
        self.nav_goto_by_index()
        
        
        self.print_keybinds()
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
    
    def set_dir_focus_img(self,path):
        """Sets the dir path to the focus_img folder. 
        
        Is assumed to be relative to the Extraction folder, unless an abs_path is specified."""
        if os.path.isabs(path):
            self._dir_focus_img = self.check_isdir( path )
        else:
            self._dir_focus_img = self.check_isdir( os.path.join(self._dir_extracted, path) )
        pass
    
    def set_dir_roi_img(self,path):
        """Sets the dir path to the roi_img folder. 
        
        Is assumed to be relative to the Extraction folder, unless an abs_path is specified."""
        if os.path.isabs(path):
            self._dir_roi_img = self.check_isdir( path )
        else:
            self._dir_roi_img = self.check_isdir( os.path.join(self._dir_extracted, path) )
        pass
    
    def set_path_file_focus(self,path:str):
        m = path.lower()
        if not m.endswith(".csv"):
            raise Exception("'path_file_focus' must end in '.csv'!")
        
        if os.path.isabs(path):
            self._path_file_focus = self.check_isfile( path )
        else:
            self._path_file_focus = self.check_isfile( os.path.join(self._dir_extracted, path) )
        pass
    
    def print_keybinds(self):
        print("[ESCAPE] Save to CSV")
        print("[SPACE]  Write & Next")
        print("[B][<]   Previous")
        print("[N][>]   Next")
        print("[Y]      Toggle Box 1")
        print("[X]      Toggle Box 2")
        print("[C]      Toggle Box 3")
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
        f_labels_exists = os.path.isfile(self._df_fname_parent_csv)
        
        if f_labels_exists: # if exists: load
            self._df_labels = pd.read_csv(self._df_fname_labels_csv, index_col=0)
        else: # otherwise make new from focus file
            cols_label =   ["has_bee",
                            "img_sharp",
                            "rel_pos_abdomen"]
            # load the focus file as basis
            self._df_labels = pd.read_csv(self._path_file_focus, index_col=0)
            # append the columns with NaN as default values
            for col in cols_label:
                self._df_labels[col] = np.nan
            # save the new file
            self.df_store()
        pass
    
    def df_store(self):
        """Saves the dataframe"""
        self._df_labels.to_csv(self._df_fname_labels_csv,  sep=",")
        self._df_labels.to_csv(self._df_fname_labels_scsv, sep=";")
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
        
        self.fig_shape = ( int(self.fig.get_figwidth() * self.fig.dpi),
                           int(self.fig.get_figheight() * self.fig.dpi) )
        
        self.figim = self.fig.figimage( np.random.random((100,100)) )
        self.img_extr_temp = np.random.random((100,100))
        
        self.widget_figtext_left = \
            plt.figtext(0.25, 0.99, "title", 
                        va ="top", ha ="center", wrap = True, fontsize = 10) 
            
        
        # axes on right top to show the ROI in the original image
        self.ax_roi = plt.axes([0.5, 0.5, 0.5, 0.45])
        self.ax_roi.axis("off")
        self.ax_roi.title.set_text("title")
        
        # axes for checkbox
        self.ax_checkbox = plt.axes([0.525, 0.3, 0.2, 0.185])
        self.ax_checkbox_labels = ["y is bee", "x mostly visible", "c has mite"]
        self.widget_check = mpl.widgets.CheckButtons(self.ax_checkbox, 
                                                     self.ax_checkbox_labels)
        
        #axes for "Write and Next" button
        self.ax_button_WnN = plt.axes([0.525, 0.2, 0.1, 0.075])
        self.widget_button_WnN = \
            mpl.widgets.Button(self.ax_button_WnN, "W & N")
        self.widget_button_WnN.on_clicked(self.on_click_WnN)
        
        #axes for "writeToDF" button
        self.ax_button_writeToDF = plt.axes([0.525+0.12, 0.2, 0.15, 0.075])
        self.widget_button_writeToDF = \
            mpl.widgets.Button(self.ax_button_writeToDF, "Write to DF")
        self.widget_button_writeToDF.on_clicked(self.on_click_write_to_df)
        
        #axes for "restore from df" button
        self.ax_button_restoreFromDF = plt.axes([0.525+0.32, 0.2, 0.15, 0.075])
        self.widget_button_restoreFromDF = \
            mpl.widgets.Button(self.ax_button_restoreFromDF, "Restore from DF")
        self.widget_button_restoreFromDF.on_clicked(self.on_click_restore_from_df)
        
        # ax for save csv
        self.ax_button_saveCSV = plt.axes([0.525+0.375, 0.1, 0.1, 0.075])
        self.widget_button_saveCSV =  mpl.widgets.Button(self.ax_button_saveCSV,"save CSV")
        self.widget_button_saveCSV.on_clicked(self.on_click_saveCSV)
        
        # axes for navigation buttons
        self.ax_button_j2_first = plt.axes([0.525, 0.0, 0.1, 0.075])
        self.ax_button_j2_prev =  plt.axes([0.525+0.125, 0.0, 0.1, 0.075])
        self.ax_button_j2_next =  plt.axes([0.525+0.25, 0.0, 0.1, 0.075])
        self.ax_button_j2_last =  plt.axes([0.525+0.375, 0.0, 0.1, 0.075])
        
        self.ax_button_j2_select = plt.axes([0.525+0.25, 0.1, 0.1, 0.075])
        
        # make buttons
        self.widget_button_j2_first =  mpl.widgets.Button(self.ax_button_j2_first,  "First")
        self.widget_button_j2_prev =   mpl.widgets.Button(self.ax_button_j2_prev,   "Prev")
        self.widget_button_j2_next =   mpl.widgets.Button(self.ax_button_j2_next,   "Next")
        self.widget_button_j2_last =   mpl.widgets.Button(self.ax_button_j2_last,   "Last")
        self.widget_button_j2_select = mpl.widgets.Button(self.ax_button_j2_select, "To (x)")
        
        #link to on-klick functions
        self.widget_button_j2_first.on_clicked(self.on_click_j2_first)
        self.widget_button_j2_prev.on_clicked(self.on_click_j2_prev)
        self.widget_button_j2_next.on_clicked(self.on_click_j2_next)
        self.widget_button_j2_last.on_clicked(self.on_click_j2_last)
        self.widget_button_j2_select.on_clicked(self.on_click_j2_select)
        
        # make textbox widget
        self.ax_textbox_j2_select = plt.axes([0.525+0.125, 0.1, 0.1, 0.075])
        self.widget_textbox_j2_select = \
            mpl.widgets.TextBox(self.ax_textbox_j2_select, "To:","1")
        self.widget_textbox_j2_select.on_submit(self.txt_submit)
        
        self.widget_figtext = \
            plt.figtext(0.75+0.01, 0.45, 
                        "is bee: X\nmostly visible: X\nhas mite: X", 
                        va ="top", ha ="left", wrap = True, fontsize = 10, 
                        bbox ={'facecolor':'grey', 'alpha':0.1, 'pad':5}) 
        
        
        # save to csv on figure closing
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # add key event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        pass
    
    
    def func_save_CSV(self):
        # save to csv
        self.df.to_csv(self.path_MAIN+"Extracted_2.csv",sep=";")
        
        # note where we last where
        path = self.path_MAIN+"last_index.txt"
        with open(path, "w") as f:
            f.write(str(self.index))
        print("index:",self.index) # display current index
        pass
    
    def on_close(self,event):
        # save to csv on figure closing
        self.func_save_CSV()
        print("Figure closed.")
        pass
    
    def on_click_saveCSV(self, event):
        self.func_save_CSV()
        pass
    
    
    def on_click_write_to_df(self, event):
        state = self.widget_check.get_status()
        self.df.loc[self.index,"isBee"] = state[0]
        self.df.loc[self.index,"mostlyVisible"] = state[1]
        self.df.loc[self.index,"hasMites"] = state[2]
        
        self.index_goto = self.index
        self.nav_goto_by_index()
        pass
    
    def on_click_WnN(self, event):
        state = self.widget_check.get_status()
        self.df.loc[self.index,"isBee"] = state[0]
        self.df.loc[self.index,"mostlyVisible"] = state[1]
        self.df.loc[self.index,"hasMites"] = state[2]
        
        self.index_goto = min( self.index+1, self.index_max )
        self.nav_goto_by_index()
        pass
    
    def on_click_restore_from_df(self, event):
        self.index_goto = self.index
        self.nav_goto_by_index()
        pass
    
    def on_click_j2_first(self, event):
        self.index_goto = 0
        self.nav_goto_by_index()
        pass
    
    def on_click_j2_last(self, event):
        self.index_goto = self.index_max
        self.nav_goto_by_index()
        pass
    
    def on_click_j2_prev(self, event):
        self.index_goto = max( 0, self.index-1 )
        self.nav_goto_by_index()
        pass
    
    def on_click_j2_next(self, event):
        self.index_goto = min( self.index+1, self.index_max )
        self.nav_goto_by_index()
        pass
    
    def on_click_j2_select(self, event):
        self.nav_goto_by_index()
        pass
    
    def update_button_label_j2_select(self, newLabel):
        # function just for convenience
        self.widget_button_j2_select.label.set_text( newLabel )
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
        self.widget_textbox_j2_select.set_val( str(self.index_goto) )
        # also the button label updates to the jump-to index
        self.update_button_label_j2_select( "To "+str(self.index_goto) )
        
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
        elif event.key in ["b","left"]:
            self.on_click_j2_prev(None)
        elif event.key in ["n","right"]:
            self.on_click_j2_next(None)
        elif (event.key == "y"):
            self.widget_check.set_active(0)
        elif (event.key == "x"):
            self.widget_check.set_active(1)
        elif (event.key == "c"):
            self.widget_check.set_active(2)
        pass
    
    def nav_goto_by_index(self):
        self.index = self.index_goto
        self.df_row = self.df.loc[self.index,:]
        
        img_extr_path = self.path_MAIN + self.path_imgs + self.df_row["img_extr"]
        img_roi_path = self.path_MAIN + self.path_showroi + self.df_row["img_overlay"]
        
        try:
            self.img_extr = plt.imread(img_extr_path)
            self.img_roi = plt.imread(img_roi_path)
        except:
            print("Exception at index ",self.index,". (Reading image)")
            raise Exception("ERROR reading of images.") 
        
        self.txt_submit("")
        self.update_fig()
        pass
    
    
    def update_fig(self):
        self.fig_shape = ( int(self.fig.get_figwidth() * self.fig.dpi),
                           int(self.fig.get_figheight() * self.fig.dpi) )
        
        # self.ax_img.clear()
        # self.ax_img.axis("off")
        # self.ax_img.imshow(self.img_extr)
        title = str(self.index) + ": " + str(self.df_row["img_extr"])
        # self.ax_img.title.set_text(title)
        self.widget_figtext_left.set_text(title)
        
        scale = 0.5
        shape = self.img_extr.shape
        dim = ( int(shape[1]*scale), int(shape[0]*scale) )
        self.img_extr_temp = cv2.resize(self.img_extr, dsize=dim,
                                     interpolation = cv2.INTER_AREA )
        
        self.figim.set_data(self.img_extr_temp)
        # print("self.fig_shape",self.fig_shape)
        # print("self.img_extr_temp.shape",self.img_extr_temp.shape)
        self.figim.ox = int( self.fig_shape[0]/4 - self.img_extr_temp.shape[1]/2 )
        self.figim.oy = int( self.fig_shape[1]/2 - self.img_extr_temp.shape[0]/2 )
        # print("ox,oy",str((self.figim.ox,self.figim.oy)))
        
        self.ax_roi.clear()
        self.ax_roi.axis("off")
        self.ax_roi.imshow(self.img_roi)
        self.ax_roi.title.set_text(self.df_row["img_overlay"])
        
        index_list = ["isBee","mostlyVisible","hasMites"]
        check_status = [self.df_row[k] for k in index_list]
        figtext_list = ["{}: {}".format(k,self.df_row[k]) for k in index_list ]
        
        # if (check_status[0]): self.widget_check.set_active(0)
        # if (check_status[1]): self.widget_check.set_active(1)
        # if (check_status[2]): self.widget_check.set_active(2)
        self.ax_checkbox.clear()
        self.widget_check = mpl.widgets.CheckButtons(self.ax_checkbox, 
                                                     self.ax_checkbox_labels,
                                                     check_status)
        
        
        figtext = figtext_list[0]+"\n"+figtext_list[1]+"\n"+figtext_list[2]
        self.widget_figtext.set_text(figtext)
    
        self.fig.canvas.draw()  # update the display manually
        pass

# %% 


# %% 


# %% 
test = ManualLabelHelper()
# test._new_fig_()