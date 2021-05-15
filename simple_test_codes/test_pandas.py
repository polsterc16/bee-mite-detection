# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:51:50 2021

@author: Admin
"""

import numpy as np
import pandas as pd

print("pandas version: {}\n".format(str(pd.__version__)))

TEST = 3

#%% Pandas as pd
# https://www.w3schools.com/python/pandas/pandas_getting_started.asp

if TEST ==1:
    mydataset = {
      'cars': ["BMW", "Volvo", "Ford"],
      'passings': [3, 7, 2]
    }
    
    myvar = pd.DataFrame(mydataset)
    
    print(myvar) 
    
    #%% What is a Series?
    # https://www.w3schools.com/python/pandas/pandas_series.asp
    
    a = [1, 7, 2]
    
    myvar = pd.Series(a)
    
    print(myvar)
    
    # Labels
    
    print(myvar[0])
    
    #%% 
    # Create Labels
    
    a = [1, 7, 2]
    
    myvar = pd.Series(a, index = ["x", "y", "z"])
    
    print(myvar)
    
    print(myvar["y"])
    
    #%% Key/Value Objects as Series
    
    calories = {"day1": 420, "day2": 380, "day3": 390}
    
    myvar = pd.Series(calories)
    
    print(myvar)
    
    #%% DataFrames
    
    data = {
      "calories": [420, 380, 390],
      "duration": [50, 40, 45]
    }
    
    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    
    print(df) 
    
    #refer to the row index:
    print(df.loc[0])
    
    #use a list of indexes:
    print(df.loc[[0, 1]])
    
    #%% Named Indexes
    
    data = {
      "calories": [420, 380, 390],
      "duration": [50, 40, 45]
    }
    
    df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
    
    print(df) 
    
    #refer to the named index:
    print(df.loc["day2"])
    
    #%% Info About the Data
    
    # https://www.w3schools.com/python/pandas/pandas_analyzing.asp
    
    
    data = {
      "calories": [420, 380, 390],
      "duration": [50, 40, 45]
    }
    
    df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
    
    print(df.info()) 
    
    
    #%% MY TEST
    
    data = {
      "src_name":       ["420",   "380",    "390"],
      "src_path":       ["50",    "40",     "45"],
      "debug_path":       ["50",    "40",     "45"],
      "contours_raw":   [2,     3,      6],
      "contours_valid": [0,     1,      3],
      "children":       [0,     1,      3],
      "children_name":  [[],    ["0",], ["0","1","2"]]
    }
    
    #load data into a DataFrame object:
    df = pd.DataFrame(data)
    print(df.info())
    
    df.loc[6] = {"src_name": "420",
                 "src_path": "420",
                 "debug_path": "420",
                 "contours_raw": 17,
                 "contours_valid": 12,
                 "children": 12,
                 "children_name": ["a","b","c","d","e"]}
    
    #%%
    columns = ["src_name",
               "src_path",
               "debug_path",
               "contours_raw",
               "contours_valid",
               "children_name"
               ]
    
    df = pd.DataFrame(columns=columns)
    
    
    print(df.info())


#%% test appending

if TEST ==2:
    cols_parent =  ["src_index",
                    "src_fname",
                    "src_fpath",
                    "roi_fpath",
                    "contours_raw",
                    "contours_valid",
                    "children_names"]
    
    cols_focus =   ["index",
                    "fname",
                    "fpath",
                    "parent_index",
                    "parent_fname",
                    "parent_fpath",
                    "roi_fpath",
                    "pos_center",
                    "pos_anchor",
                    "minAreaRect"
                    ]
    
    df_parent = pd.DataFrame(columns=cols_parent)
    df_focus =  pd.DataFrame(columns=cols_focus)
    
    
    focus =    {"index":0,
                "fname":"asdasd",
                "fpath":"asdasd",
                "parent_index":0,
                "parent_fname":"asdasd",
                "parent_fpath":"wqewedas",
                "roi_fpath":"sdfsdfs",
                "pos_center":(1,2),
                "pos_anchor":(3,6),
                "minAreaRect":((1,1),(2,2),10)
                }
    # df_focus.loc[0] = focus
    new_row = pd.Series(focus)
    df_focus=df_focus.append(new_row,ignore_index=True)
    
    print(df_focus.info())
    
#%% test adding columns

if TEST ==3:
    
    cols_focus =   ["index",
                    "fname",
                    "fpath",
                    "parent_index",
                    "parent_fname",
                    "parent_fpath",
                    "roi_fpath",
                    "pos_center",
                    "pos_anchor",
                    "minAreaRect"
                    ]
    
    df_focus =  pd.DataFrame(columns=cols_focus)
    
    
    focus =    {"index":0,
                "fname":"asdasd",
                "fpath":"asdasd",
                "parent_index":0,
                "parent_fname":"asdasd",
                "parent_fpath":"wqewedas",
                "roi_fpath":"sdfsdfs",
                "pos_center":(1,2),
                "pos_anchor":(3,6),
                "minAreaRect":((1,1),(2,2),10)
                }
    # populate df real quick
    for i in range(5):
        df_focus.loc[i] = focus
    print(df_focus.info())
    
    # df_focus.to_csv("test_df_temp.csv")
    #%%
    # df_focus=None
    # df_focus = pd.read_csv("test_df_temp.csv", index_col=0)
    # print(df_focus.info())
    
    
    cols_label =   [("has_bee",np.nan),
                    ("img_sharp",np.nan),
                    ("rel_pos_abdomen"," ")]
    for col in cols_label:
        df_focus[col[0]] = col[1]
    print(df_focus.info())
    
    for i in range(3):
        df_focus["has_bee"].iat[i]=  1
        df_focus["img_sharp"].iat[i]=  0
        df_focus["rel_pos_abdomen"].iat[i]= str((1,2))
        pass
    #%%
    import pandas as pd
    import numpy as np
    
    data = {'Column_A': [1,2,3,4,5,np.nan,6,7,np.nan],
            'Column_B': [11,22,33,44,55,66,77,88,99],
            'Column_C': ['a','b',np.nan,np.nan,'c','d','e',np.nan,'f'],
            'Column_D': ['aa','bb','cc','dd','ee','ff','gg','hh','ii']
            }
    
    df = pd.DataFrame(data,columns=['Column_A','Column_B','Column_C','Column_D'])
    
    # nan_values = df[df.columns[df.isna().any()]]
    
    # print (nan_values)
    
    index = df['Column_A'].index[df['Column_A'].apply(np.isnan)]
    print(index)







































