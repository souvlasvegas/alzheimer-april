# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:49:15 2022

@author: m_ant
"""

import csv
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
import numpy as np

#This script takes all csv files and create train-test splits for each one of them

files = filedialog.askopenfilenames(title="Select all csv's to be merged ")
copy_files=files

directory = filedialog.askdirectory(title="choose directory for the merged csv to be created")
#endfile = directory +"/"+ name+".csv"

for file in files:
    df=pd.DataFrame()
    for copy in copy_files:
        if copy!=file:
            copy_df=pd.read_csv(copy)
            df=pd.concat([df,copy_df], axis=0, ignore_index=True)
            print("antexw")
    filename = Path(file).name
    split_tup = os.path.splitext(filename)
    file_name = split_tup[0]
    name=file_name.split('_')[0]
    try:
        os.makedirs(directory+"/train_test_"+name)
    except FileExistsError:
        # directory already exists
        pass
    df.to_csv(directory+"/train_test_"+name+"/train_"+name+".csv", index=False)
    file_df=pd.read_csv(file)
    file_df.to_csv(directory+"/train_test_"+name+"/test_"+name+".csv", index=False)
    


