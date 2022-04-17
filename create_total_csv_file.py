# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:35:27 2022

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
#use it if you are going to make the train-test split inside the code

files = filedialog.askopenfilenames(title="Select all csv's to be merged ")

directory = filedialog.askdirectory(title="choose directory for the merged csv to be created")
#endfile = directory +"/"+ name+".csv"
df=pd.DataFrame()
for file in files:
    file_df=pd.read_csv(file)
    df=pd.concat([df,file_df], axis=0, ignore_index=True)
    print("antexw")
filename = Path(file).name
split_tup = os.path.splitext(filename)
file_name = split_tup[0]
name=file_name.split('_')[0]
df.to_csv(directory+"/all_epochs_total_features.csv", index=False)
