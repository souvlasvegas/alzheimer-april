# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:52:52 2022

@author: AndreasMiltiadous
"""

import csv
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
import numpy as np


#choose eyes_closed_files
files = filedialog.askopenfilenames(title="Select all csv's with eyes_closed")
column_names = ["number", "eyes_closed_bpm"]
eyes_closed_df=pd.DataFrame(columns=column_names)


for file in files:
    filename = Path(file).name
    number= filename.split(")")[0]
    file_df=pd.read_csv(file)
    average=file_df["bpm"].mean()
    thisdict = {"number": number,"eyes_closed_bpm": average}
    eyes_closed_df=eyes_closed_df.append(thisdict,ignore_index=True)
    
#choose eyes_open_calm files
files = filedialog.askopenfilenames(title="Select all csv's with eyes_open")
column_names = ["number", "eyes_open_bpm"]
eyes_open_df=pd.DataFrame(columns=column_names)

for file in files:
    filename = Path(file).name
    number= filename.split(")")[0]
    file_df=pd.read_csv(file)
    average=file_df["bpm"].mean()
    thisdict = {"number": number,"eyes_open_bpm": average}
    eyes_open_df=eyes_open_df.append(thisdict,ignore_index=True)
    
#choose stressed    
files = filedialog.askopenfilenames(title="Select all csv's with stress")  
column_names = ["number", "stressed_bpm"]
stressed_df=pd.DataFrame(columns=column_names)
  
for file in files:
    filename = Path(file).name
    number= filename.split(")")[0]
    file_df=pd.read_csv(file)
    average=file_df["bpm"].mean()
    thisdict = {"number": number,"stressed_bpm": average}
    stressed_df=stressed_df.append(thisdict,ignore_index=True)
    
newdf=pd.merge(eyes_closed_df,eyes_open_df,how='inner',on='number')
newdf=pd.merge(newdf,stressed_df,how='inner',on='number')

dirr=filedialog.askdirectory(title="where to save")

newdf.to_csv(dirr+"/bpm_total.csv",index=False)