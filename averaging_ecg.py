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

files = filedialog.askopenfilenames(title="Select all csv's")

column_names = ["number", "average_bpm"]
new_df=pd.DataFrame(columns=column_names)


for file in files:
    filename = Path(file).name
    number= filename.split(")")[0]
    file_df=pd.read_csv(file)
    average=file_df["bpm"].mean()
    thisdict = {"number": number,"average_bpm": average}
    new_df=new_df.append(thisdict,ignore_index=True)
    

    
    
    