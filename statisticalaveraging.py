# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:44:32 2022

@author: m_ant
"""

import csv
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
import mne
import numpy as np
from tkinter.filedialog import asksaveasfile

import copy



files = filedialog.askopenfilenames()
av_df=pd.DataFrame()
#add every filename in their respective file list
for file in files:
    df=pd.read_csv(file)
    new=df.mean(axis=0)
    

    filename = Path(file).name
    numb=filename.split(")")[0]
    name=filename.split(")")[1].split("_")[0]+"_"+filename.split(")")[1].split("_")[1]
    new.name=numb
    av_df=pd.concat([av_df,new],axis=1)
av_dft=av_df.T
av_dft=av_dft*100000000000
direc=filedialog.askdirectory()


av_dft.to_csv(direc+"/average_absolute_regions.csv", index=True)
