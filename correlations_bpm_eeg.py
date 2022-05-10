# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:30:37 2022

@author: AndreasMiltiadous
"""

import csv
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
import mne
import numpy as np
from tkinter import Tk
from tkinter.filedialog import asksaveasfile
import copy
import seaborn as sb

Tk().withdraw()
stress_file = filedialog.askopenfilename(title="Anoixe to arxeio me tis meses apolutes times energeiwn se katastash stress")
calm_file = filedialog.askopenfilename(title="Anoixe to arxeio me tis meses apolutes times energeiwn se katastash eyes_open_calm")

stress_df=pd.read_csv(stress_file)
calm_df=pd.read_csv(calm_file)

metabolh_df=stress_df.copy(deep=True)  
for column in metabolh_df:
    if column!="subject":
        metabolh_df[column]=metabolh_df[column]-calm_df[column]
    
bpm_file = filedialog.askopenfilename(title="Anoixe to arxeio me ta mesa bpm")
bpm_df=pd.read_csv(bpm_file)
bpm_df["bpm_var"]=bpm_df["stressed_bpm"]-bpm_df["eyes_open_bpm"]
bpm_df.drop(columns=["eyes_closed_bpm","eyes_open_bpm","stressed_bpm"],inplace=True)

bpm_df.rename(columns = {'number':'subject'}, inplace = True)

total_df=pd.merge(bpm_df,metabolh_df,how='inner',on='subject')

total_df.drop(columns="subject",inplace=True)

pearsoncorr = total_df.corr(method='pearson')
spearmancorr = total_df.corr(method='spearman')
kendallcorr = total_df.corr(method='kendall')


sb.heatmap(pearsoncorr,vmax=1,vmin=-1,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

sb.heatmap(spearmancorr,vmax=1,vmin=-1,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

sb.heatmap(kendallcorr,vmax=1,vmin=-1, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)