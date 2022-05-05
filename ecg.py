# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:47:27 2022

@author: AndreasMiltiadous
"""

import heartpy as hp
from tkinter import filedialog
import mne
import numpy as np
from heartpy import exceptions
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


file = filedialog.askopenfilename()
data=mne.io.read_raw_eeglab(file,preload=True)
sample_info=data.info
fr=sample_info["hpi_meas"]
data_array=data.get_data()
data_array=np.transpose(data_array)
data_array = data_array.flatten()


working_data, measures = hp.process_segmentwise(data_array, sample_rate=300.0, segment_width = 20, segment_overlap = 0.25,mode="full")

#windowsize in seconds, overlap in fraction (0.5 = 50%, needs to be 0 or greater and less than 1.0)
#slices now contains a list of index pairs corresponding to window segments.
seg=6
slices, segmented_measures = hp.process_segmentwise(data_array, sample_rate=300.0, segment_width=seg, segment_overlap=0.5)

bpm=segmented_measures["bpm"]

dataset=pd.DataFrame()
dataset['bpm'] = bpm
dataset.insert(0, 'time', (dataset.index + 1)*(seg/2))
dataset.fillna(method='ffill',inplace=True)
dataset['Tomorrow'] = dataset["bpm"].shift(1)
dataset["Tomorrow"].fillna(method='bfill',inplace=True)

dataset['bpm2'] = dataset["Tomorrow"].rolling(window=3).mean()
dataset["bpm2"]=dataset["bpm2"].shift(-2)
dataset.drop(columns="Tomorrow",inplace=True)
dataset["bpm2"].fillna(method="ffill",inplace=True)
###############################################################################

dataset.plot(x="time", y=["bpm", "bpm2"])

###############################################################################
#create annotation dataframe
annotations=data.annotations
annot=pd.DataFrame({'time':annotations.onset, "description":annotations.description})

###############################################################################
#logic for cutting bpm to calm, stressed, very stressed
filename = Path(file).name
split_tup = os.path.splitext(file)
prefix=split_tup[0]

temp=dataset.copy("deep")
temp.drop(columns="bpm",inplace=True)
temp.rename(columns = {'bpm2':'bpm'}, inplace = True)


found666=False
for index, row in annot.iterrows():
    if row["description"]=="2 | Eyes Close ":
        #create resting state bpm file
        pointer=index+1
        while pointer<len(annot)-1 and annot["description"][pointer]!="1 | Eyes Open ":
            pointer=pointer+1
        try:
            if annot["description"][pointer]=="1 | Eyes Open ":
                start=row["time"]
                print(start)
                end=annot["time"][pointer]
                print (end)
                temp2 = temp[temp['time'] >= start]
                temp2=temp2[temp2["time"]<=end]
                plt.axvline(x=start,color='b',label="resting closed")
                plt.axvline(x=end,color='b')
                temp2.to_csv(prefix+"_resting_eyesclosed.csv",index=False)
        except KeyError:
            print("warning key error 1")
    if row["description"]=="3 | start 666 ":
        found666=True
    if found666==False:
        #tsekaroume to stressed
        if row["description"]=="7 | elevator door open ":
            start=row["time"]
            pointer=index+1
            while annot["description"][pointer]!="condition 6" and pointer<len(annot)-1:
                pointer=pointer+1
            try:
                if annot["description"][pointer]=="condition 6":
                    end=annot["time"][pointer]+10
                    temp2 = temp[temp['time'] >= start]
                    temp2=temp2[temp2["time"]<=end]
                    plt.axvline(x=start,color='c',label="stressed")
                    plt.axvline(x=end,color='c')
                    temp2.to_csv(prefix+"_stressed.csv",index=False)
            except KeyError:
                print("warning key error 2")
        #tsekaroume to eyes open calm
        elif row["description"]=="1 | Eyes Open ":
            start=row["time"]
            pointer=index+1
            while annot["description"][pointer]!="7 | elevator door open " and pointer<len(annot)-1:
                pointer=pointer+1
            try:
                if annot["description"][pointer]=="7 | elevator door open ":
                    end=annot["time"][pointer]-8
                    temp2 = temp[temp['time'] >= start]
                    temp2=temp2[temp2["time"]<=end]
                    plt.axvline(x=start,color='r',label="eyes open")
                    plt.axvline(x=end,color='r')
                    temp2.to_csv(prefix+"_eyes_open_calm.csv",index=False)
            except KeyError:
                print("warning key error 3")
    else: # if found666==True
        if row["description"]=="3 | start 666 ":
            start=row["time"]+10
            pointer=index+1
            while annot["description"][pointer]!="2 | Eyes Close " and pointer<len(annot)-1:
                pointer=pointer+1
            try:
                if annot["description"][pointer]=="2 | Eyes Close ":
                    end=annot["time"][pointer]-5
                    temp2 = temp[temp['time'] >= start]
                    temp2=temp2[temp2["time"]<=end]
                    plt.axvline(x=start,color='y',label="666")
                    plt.axvline(x=end,color='y')
                    temp2.to_csv(prefix+"_666.csv",index=False)
            except KeyError:
                print("warning key error 4")

plt.legend(loc="upper left")
plt.show()                
###############################################################################            