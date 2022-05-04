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

dataset['bpm2'] = dataset.iloc[:,1].rolling(window=3).mean()


