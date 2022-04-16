# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:33:19 2022

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
from mne.time_frequency import psd_welch

#############################################################################
#read data functions. We are using read_data_return_epochs only
def read_data(file_path):
    data=mne.io.read_raw_eeglab(file_path,preload=True)
    epochs=mne.make_fixed_length_epochs(data,duration=2,overlap=1)
    array=epochs.get_data()
    return array



def read_data_return_epochs(file_path):
    data=mne.io.read_raw_eeglab(file_path,preload=True)
    epochs=mne.make_fixed_length_epochs(data,duration=2,overlap=1)
    return epochs
##############################################################################
def eeg_power_band(epochs,pos):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}



    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=45.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = pd.DataFrame()
    for key, values in FREQ_BANDS.items():
        fmin=values[0]
        fmax=values[1]
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        psds_band_reshaped=psds_band.reshape(len(psds),-1)
        df=pd.DataFrame(psds_band_reshaped)
        for (colname,colval) in df.iteritems():
           poscol=pos[colname]
           df.rename(columns={colname:poscol + "_" + key},inplace=True)
            
        X=pd.concat([X,df],axis=1)
    return X

##############################################################################
alzheimer_files = []
frontotemporal_files =[]
control_files = []

#choose all .set files (alzheimer , frontotemporal and control)
files = filedialog.askopenfilenames()

#add every filename in their respective file list
for file in files:
    filename = Path(file).name
    print(filename)
    split_tup = os.path.splitext(filename)
    file_name = split_tup[0]
    if "A" in file_name:
        alzheimer_files.append(file)
    elif "C" in file_name:
        control_files.append(file)
    elif "F" in file_name:
        frontotemporal_files.append(file)

#sample_data=[]
#for file in alzheimer_files:
#        s=read_data(file)
#        sample_data.append(s)

sample_data_epochs=[]
for file in alzheimer_files:
        data=mne.io.read_raw_eeglab(file,preload=True)
        pos=data.info.ch_names
        epochs=mne.make_fixed_length_epochs(data,duration=4,overlap=2)
        #sample_data_epochs.append(s)
        freq_df=eeg_power_band(epochs,pos)


    
    
