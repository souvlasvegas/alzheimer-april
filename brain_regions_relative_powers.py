# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:22:55 2022

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
from mne.time_frequency import psd_welch
from scipy import stats
import antropy as ant

import copy

#############################################################################
#read data functions. We are not using them
def read_data(file_path):
    data=mne.io.read_raw_eeglab(file_path,preload=True)
    epochs=mne.make_fixed_length_epochs(data,duration=2,overlap=1)
    array=epochs.get_data()
    return array


def read_data_return_epochs(file_path):
    data=mne.io.read_raw_eeglab(file_path,preload=True)
    epochs=mne.make_fixed_length_epochs(data,duration=2,overlap=1)
    return epochs

def get_key(val,my_dict):
    for key, value in my_dict.items():
        for v in value:
            print (v)
            if v == val:
                return key
 
    return -1
##############################################################################
#returns dataframe with spectral features for every epoch
#ATTENTION: gets as parameter Epochs object
def eeg_power_band(epochs,pos,FREQ_BANDS):
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
    X : dataframe of shape [n_samples, freq_bands*electrodes]
        Transformed data.
    """


    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=75.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    l=psds
    l = copy.copy(psds)
    #print(freqs)

    X = pd.DataFrame()
    for key, values in FREQ_BANDS.items():
        fmin=values[0]
        fmax=values[1]
        #edw eixan la8os autoi
        #psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        
        #anti autoy, auth einai h dior8wsh
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)]
        m = copy.copy(psds_band)
        psds_band=np.sum(psds_band,axis=2)
        
        psds_band_reshaped=psds_band.reshape(len(psds),-1)
        df=pd.DataFrame(psds_band_reshaped)
        for (colname,colval) in df.iteritems():
           poscol=pos[colname]
           df.rename(columns={colname:poscol + "_" + key},inplace=True)
            
        dataf=copy.copy(df)
        X=pd.concat([X,df],axis=1)
    
    X_removed=X.copy(deep=True)    
    #remove Fp1,Fp2,Fz
    for  column in X.columns:
        if column.split('_')[0]=='Fp1' or column.split('_')[0]=='Fp2' or column.split('_')[0]=='Fz':
            X_removed.drop(columns=column,inplace=True)
    return X_removed

#not used
def concatenate_features(x):
    return np.concatenate((mean(x),std(x),ptp(x),var(x),rms(x),skewness(x),
                           kurtosis(x)),axis=-1)
    #return np.concatenate((mean(x),std(x)),axis=-1)

#returns dataframe with time features for every epoch
#ATTENTION: gets as parameter array object that we get from epochs.get_data()


def create_regions(X, FREQ_BANDS="default"):
    if FREQ_BANDS=="default":
        FREQ_BANDS = {"delta": [0.5, 4],
                      "theta": [4, 8],
                      "alpha": [8, 13],
                      "beta": [13, 25],
                      "gamma": [25, 75]}
        
    regions={"frontal":["F7","F3","F4","F8"],
             "parietal":["C3","Cz","C4","P3","Pz","P4"],
             "temporal":["T3","T5","T6","T4"],
             "occipital":["O1","O2"]}
    
    for region in regions:
        for band in FREQ_BANDS:
            X[region+"_"+band] = 0

    
    for column in X:
        electrode=column.split('_')[0]
        band=column.split('_')[1]
        region=get_key(electrode,regions)
        if (region!=-1):
            colname=region+"_"+band
            X[colname]=X[colname]+X[column]
    
    for column in X:
        electrode=column.split('_')[0]
        band=column.split('_')[1]
        region=get_key(electrode,regions)
        if (region!=-1):
            X.drop(columns=column,inplace=True)
    
    for column in X:
        region=column.split('_')[0]
        band=column.split('_')[1]
        div=len(regions.get(region))
        X[column]=X[column]/div
    
    return X

    
    
#%%
##############################################################################


    
FREQ_BANDS = {"delta": [0.5, 4],
              "theta": [4, 8],
              "alpha": [8, 13],
              "beta": [13, 25],
              "gamma": [25, 75]}


#choose all .set files (alzheimer , frontotemporal and control)
files = filedialog.askopenfilenames()

#add every filename in their respective file list
for file in files:
    data=mne.io.read_raw_eeglab(file,preload=True) 
    pos=data.info.ch_names
    epochs=mne.make_fixed_length_epochs(data,duration=5,overlap=2.5)
    freq_df=eeg_power_band(epochs,pos,FREQ_BANDS)
    create_regions(freq_df,FREQ_BANDS)

    split_tup = os.path.splitext(file)
    file_to_write = split_tup[0]
    freq_df.to_csv(file_to_write+"_region_relative_spectral_power.csv", index=False)
 


#%% 
