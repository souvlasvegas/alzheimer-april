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
from scipy import stats

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
##############################################################################
#returns dataframe with spectral features for every epoch
#ATTENTION: gets as parameter Epochs object
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
    X : dataframe of shape [n_samples, freq_bands*electrodes]
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
#time domain metrics

def mean(x):
    return np.mean(x,axis=-1)
def std(x):
    return np.std(x,axis=-1)
def ptp(x):
    return np.ptp(x,axis=-1)
def var(x):
    return np.var(x,axis=-1)
def rms(x):
    return np.sqrt(np.mean(x**2,axis=-1))
def skewness(x):
    return stats.skew(x,axis=-1)
def kurtosis(x):
    return stats.kurtosis(x,axis=-1)

############################################################################
#return time domain metrics as dataframe
def mean_df(x,pos):
    arr=mean(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_mean"},inplace=True)
    return df

def std_df(x,pos):
    arr=std(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_std"},inplace=True)
    return df

def prp_df(x,pos):
    arr=ptp(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_ptp"},inplace=True)
    return df 

def var_df(x,pos):
    arr=var(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_var"},inplace=True)
    return df

def rms_df(x,pos):
    arr=rms(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_rms"},inplace=True)
    return df

def skewness_df(x,pos):
    arr=skewness(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_skewness"},inplace=True)
    return df

def kurtosis_df(x,pos):
    arr=kurtosis(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_kurtosis"},inplace=True)
    return df

############################################################################
#not used
def concatenate_features(x):
    return np.concatenate((mean(x),std(x),ptp(x),var(x),rms(x),skewness(x),
                           kurtosis(x)),axis=-1)
    #return np.concatenate((mean(x),std(x)),axis=-1)

#returns dataframe with time features for every epoch
#ATTENTION: gets as parameter array object that we get from epochs.get_data()
def time_features(x,pos):
    features=pd.DataFrame()
    mean=mean_df(x,pos)
    std=std_df(x, pos)
    prp=prp_df(x, pos)
    var=var_df(x, pos)
    rms=rms_df(x, pos)
    skewness=skewness_df(x, pos)
    kurtosis=kurtosis_df(x, pos)
    features=pd.concat([mean,std,prp,var,rms,skewness,kurtosis], axis=1, ignore_index=False)
    return features
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
        cl="A"
    elif "C" in file_name:
        control_files.append(file)
        cl="C"
    elif "F" in file_name:
        frontotemporal_files.append(file)
        cl="F"
    data=mne.io.read_raw_eeglab(file,preload=True) 
    pos=data.info.ch_names
    epochs=mne.make_fixed_length_epochs(data,duration=4,overlap=2)
    data_array=epochs.get_data()
    #sample_data_epochs.append(s)
    freq_df=eeg_power_band(epochs,pos)
    time_df=time_features(data_array,pos)
    features=pd.concat([freq_df,time_df], axis=1, ignore_index=False)
    features["Class"]=cl
    features["Subject"]=file_name
    ##
    split_tup = os.path.splitext(file)
    file_to_write = split_tup[0]
    features.to_csv(file_to_write+"_features.csv", index=False)


        
#test=mean_df(data_array,pos)
#test2=std_df(data_array,pos)
#testall=pd.concat([test,test2], axis=1, ignore_index=False)   
    
