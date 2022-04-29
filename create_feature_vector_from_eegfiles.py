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
    FREQ_BANDS = {"delta": [0.5, 4],
                  "theta": [4, 8],
                  "alpha": [8, 13],
                  "beta": [13, 25],
                  "gamma": [25, 45]}


    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=45.)
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
    return X,l,m
##############################################################################
#time domain metrics
#%%
def mean(x):
    return np.mean(x,axis=-1)

def std(x):
    return np.std(x,axis=-1)
#Oxi
def ptp(x):
    return np.ptp(x,axis=-1)
#Oxi
def var(x):
    return np.var(x,axis=-1)
#Oxi
def rms(x):
    return np.sqrt(np.mean(x**2,axis=-1))
#Oxi
def abs_diff_signal(x):
    np.sum(np.abs(np.diff(x,axis=-1)),axis=-1)
def skewness(x):
    return stats.skew(x,axis=-1)
def kurtosis(x):
    return stats.kurtosis(x,axis=-1)
#%%
###############################################################################
#antropy features
#%%
def sample_entropy(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.sample_entropy(column)
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def perm_entropy(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.perm_entropy(column,normalize=True)
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def spectral_entropy(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.spectral_entropy(column,sf=500,method='welch',normalize=True)
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def svd_entropy(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.svd_entropy(column,normalize=True)
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def approx_entropy(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.app_entropy(column)
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def hjorth_mobility(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.hjorth_params(column)[0]
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def hjorth_complexity(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.hjorth_params(column)[1]
            a[row,col]=value
            col=col+1
        row=row+1
    return a

#%%
##############################################################################

def petrosian_fractal(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.petrosian_fd(column)
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def katz_fractal(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.katz_fd(column)
            a[row,col]=value
            col=col+1
        row=row+1
    return a


def higuchi_fractal(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.higuchi_fd(column)
            a[row,col]=value
            col=col+1
        row=row+1
    return a

def detrended_fluc(x):
    a=np.zeros((data_array.shape[0],data_array.shape[1]))
    row=0
    for vector in data_array:
        col=0
        for column in vector:
            value=ant.detrended_fluctuation(column)
            a[row,col]=value
            col=col+1
        row=row+1
    return a
#%%
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

def abs_diff_df(x,pos):
    arr=abs_diff_signal(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_abs_diff"},inplace=True)
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
#%%
#############################################################################
#Antropy metrics

def perm_en_df(x,pos):
    arr=perm_entropy(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_permentropy"},inplace=True)
    return df

def spectral_en_df(x,pos):
    arr=spectral_entropy(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_spectralentropy"},inplace=True)
    return df

def svd_en_df(x,pos):
    arr=svd_entropy(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_svdentropy"},inplace=True)
    return df

def app_en_df(x,pos):
    arr=approx_entropy(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_app_entropy"},inplace=True)
    return df

def sample_en_df(x,pos):
    arr=sample_entropy(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_sampleentropy"},inplace=True)
    return df

def hjorth_mobility_df(x,pos):
    arr=hjorth_mobility(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_hjorth_mobility"},inplace=True)
    return df

def hjorth_complexity_df(x,pos):
    arr=hjorth_complexity(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_hjorth_complexity"},inplace=True)
    return df
#%%
############################################################################
def petrosian_fr_df(x,pos):
    arr=petrosian_fractal(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_petrosian_fr"},inplace=True)
    return df

def katz_fr_df(x,pos):
    arr=katz_fractal(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_katz_fr"},inplace=True)
    return df

def higuchi_fr_df(x,pos):
    arr=higuchi_fractal(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_higuchi_fr"},inplace=True)
    return df

def detrended_fl_df(x,pos):
    arr=detrended_fluc(x)
    df=pd.DataFrame(arr)
    for (colname,colval) in df.iteritems():
       poscol=pos[colname]
       df.rename(columns={colname:poscol + "_detrended_fl"},inplace=True)
    return df
############################################################################
#%%
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
    #prp=prp_df(x, pos)
    #var=var_df(x, pos)
    #rms=rms_df(x, pos)
    #abs_diff=abs_diff_df(x,pos)
    skewness=skewness_df(x, pos)
    kurtosis=kurtosis_df(x, pos)
    features=pd.concat([mean,std,skewness,kurtosis], axis=1, ignore_index=False)
    return features

def entropy_features(x,pos):
    features=pd.DataFrame()
    permen=perm_en_df(x,pos)
    spec=spectral_en_df(x,pos)
    svd=svd_en_df(x,pos)
    app=app_en_df(x,pos)
    sample=sample_en_df(x, pos)
    hj_mob=hjorth_mobility_df(x, pos)
    hj_com=hjorth_complexity_df(x, pos)
    features=pd.concat([permen,spec,svd,app,sample,hj_mob,hj_com], axis=1, ignore_index=False)
    return features

def fractal_features(x,pos):
    features=pd.DataFrame()
    petr=petrosian_fr_df(x, pos)
    katz=katz_fr_df(x, pos)
    hig=higuchi_fr_df(x, pos)
    detr=detrended_fl_df(x, pos)
    features=pd.concat([petr,katz,hig,detr], axis=1, ignore_index=False)
    return features
    
    
#%%
##############################################################################
#%%
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
    freq_df,l,m=eeg_power_band(epochs,pos)
    time_df=time_features(data_array,pos)
    entropy_df=entropy_features(data_array, pos)
    fractal_df=fractal_features(data_array, pos)
    features=pd.concat([freq_df,time_df,entropy_df,fractal_df], axis=1, ignore_index=False)
    features["Class"]=cl
    features["Subject"]=file_name
    ##
    split_tup = os.path.splitext(file)
    file_to_write = split_tup[0]
    features.to_csv(file_to_write+"_features.csv", index=False)



    

#test=mean_df(data_array,pos)
#test2=std_df(data_array,pos)
#testall=pd.concat([test,test2], axis=1, ignore_index=False)   
#%% 
