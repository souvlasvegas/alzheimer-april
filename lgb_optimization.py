# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:53:19 2022

@author: m_ant
"""

import pandas as pd
import numpy as np
from tkinter import filedialog

from sklearn import ensemble
#from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import LeaveOneGroupOut
from skopt import space
from skopt import gp_minimize
import os
from sklearn.pipeline import Pipeline
from functools import partial
import lightgbm as lgb


def optimize (params, param_names, x, y, groups):
    params= dict(zip(param_names,params))
    model=lgb.LGBMClassifier(**params,n_jobs=-1)
    #model=ensemble.RandomForestClassifier(**params,n_jobs=-1)
    kf=model_selection.LeaveOneGroupOut()
    accuracies=[]
    for idx in kf.split(X=x,y=y,groups=groups):
        train_idx, test_idx= idx[0],idx[1]
        xtrain= x[train_idx]
        ytrain= y[train_idx]
        
        xtest= x[test_idx]
        ytest=y[test_idx]
        
        model.fit(xtrain,ytrain)
        preds=model.predict(xtest)
        fold_acc=metrics.accuracy_score(ytest,preds)
        accuracies.append(fold_acc)
        
    return -1.0* np.mean(accuracies)

file = filedialog.askopenfilename()

df=pd.read_csv(file)
X=df.drop(["Class","Subject"],axis=1,inplace=False).values
y=df.Class.values
groups=df.Subject.values
split_tup = os.path.splitext(file)
file_to_write = split_tup[0]
file_object = open(file_to_write+"_results.txt", 'a')

param_space=[
        space.Real(0.01, 1.0, 'log-uniform', name='learning_rate'),
        space.Integer(2, 500, name='num_leaves'),
        space.Integer(0, 500, name='max_depth'),
        space.Integer(0, 200, name='min_child_samples'),
        space.Integer(100, 100000, name='max_bin'),
        space.Real(0.01, 1.0, 'uniform', name='subsample'),
        space.Integer(0, 10, name='subsample_freq'),
        space.Real(0.01, 1.0, 'uniform', name='colsample_bytree'),
        space.Integer(0, 10, name='min_child_weight'),
        space.Integer(100000, 500000, name='subsample_for_bin'),
        space.Real(1e-9, 1000, 'log-uniform', name='reg_lambda'),
        space.Real(1e-9, 1.0, 'log-uniform', name='reg_alpha'),
        space.Real(1e-6, 500, 'log-uniform', name='scale_pos_weight'),
        space.Integer(10, 10000, name='n_estimators')
        
    ]

param_names=[
    "learning_Rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "max_bin",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "min_child_weight",
    "subsample_for_bin",
    "reg_lambda",
    "reg_alpha",
    "scale_pos_weight",
    "n_estimators"           
    ]

optimization_function=partial(optimize,param_names=param_names,x=X,y=y,groups=groups)

result=gp_minimize(
    optimization_function,
    dimensions=param_space,
    n_calls=2,
    n_random_starts=10,
    verbose=10
    )

dic=dict(param_names,result.x)
file_object.write(str(dic))
file_object.close()