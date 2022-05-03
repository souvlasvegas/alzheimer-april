# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:02:07 2022

@author: m_ant
"""

import pandas as pd
import numpy as np
from tkinter import filedialog

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import LeaveOneGroupOut
from skopt import space
from skopt import gp_minimize
import os

from functools import partial


def optimize (params, param_names, x, y, groups):
    params= dict(zip(param_names,params))
    model=ensemble.RandomForestClassifier(**params,n_jobs=-1)
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
        space.Integer(3,15, name="max_depth"),
        space.Integer(100,600, name="n_estimators"),
        space.Categorical(["gini","entropy"],name="criterion"),
        space.Real(0.01,1,prior="uniform",name="max_features")
    ]

param_names=[
    "max_depth",
    "n_estimators",
    "criterion",
    "max_features"
    ]

optimization_function=partial(optimize,param_names=param_names,x=X,y=y,groups=groups)

result=gp_minimize(
    optimization_function,
    dimensions=param_space,
    n_calls=30,
    n_random_starts=10,
    verbose=10
    )

dic=dict(param_names,result.x)
print(dic)
file_object.write(str(dic))
file_object.close()



