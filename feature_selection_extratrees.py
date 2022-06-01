# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 19:11:45 2022

@author: m_ant
"""

from pathlib import Path
import pandas as pd
import numpy as np
from tkinter import filedialog
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
import os

######################################################################
#feature selection

file = filedialog.askopenfilename()


df=pd.read_csv(file)
print(len(df.columns))
X_df=df.drop(["Class","Subject"],axis=1,inplace=False)
X=df.drop(["Class","Subject"],axis=1,inplace=False).values
y=df.Class.values
y_df=df.Class
Subject=df.Subject

#clf = ExtraTreesClassifier(n_estimators=100)
clf=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
clf=clf.fit(X,y)

model= SelectFromModel(clf,threshold="1.5*mean", prefit=True)


cols=model.get_support(indices=True)
cols_new=np.append(cols,[len(df.columns)-2,len(df.columns)-1])
X_new=df.iloc[:,cols_new]

print(X_new.columns)

directory = filedialog.askdirectory(title="choose directory for the merged csv to be created")
filename = Path(file).name
split_tup = os.path.splitext(filename)
file_name = split_tup[0]
name=file_name.split('_')[0]
X_new.to_csv(directory+"/"+file_name+"_features_selected_with_lgbmc.csv", index=False)

