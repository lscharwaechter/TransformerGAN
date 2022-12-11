# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:31:23 2022

@author: Leon Scharwächter
"""

from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getDataset():
    data_train = arff.loadarff('NATOPS_TRAIN.arff')
    df_train = pd.DataFrame(data_train[0])

    # Find all labels in the dataset
    # print(df_train['classAttribute'].sort_values().unique())

    # Drop the labels
    df_train = df_train.drop(columns='classAttribute')

    # Create the dataset
    nData = df_train.shape[0] # 180
    nFeatures = df_train.iloc[0][0].shape[0] # 24
    sequenceLength = len(df_train.iloc[0][0][0]) # 51
    
    dataset = np.zeros((nData, sequenceLength, nFeatures))

    for i, row in df_train.iterrows():
        for f in range(nFeatures): 
            dataset[i,:,f] = list(row.iloc[0][f])
            
    return dataset

    