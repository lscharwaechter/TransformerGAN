# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

from scipy.io import arff
import pandas as pd
import torch
import numpy as np
import math

def getDataset(SOS: int = np.nan, EOS: int = np.nan):
    '''
    This function loads the NATOPS dataset in the arff-Format.
    Both the train- and test-dataset consist of a numpy-void 
    containing 180 Series objects, in which the 24 feature
    vectors are concatenated. As a first preprocessing step,
    the dataset is converted into a Torch Tensor with the format
    (N, S, E), where N is the number of multivariate sequences,
    S is the sequence length and E is the number of features. 
    '''
        
    # Load the dataset
    data_train = arff.loadarff('NATOPS_TRAIN.arff')
    data_test = arff.loadarff('NATOPS_TEST.arff')
    df_train = pd.DataFrame(data_train[0])
    df_test = pd.DataFrame(data_test[0])

    # Find all labels in the dataset
    # print(df_train['classAttribute'].sort_values().unique())

    # Drop the labels
    df_train = df_train.drop(columns='classAttribute')
    df_test = df_test.drop(columns='classAttribute')

    # Convert the dataset
    nData = df_train.shape[0] # 180
    nFeatures = df_train.iloc[0][0].shape[0] # 24
    if SOS is not np.nan:
        sequenceLength = len(df_train.iloc[0][0][0])+1 # 51+SOS
    else:
        sequenceLength = len(df_train.iloc[0][0][0])
    train_data = np.zeros((nData, sequenceLength, nFeatures))
    test_data = np.zeros((nData, sequenceLength, nFeatures))
    
    for i, row in df_train.iterrows():
        for f in range(nFeatures):
            if SOS is not np.nan:
                train_data[i,:,f] = np.concatenate((SOS,list(row.iloc[0][f])))
            else:
                train_data[i,:,f] = list(row.iloc[0][f])
     
    for i, row in df_test.iterrows():
        for f in range(nFeatures): 
            if SOS is not np.nan:
                test_data[i,:,f] = np.concatenate((SOS,list(row.iloc[0][f])))
            else:
                test_data[i,:,f] = list(row.iloc[0][f])
                        
    # Convert into Torch Tensor
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
         
    return train_data, test_data

def batchify(dataset: float, batch_size: int):
    '''
    Creates a list of batches from a given dataset.
    Everytime this function is called, the dataset is shuffled
    resulting in a different sample distribution per batch.
    '''
    idx = torch.randperm(dataset.shape[0])
    dataset = dataset[idx].view(dataset.size())
    
    num_batches = math.ceil(dataset.size()[0]/batch_size)
    batches = [dataset[batch_size*y:batch_size*(y+1),:,:] for y in range(num_batches)]
    
    return batches