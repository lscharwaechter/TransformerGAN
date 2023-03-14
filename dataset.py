# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

from scipy.io import arff
import scipy.signal
import pandas as pd
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def getDataset(seqlen: int = 51, SOS: int = np.nan, EOS: int = np.nan):
    '''
    This function loads the NATOPS dataset in the arff-Format.
    Both the train- and test-dataset consist of a numpy-void 
    containing 180 Series objects, in which the 24 feature
    vectors are concatenated. As a first preprocessing step,
    the dataset is converted into a Torch Tensor with the format
    (N, S, E), where N is the number of multivariate sequences,
    S is the sequence length (default = 51) and E is the number 
    of features. The sequence can be scaled to another length
    using the input argument seqlen. 
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
    
    #if SOS is not np.nan:
     #   if seqlen == 51:
      #      sequenceLength = len(df_train.iloc[0][0][0])+1 # 51+SOS
       # else:
        #    sequenceLength = seqlen+1
    #else:
     #   if seqlen == 51:
      #      sequenceLength = len(df_train.iloc[0][0][0])
       # else:
        #    sequenceLength = seqlen
        
    sequenceLength = seqlen
    train_data = np.zeros((nData, sequenceLength, nFeatures))
    test_data = np.zeros((nData, sequenceLength, nFeatures))
    
    for i, row in df_train.iterrows():
        for f in range(nFeatures):
            train_data[i,:,f] = list(row.iloc[0][f])
            
    for i, row in df_test.iterrows():
        for f in range(nFeatures): 
            test_data[i,:,f] = list(row.iloc[0][f])
          
    #for i, row in df_train.iterrows():
     #   for f in range(nFeatures):
      #      if SOS is not np.nan:
       #         train_data[i,:,f] = np.concatenate((SOS,list(row.iloc[0][f])))
        #    else:
         #       train_data[i,:,f] = list(row.iloc[0][f])
     
    #for i, row in df_test.iterrows():
     #   for f in range(nFeatures): 
      #      if SOS is not np.nan:
       #         test_data[i,:,f] = np.concatenate((SOS,list(row.iloc[0][f])))
        #    else:
         #       test_data[i,:,f] = list(row.iloc[0][f])
                        
    # Convert into Torch Tensor
    train_data = torch.from_numpy(train_data)
    test_data = torch.from_numpy(test_data)
         
    return train_data, test_data

def getLabels():
    '''
    Returns the labels of the dataset
    '''
    # Load the dataset
    data_train = arff.loadarff('NATOPS_TRAIN.arff')
    data_test = arff.loadarff('NATOPS_TEST.arff')
    df_train = pd.DataFrame(data_train[0])
    df_test = pd.DataFrame(data_test[0])
    
    # Define dictionary to remap the label values
    labeldict = {b'1.0' : 1, b'2.0' : 2, b'3.0' : 3,
                 b'4.0' : 4, b'5.0' : 5, b'6.0' : 6}  
    df_train = df_train.replace({'classAttribute': labeldict})
    df_test = df_test.replace({'classAttribute': labeldict})
    
    # Get labels
    train_labels = df_train.iloc[:,1].values
    test_labels = df_test.iloc[:,1].values
    
    return train_labels, test_labels
    
def addTokens(dataset: float, SOS: int = np.nan, EOS: int = np.nan):
    '''
    Adds a SOS-Token to each sequence of the multivariate dataset.
    '''
    
    N, _, E = dataset.shape
    SOS_= torch.ones((N,1,E))*SOS

    return torch.cat((SOS_,dataset),1)

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

def scale(dataset: float, SOS: int = np.nan, seqlen: int = 51):
    '''
    This function is used to scale the dataset samples to a different
    sequence length after the dataset is constructed.
    Thereby it ensures that the SOS-Token still only appears once.
    If a SOS-Token is used, then the returned sequences have a
    length of seqlen+1.
    '''
    N, _, E = dataset.shape
    if SOS is np.nan:
        new_dataset = np.zeros((N, seqlen, E))
        for i, seq in enumerate(dataset):
            new_dataset[i,:,:] = scipy.signal.resample(seq,seqlen,axis=0)
    else:
        new_dataset = np.zeros((N, seqlen+1, E))
        new_dataset[:,0,:] = torch.ones((1,E))*SOS
        for i, seq in enumerate(dataset):
            new_dataset[i,1:,:] = scipy.signal.resample(seq[1:],seqlen,axis=0)
    return torch.tensor(new_dataset)

#train_labels, test_labels = getLabels()
#train_data, test_data = getDataset()

#plt.figure()
#palette = sns.color_palette("tab10",24)
#sns.lineplot(data=train_data[0,:,:], dashes=False, palette=palette, 
#             legend=False, alpha=0.8)



# Plot a comparison of an original timeseries from the batch
# with the current autoencoder reconstruction
#src_plt = scaler.inverse_transform(src_input[:,1:,:].detach().clone().numpy())
#outs_plt = scaler.inverse_transform(outs_real[:,:-1,:].detach().clone().numpy())
#fig, axes = plt.subplots(1,2)
#palette = sns.color_palette("deep",24)
#sns.lineplot(data=src_plt[0,:,:], dashes=False, palette=palette, 
#             legend=False, ax=axes[0]).set(title='Original')
#sns.lineplot(data=outs_plt[0,:,:], dashes=False, palette=palette, 
#             legend=False, ax=axes[1]).set(title='Reconstruction')
#fig.suptitle(f'Epoch Nr. {epoch}')
#plt.show()