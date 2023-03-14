# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

import torch
from torch import Tensor
import numpy as np
from sklearn.preprocessing import MinMaxScaler

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def padding(src: Tensor, maxlen: int, value: int = 0):
    '''
    This function performs a padding along the sequence length
    dimension.
    src shape: (len, num_src_var) or (bz, len, num_src_var)
    maxlen: new size of the sequence length
    value: the padding value which is placed between
    the last element of src and maxlen
    '''
    if src.dim() == 2:  
        src_pad = torch.ones((maxlen,src.shape[1]),device=DEVICE)*value
        src_pad[:src.shape[0],:] = src
    elif src.dim() == 3:
        src_pad = torch.ones((src.shape[0],maxlen,src.shape[2]),device=DEVICE)*value
        src_pad[:,:src.shape[1],:] = src
        
    return src_pad

def create_masks(src: Tensor, tgt: Tensor, maxlen: int):
    '''
    src shape: (len, num_src_var) or (bz, len, num_src_var)
    tgt shape: (len, num_tgt_var) or (bz, len, num_src_var)
    src/tgt_padding_mask:
        The positions with the value of True will be ignored 
        while the position with the value of False will be unchanged.
    '''
    if src.dim() == 2 and tgt.dim() == 2:
        src_mask = torch.zeros((maxlen, maxlen),device=DEVICE).type(torch.bool)
        tgt_mask = generate_square_subsequent_mask(maxlen) #tgt.shape[0]
        
        src_padding_mask = torch.ones(maxlen, dtype=torch.bool,device=DEVICE)
        src_padding_mask[:src.shape[0]] = False
        
        tgt_padding_mask = torch.ones(maxlen, dtype=torch.bool,device=DEVICE)
        tgt_padding_mask[:tgt.shape[0]] = False
        
        memory_key_padding_mask = src_padding_mask
        
    elif src.dim() == 3 and tgt.dim() == 3:
        src_mask = torch.zeros((maxlen, maxlen),device=DEVICE).type(torch.bool)
        tgt_mask = generate_square_subsequent_mask(maxlen) #tgt.shape[0]
 
        src_padding_mask = torch.ones((src.shape[0],maxlen), dtype=torch.bool,device=DEVICE)
        for batch in range(src.shape[0]):
            src_padding_mask[batch,:src.shape[1]] = False
        
        tgt_padding_mask = torch.ones((tgt.shape[0],maxlen), dtype=torch.bool,device=DEVICE)
        for batch in range(tgt.shape[0]):
            tgt_padding_mask[batch,:tgt.shape[1]] = False
        
        memory_key_padding_mask = src_padding_mask
        
    else:
        raise RuntimeError("src and tgt have a different number of dimensions: batched vs. unbatched")
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask 

def interpolate_points(m1, m2, n_dim=60, n_steps=10):
    '''
    This function creates a uniform interpolation between
    two points in the latent space (memories) and returns
    them as an array (S, D) where S is the number of
    interpolation steps and D is the number of latent dimensions.
    '''
    # A single memory usually comes with a batch dimension (1, D),
    # which is to be removed
    if m1.dim() == m2.dim() == 2:
        m1 = m1.squeeze()
        m2 = m2.squeeze()

    # Initialize array of latent vectors
    vectors = np.zeros((n_steps, n_dim))
    
    # Create the ratios between the memory points
    ratios = np.linspace(0, 1, num=n_steps)
    
	# Linear interpolation
    for i, ratio in enumerate(ratios):
        for d in range(n_dim):
            vectors[i, d] = (1.0 - ratio) * m1[d] + ratio * m2[d]
    
    return vectors

def interp_error(latent_preds: float):
    '''
    This function iterates over all reconstructions of the
    interpolation between two memory latent points
    and accumulates the error between one interpolation point 
    and the subsequent interpolation point.
    '''
    loss_MSE = torch.nn.MSELoss()
    total_error = 0
    
    for i, _ in enumerate(latent_preds):
        if i == latent_preds.shape[0]-1:
            break
        total_error += loss_MSE(latent_preds[i],latent_preds[i+1])
  
    return total_error

class MinMaxScaler3D():
    '''
    This class contains methods to perform a scaling of values within
    a three-dimensional dataset of size (bz, length, ch) where
    bz is the batch size, length is the sequence length and ch is 
    the number of channels / features of the dataset.
    For every channel, a scaler is initialized independently and applied
    to the corresponding channel dimension. 
    The range in which the features should be scaled can be determined
    using the init() argument feature_range.
    '''
    def __init__(self,
                 feature_range: int = (-1, 1)):
        self.scalers = {}
        self.feature_range = feature_range
    
    def fit_transform(self, dataset: Tensor):
        for i in range(dataset.shape[2]):
            self.scalers[i] = MinMaxScaler(feature_range=self.feature_range)
            dataset[:, :, i] = torch.from_numpy(self.scalers[i].fit_transform(dataset[:, :, i]))          
        return dataset
    
    def transform(self, dataset: Tensor):
        for i in range(dataset.shape[2]):
            dataset[:, :, i] = torch.from_numpy(self.scalers[i].transform(dataset[:, :, i]))    
        return dataset
    
    def inverse_transform(self, dataset: Tensor):
        for i in range(dataset.shape[2]):
            dataset[:, :, i] = torch.from_numpy(self.scalers[i].inverse_transform(dataset[:, :, i]))
        return dataset
            
