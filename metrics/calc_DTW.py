# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:43:01 2023

@author: Leon Scharwächter
"""

import dataset
import transformerGAN as tg
#import convolutionalAE as cAE
from utils import MinMaxScaler3D

import torch
import os
import numpy as np
import scipy.stats as stats
from dtaidistance import dtw_ndim

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get dataset
maxlen = 52
train_data, test_data = dataset.getDataset()
#train_data, test_data = dataset_full.getDataset(maxlen-1, train_size = 0.83)

## Normalize feature vectors
scaler = MinMaxScaler3D(feature_range=(-1,1))
train_data = scaler.fit_transform(train_data.clone())
test_data = scaler.transform(test_data.clone())
test_data = test_data.numpy()

# %%

# Load the model and generate fake samples

model_name = 'rami'
num_src_variables, num_tgt_variables, d_model = 24, 24, 24
d_latent = 60
ffn_hidden_dim = 128 
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6

# Initialize the model
transformer = tg.TransformerModel(d_model,
                                  d_latent,
                                  maxlen,
                                  num_src_variables,
                                  num_tgt_variables,
                                  num_encoder_layers,
                                  num_decoder_layers,
                                  nhead,
                                  ffn_hidden_dim)

# Load state
transformer.load_state_dict(
        torch.load(os.path.join(os.getcwd(),model_name + ".pt"),map_location=torch.device('cpu')))

# %%

bz = 50
SOS = -3
memory_fake = transformer.sample(bz)
fake_data = transformer.greedy_decode_bz(memory_fake, maxlen, num_tgt_variables, SOS)
fake_data = fake_data[:,1:,:].detach().numpy() #exclude <SOS>


# %%

# Calculate average dynamic time warping (DTW)
print('Calculate average DTW Score...')
d_total = 0.
for i, fake_sample in enumerate(fake_data):
    d = torch.inf
    print(f'{i+1}/{len(fake_data)}')
    for j, real_sample in enumerate(test_data):
        for dim in range(num_tgt_variables):
            real_sample[:,dim] = stats.zscore(real_sample[:,dim])
            fake_sample[:,dim] = stats.zscore(fake_sample[:,dim])
        d_temp = dtw_ndim.distance(real_sample.T, fake_sample.T, window=10)
        if d_temp < d:
            d = d_temp
    d_total += d
d_total /= len(fake_data)
print('d =',d_total)
