# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:43:01 2023

@author: Leon
"""

import dataset
import transformerGAN as tg
#import convolutionalAE as cAE
from utils import MinMaxScaler3D

import torch
import os
import numpy as np
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

fake_data = scaler.inverse_transform(fake_data)

# %%

n_total = bz*(maxlen-1)
fake_data = fake_data.reshape((n_total,num_src_variables))
p = np.zeros(4)

H_E = 0
for _, values in enumerate(fake_data.T):
    p[0] = len(list(x for x in values if x >= 1))/n_total
    p[1] = len(list(x for x in values if 1 > x >= 0))/n_total
    p[2] = len(list(x for x in values if 0 > x >= -1))/n_total
    p[3] = len(list(x for x in values if x < -1))/n_total
    H_max = -len(p)*(1/len(p)*np.log(1/len(p)))
    entropy = 0.
    for i in p:
        if i != 0:
            entropy += -i*np.log(i)
    H_E += 1/H_max * entropy
H = H_E/num_src_variables
print(H)
#transGAN: 0.3664
#transWGAN: 0.3503

