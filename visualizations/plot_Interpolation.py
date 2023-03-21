# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter

Recommended inline plot size (8, 20) inches
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import transformerGAN as tg
#import convolutionalAE as convAE
import dataset
from utils import MinMaxScaler3D, interpolate_points, interp_error

torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

# Load the dataset
SOS, EOS, PAD = -3, 4, -5
maxlen = 52
num_src_variables = 24
num_tgt_variables = 24
train_data, test_data = dataset.getDataset()
#train_data, test_data = dataset_full.getDataset(maxlen-1, train_size = 0.83)

## Normalize feature vectors
scaler = MinMaxScaler3D(feature_range=(-1,1))
train_data = scaler.fit_transform(train_data.clone())
test_data = scaler.transform(test_data.clone())

# Add SOS-Token
train_data = dataset.addTokens(train_data,SOS=SOS)
test_data = dataset.addTokens(test_data,SOS=SOS)

# Define model hyperparameters
model_name = 'rami'
d_model = 24
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

# Load the learned model parameters
transformer.load_state_dict(
    torch.load(os.path.join(os.getcwd(),model_name + ".pt")))

# %%

# Choose two time series
p1 = test_data[0,:,:]
p2 = test_data[5,:,:]

# Set number of interpolation steps
n_steps = 10

# Get the latent encoding of both points
p1 = p1[np.newaxis,:,:].type(torch.FloatTensor)
p2 = p2[np.newaxis,:,:].type(torch.FloatTensor)
p1_memory = transformer.encode(p1)
p2_memory = transformer.encode(p2)

# Create interpolations in the latent space
latent_points = interpolate_points(p1_memory, p2_memory, n_dim=d_latent, n_steps=n_steps)
latent_points = torch.from_numpy(latent_points).type(torch.FloatTensor)

# Decode memories of the latent points into time series signals
interpolated_timeseries = transformer.greedy_decode_bz(latent_points, maxlen, num_tgt_variables, SOS)
interpolated_timeseries = scaler.inverse_transform(interpolated_timeseries[:,1:,:].detach().numpy())
interpolated_timeseries = torch.from_numpy(interpolated_timeseries).type(torch.FloatTensor)

#%%

print("Plotting interpolation...")
fig, axs = plt.subplots(n_steps, 1)
fig.tight_layout(pad=2.0)
for i, signal in enumerate(interpolated_timeseries):
    axs[i].plot(signal)
    axs[i].set_title(f'Decoding of Latent Point Nr. {i+1}/10',fontsize=10)

error = interp_error(interpolated_timeseries)
axs[-1].text(-2.5,-4.5,f'Interpolation Error = {error:.7f}', fontsize=10)
plt.show()
print(f'Interpolation Error = {error:.7f}')
print('Done.')