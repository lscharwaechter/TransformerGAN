# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 15:43:01 2023

@author: Leon
"""

import dataset
import transformerGAN as tg
#import convolutionalAE as cAE
from utils import MinMaxScaler3D
import matplotlib.pyplot as plt
import torch
import os

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get dataset
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
maxlen = 52

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

bz = 10
SOS = -3
memory_fake = transformer.sample(bz)
fake_data = transformer.greedy_decode_bz(memory_fake, maxlen, num_tgt_variables, SOS)
fake_data = fake_data[:,1:,:].detach().numpy() #exclude <SOS>

# %%

# Rescale
fake_data = scaler.inverse_transform(fake_data)

'''
# Plot the figures
for i in range(bz):
    plt.figure(i, figsize=(1,1))
    fig, ax1 = plt.subplots(1,1)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Example {i+1}')
    ax1.plot(fake_data[i])
    plt.show()
'''

# Create a figure with 10 subplots and 5 plots in two rows
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 6))
# Flatten the 2D array of axes to a 1D array
axs = axs.flatten()
# Generate the subplots
for i in range(len(axs)):
    axs[i].plot(fake_data[i])
    axs[i].set_title(f"Example {i+1}")
# Adjust the space between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.3)
# Show the plot
plt.show()