# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter

Recommended inline plot size (8, 20) inches
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os

import transformerGAN as tg
#import convolutionalAE as convAE
import dataset
from utils import interpolate_points, interp_error

torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

# Load the dataset
SOS, EOS, PAD = [-3], [-4], -5
maxlen = 52
num_src_variables = 24
num_tgt_variables = 24
train_data, test_data = dataset.getDataset(SOS=SOS, EOS=EOS)

# Define model hyperparameters
model_name = 'elliot'
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
p2 = test_data[2,:,:]

# Set number of interpolation steps
n_steps = 10

# Get the latent encoding of both points
p1 = p1[np.newaxis,:,:].type(torch.FloatTensor)
p2 = p2[np.newaxis,:,:].type(torch.FloatTensor)
p1_memory = transformer.encode(p1)
p2_memory = transformer.encode(p2)

# Create interpolations
latent_points = interpolate_points(p1_memory, 
                                   p2_memory, 
                                   n_dim=d_latent, n_steps=n_steps)
latent_points = torch.from_numpy(latent_points)
latent_points = latent_points.type(torch.FloatTensor)

# Initialize error and latent reconstructions
total_error = 0
preds = torch.zeros([n_steps, maxlen, d_model])

print("Plotting interpolation...")
fig, axs = plt.subplots(n_steps, 1)
fig.tight_layout(pad=2.0)
for i, memory in enumerate(latent_points):
    preds[i] = transformer.greedy_decode(memory, maxlen, 24, SOS, EOS)
    axs[i].plot(preds[i,1:-1,:].detach().numpy())
    axs[i].set_title(f'Decoding of Latent Point Nr. {i+1}/10',fontsize=10)
error = interp_error(preds).detach().numpy()
axs[-1].text(-2.5,-4.5,f'Interpolation Error = {error:.7f}', fontsize=10)
plt.show()
print(f'Interpolation Error = {error:.7f}')
print('Done.')
