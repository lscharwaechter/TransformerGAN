# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter

Recommended inline plot size (12, 4) inches
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

import transformerGAN as tg
#import convolutionalAE as convAE
import dataset

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
nhead = 1
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

# Choose a time series
p1 = test_data[2,:,:]

# Get the latent encoding of the point
p1 = p1[np.newaxis,:,:].type(torch.FloatTensor)
p1_memory = transformer.encode(p1)

# Make a reconstruction from the memory
p1_decoding = transformer.greedy_decode(p1_memory, maxlen, 24, SOS, EOS)

# Calculate Deviation
loss_fn = torch.nn.MSELoss()
loss = loss_fn(p1, p1_decoding)

# Plot the figure
plt.figure(1, figsize=(2,1))
fig, (ax1, ax2) = plt.subplots(1,2)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Memory Decoding')
ax1.plot(p1[0,1:,:])
ax1.set_title('Original')
ax1.text(-2,-3.5,f'Loss = {loss:.7f}', fontsize=15)
ax2.plot(p1_decoding[0,1:-1,:].detach().numpy())
ax2.set_title('Reconstruction')
plt.show()
