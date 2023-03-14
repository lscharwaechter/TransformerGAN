# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

import transformerGAN as tg
import dataset
#import dataset_full
from utils import padding, create_masks, MinMaxScaler3D

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:',DEVICE)

# %%

# Define Training hyperparameters
epochs = 2000
lr = 0.0001 

# Model name
model_save = 1
model_name = 'rami'

# Load the dataset
SOS, EOS, PAD = -3, -4, -5
batch_size = 32
maxlen = 52 # number of sample points INCL. TOKENS
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

# %%

# Define model hyperparameters
d_model = num_src_variables
d_latent = 60
ffn_hidden_dim = 128
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6

# Initialize the models
transformer = tg.TransformerModel(d_model,
                                  d_latent,
                                  maxlen,
                                  num_src_variables,
                                  num_tgt_variables,
                                  num_encoder_layers,
                                  num_decoder_layers,
                                  nhead,
                                  ffn_hidden_dim)

transformer.init_parameters()
transformer = transformer.to(DEVICE)

# %%

# Define loss function and optimizer

# Transformer
loss_MSE = torch.nn.MSELoss() #ignore_index=PAD?
optimizer = torch.optim.Adam(transformer.parameters(),
                             lr=lr, betas=(0.9,0.98), eps=1e-9)
    
def train_AE_epoch(train_batches: float, epoch: int):
    '''
    This function optimizes the autoencoder, i.e. the reconstruction
    from a latent memory back to the original time series for one epoch,
    and returns the loss.
    '''
    transformer.train()
    total_loss = 0.
    for i, batch in enumerate(train_batches):
        maxlen = batch.shape[1]
        src_input = batch.type(torch.FloatTensor)
        tgt_input = batch[:,:-1,:]
        tgt_output = batch[:,1:,:]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask = create_masks(src_input, tgt_input, maxlen)
        
        # src_input = padding(src_input,maxlen,PAD)
        tgt_input = padding(tgt_input,maxlen,PAD).type(torch.FloatTensor)
        tgt_output = padding(tgt_output,maxlen,PAD).type(torch.FloatTensor)
       
        outs_real = transformer(src_input.to(DEVICE),
                                tgt_input.to(DEVICE),
                                src_mask.to(DEVICE), tgt_mask.to(DEVICE),
                                src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE),
                                memory_key_padding_mask.to(DEVICE))
        
        transformer.zero_grad()
        loss = loss_MSE(outs_real.to(DEVICE), tgt_output.to(DEVICE))
        
        loss.backward() 
        optimizer.step()
        total_loss += loss.item()
    
    # Plot a comparison of an original timeseries from the batch
    # with the current autoencoder reconstruction
    src_plt = scaler.inverse_transform(src_input[:,1:,:].detach().clone().numpy())
    outs_plt = scaler.inverse_transform(outs_real[:,:-1,:].detach().clone().numpy())
    fig, axes = plt.subplots(1,2)
    palette = sns.color_palette("deep",24)
    sns.lineplot(data=src_plt[0,:,:], dashes=False, palette=palette, 
                 legend=False, ax=axes[0]).set(title='Original')
    sns.lineplot(data=outs_plt[0,:,:], dashes=False, palette=palette, 
                 legend=False, ax=axes[1]).set(title='Reconstruction')
    fig.suptitle(f'Epoch Nr. {epoch}')
    plt.show()
        
    return total_loss/len(train_batches)

def validate_AE_epoch(test_batches: float):
    '''
    This function calculates the loss of a reconstruction from the memory
    back to the original time series (autoencoder) for one epoch
    without optimizing.
    '''
    transformer.eval()
    total_loss = 0.
    for i, batch in enumerate(test_batches):
        maxlen = batch.shape[1]
        src_input = batch.type(torch.FloatTensor)
        tgt_input = batch[:,:-1,:]
        tgt_output = batch[:,1:,:]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask = create_masks(src_input, tgt_input, maxlen)
        
        tgt_input = padding(tgt_input,maxlen,PAD).type(torch.FloatTensor)
        tgt_output = padding(tgt_output,maxlen,PAD).type(torch.FloatTensor)
       
        outs_real = transformer(src_input.to(DEVICE),
                                tgt_input.to(DEVICE),
                                src_mask.to(DEVICE), tgt_mask.to(DEVICE),
                                src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE),
                                memory_key_padding_mask.to(DEVICE))
        
        loss = loss_MSE(outs_real.to(DEVICE), tgt_output.to(DEVICE))
        total_loss += loss.item()
        
    return total_loss/len(test_batches)

best_val_loss = float('inf')
best_epoch = float('inf')
best_model = None

x_steps = range(epochs)
train_AE_errs = []
valid_errs = []

for epoch in range(1, epochs+1):
    print(f'Epoch: {epoch}/{epochs}')
    # Get batches
    train_batches = dataset.batchify(train_data, batch_size=batch_size)
    test_batches = dataset.batchify(test_data, batch_size=batch_size)
    # Optimize networks and get the losses
    train_AE_loss = train_AE_epoch(train_batches, epoch)  
    val_loss = validate_AE_epoch(test_batches)    
    # Save errors in lists
    train_AE_errs.append(train_AE_loss)
    valid_errs.append(val_loss)
    
    # Print the current losses
    print(f'[Autoencoder] Train Loss: {train_AE_loss:.4f}, Test Loss: {val_loss:.4f}')
    
    # Save the best model during training
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

# Plot the errors
plt.figure(epoch+1, figsize=(2,1))
fig, ax1 = plt.subplots(1,1)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Train- and Test-Error')
ax1.plot(x_steps, train_AE_errs)
ax1.plot(x_steps, valid_errs)
plt.legend(["Train Error","Test Error"])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

# Save the weights of the model into a file
if model_save:
    torch.save(transformer.state_dict(),
               os.path.join(os.getcwd(),model_name + ".pt"))
    np.save('./train_AE_errs.npy',train_AE_errs)
    np.save('./valid_errs.npy',valid_errs)
    stats = {
        "best val loss": best_val_loss,
        "best epoch": best_epoch
        }
    np.save('./stats.npy',stats)
    
print('Done.')
