# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

import convolutionalAE as cAE
from utils import MinMaxScaler3D
import dataset
#import dataset_full

torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',DEVICE)

# Define Training hyperparameters
epochs = 2000
lr = 0.001

# Model name
model_save = 1
model_name = 'darlene'

# Load the dataset
batch_size = 32
train_data, test_data = dataset.getDataset()
#train_data, test_data = dataset_full.getDataset(maxlen, train_size = 0.7)
maxlen, num_variables = train_data.shape[1], train_data.shape[2]

# %% 
# Define Model hyperparameters
d_latent = 60

# Initialize classes
scaler = MinMaxScaler3D(feature_range=(-1,1))       
convNet = cAE.ConvolutionalAE(maxlen, num_variables, d_latent)

# custom weights initialization called on netG and netD
def weights_init(m):
    if type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d or type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# Initialize weights  
convNet.apply(weights_init)

# Preprocessing
train_data_scaled = scaler.fit_transform(train_data.clone())
test_data_scaled = scaler.transform(test_data.clone())

train_data_scaled.to(DEVICE)
test_data_scaled.to(DEVICE)

# %%

# Define loss and optimizer function
loss_MSE = torch.nn.MSELoss()
optimizer = torch.optim.Adam(convNet.parameters(),
                             lr=lr, betas=(0.9,0.98), eps=1e-9)

def train_AE_epoch(train_batches: float, epoch: int):
    '''
    This function optimizes the autoencoder, i.e. the reconstruction
    from a latent point back to the original time series for one epoch,
    and returns the loss.
    '''
    convNet.train()
    total_loss = 0.
    for _, batch in enumerate(train_batches):
        src_input = batch.type(torch.FloatTensor) 
        # Clear previous gradients
        convNet.zero_grad()
        # Get output
        outs = convNet(src_input)
        # Calculate error
        loss = loss_MSE(outs, src_input)
        # Compute new gradients
        loss.backward() 
        # Perform backpropagation
        optimizer.step()
        # Accumulate error
        total_loss += loss.item()
    
    # Plot a comparison of an original timeseries from the batch
    # with the current autoencoder reconstruction
    src_plt = scaler.inverse_transform(src_input.detach().clone().numpy())
    out_plt = scaler.inverse_transform(outs.detach().clone().numpy())
    fig, axes = plt.subplots(1,2)
    palette = sns.color_palette("deep",24)
    sns.lineplot(data=src_plt[0,:,:], dashes=False, palette=palette, 
                 legend=False, ax=axes[0]).set(title='Original')
    sns.lineplot(data=out_plt[0,:,:], dashes=False, palette=palette, 
                 legend=False, ax=axes[1]).set(title='Reconstruction')
    fig.suptitle(f'Epoch Nr. {epoch}')
    plt.show()
        
    # Return mean error of the batch
    return total_loss/len(train_batches)

def validate_AE_epoch(test_batches: float):
    '''
    This function calculates the loss of a reconstruction from a latent
    point back to the original time series (autoencoder) for one batch
    without optimizing.
    '''
    convNet.eval()
    total_loss = 0.
    for _, batch in enumerate(test_batches):
        src_input = batch.type(torch.FloatTensor)       
        # Get output
        outs = convNet(src_input)
        # Calculate error
        loss = loss_MSE(outs, src_input)
        # Accumulate error
        total_loss += loss.item()
        
    # Return mean error of all batches
    return total_loss/len(test_batches)

best_val_loss = float('inf')
best_epoch = float('inf')
best_model = None

train_AE_errs = []
valid_errs = []
x_steps = range(epochs)

for epoch in range(1, epochs+1):
    print(f'Epoch: {epoch}/{epochs}')
    # Get batches
    train_batches = dataset.batchify(train_data_scaled, batch_size=batch_size)
    test_batches = dataset.batchify(test_data_scaled, batch_size=batch_size)
    # Optimize networks and get the losses
    train_loss = train_AE_epoch(train_batches, epoch) 
    val_loss = validate_AE_epoch(test_batches)  
    # Save errors in lists
    train_AE_errs.append(train_loss)
    valid_errs.append(val_loss)
    # Print the current losses
    print(f'Train Loss: {train_loss:.3f}, Test Loss: {val_loss:.3f}') 
    # Save the best model during training
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
# Save the weights of the model into a file
if model_save:
    torch.save(convNet.state_dict(),
               os.path.join(os.getcwd(),model_name + ".pt"))    
    np.save('./train_AE_errs.npy',train_AE_errs)
    np.save('./valid_errs.npy',valid_errs)
    stats = {
        "best val loss": best_val_loss,
        "best epoch": best_epoch
        }
    np.save('./stats.npy',stats)
    
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

print('Done.')
