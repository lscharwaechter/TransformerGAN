# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

import torch
import matplotlib.pyplot as plt
import os

import convolutionalAE as cAE
from utils import MinMaxScaler3D
import dataset

# Define Training hyperparameters
epochs = 30
lr = 0.001

# Model name
model_save = 0
model_name = 'darlene'

# Load the dataset
batch_size = 1
train_data, test_data = dataset.getDataset()
maxlen, num_variables = train_data.shape[1], train_data.shape[2]

# %% 
# Define Model hyperparameters
d_latent = 30

# Initialize classes
scaler = MinMaxScaler3D(feature_range=(-1,1))       
convNet = cAE.ConvolutionalAE(maxlen, num_variables, d_latent)

# Preprocessing
train_data_scaled = scaler.fit_transform(train_data.clone())
test_data_scaled = scaler.transform(test_data.clone())

# %%

# Define loss and optimizer function
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(convNet.parameters(),
                             lr=lr, betas=(0.9,0.98), eps=1e-9)

def train_AE_epoch(train_batches: float, epoch: int):
    '''
    This function optimizes the autoencoder, i.e. the reconstruction
    from a latent point back to the original time series for one batch,
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
        loss = loss_fn(outs, src_input)
        # Compute new gradients
        loss.backward() 
        # Perform backpropagation
        optimizer.step()
        # Accumulate error
        total_loss += loss.item()
    
    # Plot a comparison of the original timeseries batch
    # with the autoencoder reconstruction
    src_plt = scaler.inverse_transform(src_input.detach().clone().numpy())
    out_plt = scaler.inverse_transform(outs.detach().clone().numpy())
    plt.figure(epoch, figsize=(2,1))
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Epoch Nr. {epoch}')    
    ax1.plot(src_plt[0,:,:])
    ax1.set_title('Original')
    ax2.plot(out_plt[0,:,:])
    ax2.set_title('Reconstruction')
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
        loss = loss_fn(outs, src_input)
        # Accumulate error
        total_loss += loss.item()
        
    # Return mean error of the batch
    return total_loss/len(test_batches)

best_val_loss = float('inf')
best_epoch = float('inf')
best_model = None

train_errs = []
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
    train_errs.append(train_loss)
    valid_errs.append(val_loss)
    # Print the current losses
    print(f'Train Loss: {train_loss:.3f}, Test Loss: {val_loss:.3f}')  
    # Save the best model during training
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model = convNet.state_dict()
        
# Save the weights of the model into a file
if model_save:
    torch.save(best_model,
               os.path.join(os.getcwd(),model_name + ".pt"))
    
# Plot the errors
plt.figure(epoch+1, figsize=(2,1))
fig, ax1 = plt.subplots(1,1)
fig.subplots_adjust(hspace=0.5)
fig.suptitle('Train- and Test-Error')
ax1.plot(x_steps, train_errs)
ax1.plot(x_steps, valid_errs)
plt.legend(["Train Error","Test Error"])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

print('Done.')