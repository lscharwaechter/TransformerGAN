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
maxlen, num_variables = train_data.shape[1], train_data.shape[2]

# %% 
# Define Model hyperparameters
d_latent = 60

# Initialize classes
scaler = MinMaxScaler3D(feature_range=(-1,1))       
convNet = cAE.ConvolutionalAE(maxlen, num_variables, d_latent)
convDisc = cAE.Discriminator(maxlen, num_variables, d_latent)

# custom weights initialization called on netG and netD
def weights_init(m):
    if type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d or type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

# Initialize weights  
convNet.apply(weights_init)
convDisc.apply(weights_init)

# Preprocessing
train_data = scaler.fit_transform(train_data.clone())
test_data = scaler.transform(test_data.clone())
train_data.to(DEVICE)
test_data.to(DEVICE)

# %%

# Define loss and optimizer function
loss_MSE = torch.nn.MSELoss()
optimizer = torch.optim.Adam(convNet.parameters(),
                             lr=lr, betas=(0.9,0.98), eps=1e-9)

loss_BCE = torch.nn.BCELoss()
optimizerDisc = torch.optim.Adam(convDisc.parameters(),
                                 lr=lr, betas=(0.9,0.98), eps=1e-9)

def train_AE_batch(batch: float, batchNr: int, epoch: int):
    '''
    This function optimizes the autoencoder, i.e. the reconstruction
    from a latent point back to the original time series for one batch,
    and returns the loss.
    '''
    convNet.train()
    src_input = batch.type(torch.FloatTensor)
    
    # Clear previous gradients
    convNet.zero_grad()
    # Get output
    outs = convNet(src_input)
    # Calculate error and calculate new gradients
    loss = loss_MSE(outs, src_input)
    loss.backward() 
    # Perform backpropagation
    optimizer.step()
    
    # Plot a comparison of an original timeseries from the batch
    # with the current autoencoder reconstruction
    if batchNr == len(train_batches)-1:
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
    
    # Return error
    return loss.item()

def train_GAN_batch():
    '''
    This function optimizes the discriminator- and generator part
    of the GAN and returns the loss for one batch.
    '''
    convNet.train()
    convDisc.train()
    
    real_label = 1.
    fake_label = 0.
    
    ##############################################
    ###           DISCRIMINATOR-PART           ###
    ###     maximizes log (D(x) - D(G(z)))     ###
    ##############################################
    
    # In case the Discriminator should be trained more often
    for _ in range(1):
        
        # Clear previous gradients
        convDisc.zero_grad() 
        
        #################################
        ### Train with all-real batch ###
        
        # Get a random batch from the training-dataset
        batch = dataset.batchify(train_data, batch_size=batch_size)[0]      
        # Get prediction
        output = convDisc(batch.type(torch.FloatTensor)).view(-1)    
        # Calculate loss and gradients
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=DEVICE)
        errD_real = loss_BCE(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        #################################
        ### Train with all-fake batch ###  
        
        # Generate a random batch of fake time series
        noise = convNet.sample(batch_size)
        fake = convNet.decode(noise)
        fake = fake.reshape((batch_size,maxlen,num_variables))
        # Get prediction
        output = convDisc(fake.detach()).view(-1)
        # Calculate loss and gradients
        label.fill_(fake_label)
        errD_fake = loss_BCE(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        
        # Update Discriminator
        optimizerDisc.step()
        errD = (errD_real.item() + errD_fake.item())/2
        
    ##############################################
    ###              GENERATOR-PART            ###
    ###           maximizes log D(G(z))        ###
    ##############################################
    
    # Clear previous gradients
    convNet.zero_grad()
    
    # Generate a random batch of fake time series
    noise = convNet.sample(batch_size)
    fake = convNet.decode(noise)
    fake = fake.reshape((batch_size,maxlen,num_variables))
    # For the generator loss, the fake batch is treated as real,
    # because the generator wants the discriminator to believe
    # the fake samples come from the real dataset. 
    label.fill_(real_label)
    # Get prediction
    output = convDisc(fake).view(-1)
    # Calculate loss and gradients
    errG = loss_BCE(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    
    # Update Generator
    optimizer.step()
    
    return errD, errG.item(), D_x, D_G_z1, D_G_z2

def validate_AE_epoch(test_batches: float):
    '''
    This function calculates the loss of a reconstruction from a latent
    point back to the original time series (autoencoder) for one epoch
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
        
    # Return mean error of the batch
    return total_loss/len(test_batches)

# Init best model
best_val_loss = float('inf')
best_epoch = float('inf')
best_model = None

# Init loss recordings
train_AE_errs = []
train_DISC_errs = []
train_GEN_errs = []
valid_errs = []

# Init prediction recordings
D_x_list = []
D_G_z1_list = []
D_G_z2_list = []

# Training schedule
for epoch in range(1, epochs+1):
    #start_time = time.time()
    print(f'Epoch: {epoch}/{epochs}')
    
    # Get batches
    train_batches = dataset.batchify(train_data, batch_size=batch_size)
    test_batches = dataset.batchify(test_data, batch_size=batch_size)
    
    # Init losses per epoch
    AE_epoch_loss = 0.
    DISC_epoch_loss = 0.
    GEN_epoch_loss = 0.
    D_x_epoch = 0.
    D_G_z1_epoch = 0.
    D_G_z2_epoch = 0.
    
    # Iterate over all batches
    for i, batch in enumerate(train_batches):
        #print(f'Batch: {i+1}/{len(train_batches)}')
        # Optimize networks and get the losses
        AE_epoch_loss += train_AE_batch(batch, i, epoch)
        DISC_batch_loss, GEN_batch_loss, D_x_batch, D_G_z1_batch, D_G_z2_batch = train_GAN_batch()
        DISC_epoch_loss += DISC_batch_loss
        GEN_epoch_loss += GEN_batch_loss
        D_x_epoch += D_x_batch
        D_G_z1_epoch += D_G_z1_batch
        D_G_z2_epoch += D_G_z2_batch
    # Get mean losses of the epoch
    AE_epoch_loss/=len(train_batches)
    DISC_epoch_loss/=len(train_batches)
    GEN_epoch_loss/=len(train_batches)
    D_x_epoch/=len(train_batches)
    D_G_z1_epoch/=len(train_batches)
    D_G_z2_epoch/=len(train_batches)
    
    # Calculate validation error of the epoch
    val_loss = validate_AE_epoch(test_batches)
    
    # Save errors in lists
    train_AE_errs.append(AE_epoch_loss)
    train_DISC_errs.append(DISC_epoch_loss)
    train_GEN_errs.append(GEN_epoch_loss) 
    valid_errs.append(val_loss)
    
    # Save the mean prediction of the Discriminator in lists
    D_x_list.append(D_x_epoch)
    D_G_z1_list.append(D_G_z1_epoch)
    D_G_z2_list.append(D_G_z2_epoch)
    
    # Print the current losses
    print(f'[Autoencoder] Train Loss: {AE_epoch_loss:.4f}, Test Loss: {val_loss:.4f}')
    print(f'[GAN] Discriminator loss: {DISC_epoch_loss:.4f}, Generator Loss: {GEN_epoch_loss:.4f}')    
    print(f'[GAN] D(x): {D_x_epoch:.4f}, D(G(z)) = {D_G_z1_epoch:.4f} | {D_G_z2_epoch:.4f}\n')
    
    # Save the best model during training
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

# Save the weights of the model into a file
if model_save:
    torch.save(convNet.state_dict(),
               os.path.join(os.getcwd(),model_name + ".pt"))
    np.save('./train_AE_errs.npy',train_AE_errs)
    np.save('./train_DISC_errs.npy',train_DISC_errs)
    np.save('./train_GEN_errs.npy',train_GEN_errs)
    np.save('./valid_errs.npy',valid_errs)
    np.save('./D_x_list.npy',D_x_list)
    np.save('./D_G_z1_list.npy',D_G_z1_list)
    np.save('./D_G_z2_list.npy',D_G_z2_list)
    stats = {
        "best val loss": best_val_loss,
        "best epoch": best_epoch
        }
    np.save('./stats.npy',stats)

print('Done.')
       
# Plot the errors
x_steps = range(epochs)
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
