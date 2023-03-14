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

train_data = train_data.to(DEVICE)
test_data = test_data.to(DEVICE)

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

discriminator = tg.Discriminator(d_model,
                                 d_latent,
                                 maxlen,
                                 num_src_variables,
                                 num_encoder_layers,
                                 nhead,
                                 ffn_hidden_dim)

transformer.init_parameters()
transformer = transformer.to(DEVICE)

discriminator.init_parameters()
discriminator = discriminator.to(DEVICE)

# %%

# Define loss function and optimizer

# Transformer
loss_MSE = torch.nn.MSELoss() #ignore_index=PAD?
optimizer = torch.optim.Adam(transformer.parameters(),
                             lr=lr, betas=(0.9,0.98), eps=1e-9)

# Wasserstein GAN Optimizer
G_optim = torch.optim.RMSprop(transformer.parameters(), lr=5e-5)
D_optim = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)
    
def train_AE_batch(batch: float, batchNr: int, epoch: int):
    '''
    This function optimizes the autoencoder, i.e. the reconstruction
    from a latent memory back to the original time series for one batch,
    and returns the loss.
    '''
    transformer.train()
    transformer.zero_grad()

    maxlen = batch.shape[1]
    src_input = batch.type(torch.FloatTensor)
    tgt_input = batch[:,:-1,:]
    tgt_output = batch[:,1:,:]
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask = create_masks(src_input, tgt_input, maxlen)
    
    # src_input = padding(src_input,maxlen,PAD)
    tgt_input = padding(tgt_input,maxlen,PAD).type(torch.FloatTensor)
    tgt_output = padding(tgt_output,maxlen,PAD).type(torch.FloatTensor)
   
    # Get prediction
    outs_real = transformer(src_input.to(DEVICE),
                            tgt_input.to(DEVICE),
                            src_mask.to(DEVICE), tgt_mask.to(DEVICE),
                            src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE),
                            memory_key_padding_mask.to(DEVICE))
    
    # Calculate loss and gradients
    loss = loss_MSE(outs_real.to(DEVICE), tgt_output.to(DEVICE))
    loss.backward() 
    
    # Perform weight update
    optimizer.step()
    
    # Plot a comparison of an original timeseries from the batch
    # with the current autoencoder reconstruction
    if batchNr == len(train_batches)-1:
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
    
    return loss.item()

def create_fake_timeseries(bz: int):
    '''
    Creates a batch of fake time series and returns the corresponding
    src_masks and src_padding_masks.
    This is necessary if sequences of different lengths are contained
    in one batch (not used with NATOPS)
    '''
    # Sample random memories from the latent space
    memory_fake = transformer.sample(bz)
    # Decode the memories into time series
    src_fake = torch.zeros((bz, maxlen, num_tgt_variables), device=DEVICE)
    srcf_padding_mask = torch.zeros((bz, maxlen), device=DEVICE)
    for b in range(bz):
        src_fake[b] = transformer.greedy_decode(memory_fake[b],
                                                maxlen, 
                                                num_tgt_variables, 
                                                SOS, EOS)
        srcf_mask, _, padding_mask, _, _ = create_masks(src_fake[b], src_fake[b], maxlen)
        # Padding in case the generated time series is shorter than maxlen
        src_fake[b] = padding(src_fake[b], maxlen)
        srcf_padding_mask[b] = padding_mask
        
    return src_fake, srcf_mask, srcf_padding_mask
    
def train_GAN_batch():
    '''
    This function optimizes the discriminator- and generator part
    of the Wasserstein GAN and returns the loss for one batch.
    '''
    discriminator.train()
    transformer.train()
    
    ##############################################
    ###           DISCRIMINATOR-PART           ###
    ###       maximizes (D(x) - D(G(z)))       ###
    ##############################################
    
    # Train discriminator's critic more often
    for _ in range(5):
        # Clear previous gradients
        discriminator.zero_grad()
            
        #################################
        ### Train with all-real batch ###   
        
        # Get a random batch from the training-dataset
        src_real = dataset.batchify(train_data, batch_size=batch_size)[0]
        src_real = src_real.type(torch.FloatTensor) 
        
        # Get prediction                  
        logits_real, p_real = discriminator(src_real.to(DEVICE)) 
        logits_real = logits_real.to(DEVICE)
        D_x = logits_real.mean().item()
        
        # Calculate loss and gradients
        loss_real = -torch.mean(logits_real)
        loss_real.backward()
        
        #################################
        ### Train with all-fake batch ###
        
        # Get a random batch of fake time series
        memory_fake = transformer.sample(batch_size)
        src_fake = transformer.greedy_decode_bz(memory_fake, maxlen, num_tgt_variables, SOS)
        
        # Get prediction     
        src_fake = src_fake.to(DEVICE)
        # The generation of the fake samples 
        # should not be considered/optimized by the discriminator: detach() 
        logits_fake, p_fake = discriminator(src_fake.detach())
        logits_fake = logits_fake.to(DEVICE)
        D_G_z1 = logits_fake.mean().item()
        
        # Calculate loss and gradients
        loss_fake = torch.mean(logits_fake)
        loss_fake.backward()
        
        # Update Discriminator
        D_optim.step()       

        ### Weight Clipping
        for p in discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)        
        
    ##############################################
    ###              GENERATOR-PART            ###
    ###             maximizes D(G(z))          ###
    ##############################################
    
    # Clear previous gradients
    transformer.zero_grad()
    
    # Get a random batch of fake time series
    memory_fake = transformer.sample(batch_size)
    src_fake = transformer.greedy_decode_bz(memory_fake, maxlen, num_tgt_variables, SOS)
    
    # Get prediction
    logits_fake, p_fake = discriminator(src_fake)
    logits_fake = logits_fake.to(DEVICE)
    D_G_z2 = logits_fake.mean().item()
    
    # Calculate loss and new gradients
    loss_gen = -torch.mean(logits_fake)
    loss_gen.backward()
    
    # Update Generator (Transformer-Decoder)
    G_optim.step()
    
    return loss_real.item(), loss_fake.item(), loss_gen.item(), D_x, D_G_z1, D_G_z2
    
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

# Init best model
best_val_loss = float('inf')
best_epoch = float('inf')
best_model = None

# Init loss recordings
train_AE_errs = []
train_DISC_real_errs = []
train_DISC_fake_errs = []
train_GEN_errs = []
valid_errs = []

# Init prediction recordings
D_x_list = []
D_G_z1_list = []
D_G_z2_list = []

# Training schedule
for epoch in range(1, epochs+1):
    print(f'Epoch: {epoch}/{epochs}')
    
    # Get batches
    train_batches = dataset.batchify(train_data, batch_size=batch_size)
    test_batches = dataset.batchify(test_data, batch_size=batch_size)
    
    # Init losses per epoch
    AE_epoch_loss = 0.
    DISC_epoch_real_loss = 0.
    DISC_epoch_fake_loss = 0.
    GEN_epoch_loss = 0.
    D_x_epoch = 0.
    D_G_z1_epoch = 0.
    D_G_z2_epoch = 0.
    
    # Iterate over all batches
    for i, batch in enumerate(train_batches):
        #print(f'Batch: {i+1}/{len(train_batches)}')
        # Optimize networks and get the losses
        AE_epoch_loss += train_AE_batch(batch, i, epoch)
        DISC_batch_real_loss, DISC_batch_fake_loss, GEN_batch_loss, D_x_batch, D_G_z1_batch, D_G_z2_batch = train_GAN_batch()
        DISC_epoch_real_loss += DISC_batch_real_loss
        DISC_epoch_fake_loss += DISC_batch_fake_loss
        GEN_epoch_loss += GEN_batch_loss
        D_x_epoch += D_x_batch
        D_G_z1_epoch += D_G_z1_batch
        D_G_z2_epoch += D_G_z2_batch
    # Get mean losses of the epoch
    AE_epoch_loss/=len(train_batches)
    DISC_epoch_real_loss/=len(train_batches)
    DISC_epoch_fake_loss/=len(train_batches)
    GEN_epoch_loss/=len(train_batches)
    D_x_epoch/=len(train_batches)
    D_G_z1_epoch/=len(train_batches)
    D_G_z2_epoch/=len(train_batches)
    
    # Calculate validation error of the epoch
    val_loss = validate_AE_epoch(test_batches)
    
    # Save errors in lists
    train_AE_errs.append(AE_epoch_loss)
    train_DISC_real_errs.append(DISC_epoch_real_loss)
    train_DISC_fake_errs.append(DISC_epoch_fake_loss)
    train_GEN_errs.append(GEN_epoch_loss) 
    valid_errs.append(val_loss)
    
    # Save the mean prediction of the Discriminator in lists
    D_x_list.append(D_x_epoch)
    D_G_z1_list.append(D_G_z1_epoch)
    D_G_z2_list.append(D_G_z2_epoch)
    
    # Print the current losses
    print(f'[Autoencoder] Train Loss: {AE_epoch_loss:.4f}, Test Loss: {val_loss:.4f}')
    print(f'[GAN] Discriminator Real loss: {DISC_epoch_real_loss:.4f}, Discriminator Fake loss: {DISC_epoch_fake_loss:.4f}, Generator Loss: {GEN_epoch_loss:.4f}')    
    print(f'[GAN] D(x): {D_x_epoch:.4f}, D(G(z)) = {D_G_z1_epoch:.4f} | {D_G_z2_epoch:.4f}\n')
    
    # Save the best model during training
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch

# Save the weights of the model into a file
if model_save:
    torch.save(transformer.state_dict(),
               os.path.join(os.getcwd(),model_name + ".pt"))
    np.save('./train_AE_errs.npy',train_AE_errs)
    np.save('./train_DISC_real_errs.npy',train_DISC_real_errs)
    np.save('./train_DISC_fake_errs.npy',train_DISC_fake_errs)
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

print('Done.')
