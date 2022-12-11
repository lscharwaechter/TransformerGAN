# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter

Recommended inline plot size (12, 4) inches
"""

import torch
import matplotlib.pyplot as plt
import os

import transformerGAN as tg
import dataset
from utils import padding, create_masks

torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%

# Define Training hyperparameters
epochs = 3
lr = 0.0001

# Model name
model_save = 0
model_name = 'rami'

# Load the dataset
SOS, EOS, PAD = [-3], [-4], -5
batch_size = 1 #4
maxlen = 52
num_src_variables = 24
num_tgt_variables = 24
train_data, test_data = dataset.getDataset(SOS=SOS, EOS=EOS)
## normalize feature vectors, toDo

# %%

# Define model hyperparameters
d_model = 24
d_latent = 60
ffn_hidden_dim = 128
nhead = 1
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
# Discriminator
loss_BCE = torch.nn.BCELoss()
optimizer_disc = torch.optim.Adam(discriminator.parameters(),
                                  lr=lr, betas=(0.9,0.98), eps=1e-9)
    
def train_AE_epoch(train_batches: float, epoch: int):
    '''
    This function optimizes the autoencoder, i.e. the reconstruction
    from a latent memory back to the original time series for one batch,
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
       
        outs_real = transformer(src_input,
                                tgt_input,
                                src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        
        transformer.zero_grad()
        loss = loss_MSE(outs_real, tgt_output)
        
        loss.backward() 
        optimizer.step()
        total_loss += loss.item()
    
    plt.figure(epoch, figsize=(2,1))
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(f'Epoch Nr. {epoch}')
    ax1.plot(src_input[0,1:,:])
    ax1.set_title('Original')
    ax2.plot(outs_real[0,:-1,:].detach().numpy())
    ax2.set_title('Reconstruction')
    plt.show()
        
    return total_loss/len(train_batches)

def create_fake_timeseries(bz: int):
    '''
    Creates a batch of fake time series and returns the corresponding
    src_masks and src_padding_masks
    '''
    # Sample random memories from the latent space
    memory_fake = transformer.sample(bz)
    # Decode the memories into time series
    src_fake = torch.zeros((bz, maxlen, num_tgt_variables), device=DEVICE)
    srcf_padding_mask = torch.zeros((bz, maxlen))
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
    
def train_GAN_epoch(train_batches: float):
    '''
    This function optimizes the discriminator- and generator part
    of the GAN and returns the loss for one batch.
    '''
    discriminator.train()
    transformer.train()
    
    nbatches = len(train_batches)   
    total_loss_disc = 0.
    total_loss_gen = 0.
    
    real_label = 1.
    fake_label = 0.
    
    for i, batch in enumerate(train_batches):
        ##############################################
        ###             DISCRIMINATOR-PART         ###
        ### maximizes log(D(x)) + log(1 - D(G(z))) ###
        ##############################################
    
        # Clear previous gradients
        discriminator.zero_grad()
            
        ### Train with all-real batch ###       
        # Get output
        src_input = batch.type(torch.FloatTensor)
        outs_real = discriminator(src_input)   
        # Create labels for the real batch
        bz = batch.size(0)
        labels = torch.full((bz,1), real_label, dtype=torch.float, device=DEVICE)
        # Calculate gradients
        loss_real = loss_BCE(outs_real, labels)
        loss_real.backward()
        
        ### Train with all-fake batch ###
        src_fake, srcf_mask, srcf_padding_mask = create_fake_timeseries(bz)      
        # The generation of the fake samples 
        # should not be considered/optimized by the discriminator: detach() 
        labels.fill_(fake_label)
        outs_fake = discriminator(src_fake.detach())
        loss_fake = loss_BCE(outs_fake, labels)
        loss_fake.backward()
        
        # Update Discriminator
        #loss_disc = loss_real + loss_fake
        loss_disc = (loss_real.item() + loss_fake.item())/2
        #loss_disc.backward()
        optimizer_disc.step()
        total_loss_disc += loss_disc
        
        ##############################################
        ###              GENERATOR-PART            ###
        ###          maximizes log(D(G(z)))        ###
        ##############################################
        
        # Clear previous gradients
        transformer.zero_grad()
    
        # For the generator loss, the fake batch is treated as real,
        # because the generator wants the discriminator to believe
        # the fake samples come from the real dataset
        labels.fill_(real_label)      
        outs_fake = discriminator(src_fake)
        
        # Calculate new gradients
        loss_gen = loss_BCE(outs_fake, labels)
        loss_gen.backward()
        
        #loss = loss_gen + loss_disc
        #loss.backward()
        optimizer_disc.step()
        
        # Update Generator (Transformer-Autoencoder)
        optimizer.step()
        total_loss_gen += loss_gen.item()

    return total_loss_disc/nbatches, total_loss_gen/nbatches

def validate_AE_epoch(test_batches: float):
    '''
    This function calculates the loss of a reconstruction from the memory
    back to the original time series (autoencoder) for one batch
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
       
        outs_real = transformer(src_input,
                                tgt_input,
                                src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        
        loss = loss_MSE(outs_real, tgt_output)
        total_loss += loss.item()
        
    return total_loss/len(test_batches)

best_val_loss = float('inf')
best_epoch = float('inf')
best_model = None

x_steps = range(epochs)
train_AE_errs = []
train_DISC_errs = []
train_GEN_errs = []
valid_errs = []

for epoch in range(1, epochs+1):
    print(f'Epoch: {epoch}/{epochs}')
    # Get batches
    train_batches = dataset.batchify(train_data, batch_size=batch_size)
    test_batches = dataset.batchify(test_data, batch_size=batch_size)
    # Optimize networks and get the losses
    train_AE_loss = train_AE_epoch(train_batches, epoch)
    train_DISC_loss, train_GEN_loss = train_GAN_epoch(train_batches)   
    val_loss = validate_AE_epoch(test_batches)    
    # Save errors in lists
    train_AE_errs.append(train_AE_loss)
    train_DISC_errs.append(train_DISC_loss)
    train_GEN_errs.append(train_GEN_loss) 
    valid_errs.append(val_loss)
    # Print the current losses
    print(f'Train Loss: {train_AE_loss:.3f}, Test Loss: {val_loss:.3f}')
    print(f'Discriminator loss: {train_DISC_loss:.3f}, Generator Loss: {train_GEN_loss:.3f}\n')    
    
    # Save the best model during training
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model = transformer.state_dict()

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
    torch.save(best_model,
               os.path.join(os.getcwd(),model_name + ".pt"))
    
print('Done.')