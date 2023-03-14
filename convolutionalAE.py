# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

''' 
-- Notes --
Checkerboard effekt:
Use a kernel size that is divided by the stride,
avoiding the overlap issue.

If not using ConvTranspose1d:
each UpSampling2D layer must be followed by a Conv2D layer
that will learn to interpret the doubled input and be trained
to translate it into meaningful detail.
'''

class ConvolutionalAE(nn.Module):
    def __init__(self,
                 maxlen: int,
                 num_variables: int,
                 d_latent: int):
        super(ConvolutionalAE, self).__init__()
        self.d_latent = d_latent

        ##############################################
        ###               ENCODER-PART             ###
        ##############################################
        
        self.conv1_enc = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=21, 
                                   stride=1, padding=10, padding_mode='replicate')
        self.conv2_enc = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=18, 
                                   stride=1, padding=8, padding_mode='replicate')
        self.conv3_enc = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=15, 
                                   stride=1, padding=7, padding_mode='replicate')
        self.conv4_enc = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=13, 
                                   stride=1, padding=6, padding_mode='replicate')
        self.conv5_enc = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, 
                                   stride=1, padding=5, padding_mode='replicate')
        self.conv6_enc = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8,
                                   stride=1, padding=3, padding_mode='replicate')
        self.conv7_enc = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, 
                                   stride=1, padding=2, padding_mode='replicate')
        self.conv8_enc = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, 
                                   stride=1, padding=1, padding_mode='replicate')
        ## Aggregation layer
        self.conv9_enc = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3,
                                   padding=2, stride=1)
        self.dense_enc = nn.Linear(maxlen*num_variables, d_latent)
        
        ##############################################
        ###               DECODER-PART             ###
        ##############################################
        
        # From the latent space to hidden convolution size
        self.dense_dec = nn.Linear(d_latent, maxlen*num_variables)
        # De-Aggregation layer (1->256)
        self.conv1_dec = nn.ConvTranspose1d(in_channels=1, out_channels=256, kernel_size=3,
                                            padding=2, stride=1)
        self.conv2_dec = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3,
                                            padding=1, stride=1)
        self.conv3_dec = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5,
                                            padding=2, stride=1)
        self.conv4_dec = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=8,
                                            padding=3, stride=1)
        self.conv5_dec = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=11,
                                            padding=5, stride=1)
        self.conv6_dec = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=13,
                                            padding=6, stride=1)
        self.conv7_dec = nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=15,
                                            padding=7, stride=1)
        self.conv8_dec = nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=18,
                                            padding=8, stride=1)
        self.conv9_dec = nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=21,
                                            padding=10, stride=1)
        
        ## Bound the activation
        #self.Tanh = nn.Tanh()
        #self.ReLU = nn.ReLU()
        #self.ELU = nn.ELU()()
        
        # Best performance was achieved without the use of
        # an activation function 
        
    def encode(self, timeseries: Tensor):
        # Encode timeseries into latent space
        x = self.conv1_enc(timeseries)
        x = self.conv2_enc(x)
        x = self.conv3_enc(x)
        x = self.conv4_enc(x)
        x = self.conv5_enc(x)
        x = self.conv6_enc(x)
        x = self.conv7_enc(x)
        x = self.conv8_enc(x)
        x = self.conv9_enc(x)
        x = self.dense_enc(x)
        return x
    
    def decode(self, latentpoint: Tensor):
        # Decode latent point back into a time series
        x = self.dense_dec(latentpoint)
        x = self.conv1_dec(x)
        x = self.conv2_dec(x)
        x = self.conv3_dec(x)
        x = self.conv4_dec(x)
        x = self.conv5_dec(x)
        x = self.conv6_dec(x)
        x = self.conv7_dec(x)
        x = self.conv8_dec(x)
        x = self.conv9_dec(x)
        return x
        
    def forward(self, src_input: Tensor):
        # Reshape input from (bz, seqlength, dim) to
        # (bz, ch = 1, seqlength*dim) 
        # as required input for the 1d-convolution layers
        bz, seqlength, dim = src_input.shape
        src_input = src_input.reshape((bz,1,seqlength*dim))
        
        latent_point = self.encode(src_input)
        reconstruction = self.decode(latent_point)
        
        # Reshape flattened time series to the number of dimensions
        return reconstruction.reshape((bz,seqlength,dim))
    
    def sample(self, bz: int = 1):
        '''
        Samples a random point from the latent space.
        Currently, values are uniformly sampled 
        from the interval [-1,1)
        '''
        point = torch.rand((bz, 1, self.d_latent),device=DEVICE)
        upperl, lowerl = torch.ones(1,device=DEVICE)[0], -torch.ones(1,device=DEVICE)[0]
        point = (lowerl - upperl)*point + upperl
        return point.to(DEVICE)
    
class Discriminator(nn.Module):
    def __init__(self,
                 maxlen: int,
                 num_variables: int,
                 d_latent: int):
        super(Discriminator, self).__init__()
        
        # Similar to the Encoder-Part of the Autoencoder
        self.conv1_enc = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=21, 
                                   stride=1, padding=10, padding_mode='replicate')
        self.conv2_enc = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=18, 
                                   stride=1, padding=8, padding_mode='replicate')
        self.conv3_enc = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=15, 
                                   stride=1, padding=7, padding_mode='replicate')
        self.conv4_enc = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=13, 
                                   stride=1, padding=6, padding_mode='replicate')
        self.conv5_enc = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=11, 
                                   stride=1, padding=5, padding_mode='replicate')
        self.conv6_enc = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, 
                                   stride=1, padding=3, padding_mode='replicate')
        self.conv7_enc = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, 
                                   stride=1, padding=2, padding_mode='replicate')
        self.conv8_enc = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, 
                                   stride=1, padding=1, padding_mode='replicate')
        ## Aggregation layer
        self.conv9_enc = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=3,
                                   padding=2, stride=1)
        
        # Last layers
        self.dense_enc = nn.Linear(maxlen*num_variables, d_latent) #612 #1224 #4096
        self.dense_one = nn.Linear(d_latent,1)
        self.sigmoid = nn.Sigmoid()
        
    def encode(self, timeseries: Tensor):
        # Encode timeseries into prediction
        x = self.conv1_enc(timeseries)
        x = self.conv2_enc(x)
        x = self.conv3_enc(x)
        x = self.conv4_enc(x)
        x = self.conv5_enc(x)
        x = self.conv6_enc(x)
        x = self.conv7_enc(x)
        x = self.conv8_enc(x)
        x = self.conv9_enc(x)
        x = self.dense_enc(x)
        x = self.dense_one(x)
        x = self.sigmoid(x) # needs to be removed for Wasserstein GAN training
        return x
        
    def forward(self, src_input: Tensor):
        bz, seqlength, dim = src_input.shape
        src_input = src_input.reshape((bz,1,seqlength*dim))
        return self.encode(src_input)
        
