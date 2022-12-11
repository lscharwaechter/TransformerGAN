# -*- coding: utf-8 -*-
"""
@author: Leon Scharwächter
"""

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import numpy as np
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VarSeriesEmbedding(nn.Module):
    '''
    This class scales an input sequence, which is a tensor
    of one time step, containing all variables at this position,
    to the model dimension of the transformer (e.g. 512).
    This scaling is done with learnable parameters of a 
    1-dimensional feed forward neural network.
    '''
    def  __init__(self,
                  num_variables: int,
                  d_model: int):
        super(VarSeriesEmbedding, self).__init__()
        self.linear = nn.Linear(num_variables, d_model, bias=True)
        
    def forward(self, varSeries: Tensor):
        return self.linear(varSeries)
    
class PositionalEncoding(nn.Module):
    '''
    This class adds positional encoding to the sequence embedding 
    to introduce a notion of time dependend order.
    maxlen determines how far the position can have an effect 
    on a token.
    Returns the sum of the input embedding (bz, seqlength, d_model)
    with the raw positional encoding (1, seqlength, d_model)
    '''
    def __init__(self,
                 d_model: int,
                 maxlen: int = 5000,
                 dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)[:,:int(d_model/2)]
        
        # Add batch dimension (maxlen,d_model) -> (1,maxlen,d_model)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, seq_embedding: Tensor):
        encoding = self.pos_embedding[:,:seq_embedding.size(1), :]
        return self.dropout(seq_embedding + encoding)

class TransformerModel(nn.Module):
    '''
    The Transformer model serves as an Autoencoder.
    Currently, the output of the Transformer-Encoder (memory)
    is passed through a linear layer which acts as a bottleneck.
    The output of the Transformer-Decoder is passed through
    another linear layer to scale the output to the number of
    target variables
    '''
    
    def __init__(self,
                 d_model: int,
                 d_latent: int,
                 maxlen: int,
                 num_src_variables: int,
                 num_tgt_variables: int,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super(TransformerModel, self).__init__()      
        self.d_model = d_model
        self.d_latent = d_latent
        self.maxlen = maxlen
        
        # Initialize layers       
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)  
        
        self.latentEncoder = nn.Linear(maxlen*d_model, d_latent) 
        self.latentDecoder = nn.Linear(d_latent, maxlen*d_model) 
        
        # Output linear layer
        self.linear = nn.Linear(d_model, num_tgt_variables)
        
        # Initialize helper modules
        self.src_embedding = VarSeriesEmbedding(num_src_variables, d_model)
        self.tgt_embedding = VarSeriesEmbedding(num_tgt_variables, d_model)    
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Initialize helper tokens
        self.SOS = np.ones((1,3))*-2
        self.EOS = np.ones((1,3))*-3

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src_pad: Tensor,
                tgt_pad: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
         
        # Get (bottleneck) memory
        memory = self.encode(src_pad,
                             src_mask,
                             src_padding_mask)
        # Get output
        outs = self.decode(tgt_pad,
                           tgt_mask,
                           None, # memory_mask
                           tgt_padding_mask,
                           memory)
               
        # Scale output dim to the number of variables of the dataset
        outs = self.linear(outs)
        
        return outs
    
    def encode(self,
               src_pad: Tensor,
               src_mask: Tensor = None, 
               src_padding_mask: Tensor = None):  
        bz, maxlen, _ = src_pad.shape
        src_emb = torch.zeros((bz, maxlen, self.d_model)) 
        
        # Encode input batch into the embedding
        for batch in range(bz):
            src_emb[batch,:] = torch.stack([self.src_embedding(i) for i in src_pad[batch]])
        
        # Add positional encoding
        src_emb = self.positional_encoding(src_emb)
        
        # Get Transformer-Encoder output
        memory = self.transformer.encoder(src = src_emb,
                                          mask = src_mask,
                                          src_key_padding_mask = src_padding_mask)
        # Create Bottleneck
        memory = self.latentEncoder(memory.reshape((bz, self.maxlen*self.d_model)))          

        return memory
     
    def decode(self,
               tgt: Tensor,
               tgt_mask: Tensor, 
               memory_mask: Tensor,
               tgt_padding_mask: Tensor,
               memory: Tensor):
        bz, seqlength, _ = tgt.shape
        tgt_emb = torch.zeros((bz, seqlength, self.d_model)) 
        
        # Encode input batch into the embedding
        for batch in range(bz):
            tgt_emb[batch,:] = torch.stack([self.tgt_embedding(i) for i in tgt[batch]])
        
        # Add positional encoding
        tgt_emb = self.positional_encoding(tgt_emb) 
        
        # Decompress bottleneck
        memory = self.latentDecoder(memory).reshape((bz, self.maxlen, self.d_model))
        
        return self.transformer.decoder(tgt = tgt_emb,
                                        memory = memory,
                                        tgt_mask = tgt_mask,
                                        memory_mask = memory_mask,
                                        tgt_key_padding_mask = tgt_padding_mask)
    
    def greedy_decode(self,
                      memory: Tensor,
                      maxlen: int,
                      num_variables: int,
                      SOS: int,
                      EOS: int):
        '''
        Decodes a point from the latent space (memory) into 
        a time series without using a given masked output sequence
        as during training. 
        '''
        #print('Start greedy_decode() ...')
        
        SOS = np.ones((1,num_variables))*SOS[0]
        EOS = np.ones((1,num_variables))*EOS[0]

        ys = torch.FloatTensor(np.array([SOS])).to(DEVICE)
        # ys shape: (bz = 1, len = 1, num_variables)
        for i in range(maxlen-2):
            #print(f'position i = {i}')
            
            # Generate square subsequent mask
            tgt_mask = (torch.triu(torch.ones((ys.shape[1], ys.shape[1]), device=DEVICE)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
            tgt_mask.to(DEVICE)
            
            # Decode using the given memory and the current target+mask
            out = self.decode(ys,
                              tgt_mask = tgt_mask,
                              memory_mask = None,
                              tgt_padding_mask = None,
                              memory = memory)
            # Scale output to number of target variables
            out = self.linear(out) # shape: (bz, seqlength, num_var)
            
            # Pick the last element of the prediction
            # and append it to the target sequence
            out = torch.narrow(out, 1, -1 ,1) # dim, start, length
            ys = torch.cat([ys, out], dim=1)
                 
            if out == EOS:  
                break
            
        ys = torch.cat([ys, torch.FloatTensor(np.array([EOS]))], dim=1)
            
        return ys
    
    def sample(self, bz: int = 1):
        '''
        Samples a random point from the latent space.
        Currently, values are uniformly sampled 
        from the interval [-2,2)
        '''
        #memory = torch.rand((bz, self.maxlen, self.d_model))
        memory = torch.rand((bz, self.d_latent))
        upperl, lowerl = 2, -2
        memory = (lowerl - upperl)*memory + upperl
        return memory.requires_grad_()
    
class Discriminator(nn.Module):
    '''
    The Discriminator network of the GAN architecture
    consists of a Transformer-Encoder
    with the same hyperparameters as the Autoencoder-Transformer 
    and a FFN projecting to only one neuron to make a binary prediction.
    '''
    def __init__(self,                
                 d_model: int,
                 d_latent: int,
                 maxlen: int,
                 num_src_variables: int,
                 num_encoder_layers: int = 6,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        
        super(Discriminator, self).__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.maxlen = maxlen
        
        # Initialize layers
        self.transformer = Transformer(d_model=d_model,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_encoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True) 
        
        self.latentEncoder = nn.Linear(maxlen*d_model, self.d_latent) 
        self.linear = nn.Linear(self.d_latent, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize helper modules
        self.src_embedding = VarSeriesEmbedding(num_src_variables, d_model)   
        self.positional_encoding = PositionalEncoding(d_model)
        
        # (Don't forget batch norm instead of layer norm)
        
    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self,
               src_pad: Tensor,
               src_mask: Tensor = None, 
               src_padding_mask: Tensor = None):
        bz, maxlen, _ = src_pad.shape
        src_emb = torch.zeros((bz, maxlen, self.d_model)) 
            
        # Encode input batch into the embedding
        for batch in range(bz):
            src_emb[batch,:] = torch.stack([self.src_embedding(i) for i in src_pad[batch]])
        
        # Add positional encoding
        src_emb = self.positional_encoding(src_emb)

        # Get Transformer-Encoder output
        memory = self.transformer.encoder(src = src_emb,
                                          mask = src_mask,
                                          src_key_padding_mask = src_padding_mask)
        # Create Bottleneck
        memory = self.latentEncoder(memory.reshape((bz, self.maxlen*self.d_model)))   
        
        # Get prediction
        pred = self.linear(memory)
        pred = self.sigmoid(pred)
        
        return pred