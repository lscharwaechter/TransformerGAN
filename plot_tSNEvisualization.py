# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:51:11 2023

@author: Leon Scharwächter
"""

import dataset
#import dataset_full
import transformerGAN as tg
#import convolutionalAE as cAE
from utils import MinMaxScaler3D, padding, create_masks

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import pandas as pd
from random import sample

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

torch.manual_seed(0)

# Get dataset
maxlen = 52
train_data, test_data = dataset.getDataset()
train_labels, test_labels = dataset.getLabels()
#train_data, test_data = dataset_full.getDataset(maxlen-1, train_size = 0.83)

## Normalize feature vectors
scaler = MinMaxScaler3D(feature_range=(-1,1))
train_data = scaler.fit_transform(train_data.clone())
test_data = scaler.transform(test_data.clone())
#test_data = test_data.numpy()

#%%

# Reshape data
test_data = test_data.reshape((test_data.shape[0],
                           	test_data.shape[1]*test_data.shape[2]))
test_data = np.float16(test_data)

#data_test = arff.loadarff('NATOPS_TEST.arff')
#df_test = pd.DataFrame(data_test[0])
#labels = df_test['classAttribute'].sort_values().unique()

#%%

# Load the model and sample artificial data points

model_name = 'rami'
num_src_variables, num_tgt_variables, d_model = 24, 24, 24
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

# Load state
transformer.load_state_dict(
        torch.load(os.path.join(os.getcwd(),model_name + ".pt"),map_location=torch.device('cpu')))

# Sample data points
bz = 50
SOS = -3
memory_fake = transformer.sample(bz)
src_fake = transformer.greedy_decode_bz(memory_fake, maxlen, num_tgt_variables, SOS)
src_fake = src_fake[:,1:,:].detach().numpy() #exclude SOS

#Reshape fake data
src_fake = src_fake.reshape((src_fake.shape[0],
                             src_fake.shape[1]*src_fake.shape[2]))
src_fake = np.float16(src_fake)


#%%

# Plot t-SNE representation

X = np.concatenate((test_data, src_fake))
labels = [0 if i < test_data.shape[0] else 1 for i in range(X.shape[0])]

pca = PCA(n_components=50)
X = pca.fit_transform(X)

X = np.float16(X)

X_tSNE = TSNE(n_components=2, learning_rate='auto',
          	init='pca', perplexity=10).fit_transform(X)

X_tSNE_true = X_tSNE[:test_data.shape[0]]
X_tSNE_fake = X_tSNE[test_data.shape[0]:]

plt.figure(figsize=(16,10))
plt.scatter(X_tSNE_true[:,0],X_tSNE_true[:,1],s=200,alpha=0.4)
plt.scatter(X_tSNE_fake[:,0],X_tSNE_fake[:,1],s=200,alpha=0.4)
plt.xticks([], [])
plt.yticks([], [])


#%%

# Plot t-SNE representation (test data only)
'''
X = test_data
pca = PCA(n_components=50)
X = pca.fit_transform(X)

X = np.float16(X)

X_tSNE = TSNE(n_components=2, learning_rate='auto',
          	init='pca', perplexity=10).fit_transform(X)

plt.figure(figsize=(16,10))
cmap = plt.cm.get_cmap('tab10', 6)
plt.scatter(X_tSNE[:,0],X_tSNE[:,1],s=200,alpha=0.5,c=test_labels,cmap=cmap)
plt.xticks([], [])
plt.yticks([], [])
'''
