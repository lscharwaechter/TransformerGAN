#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 23:42:01 2023

@author: Leon Scharwächter
"""

import numpy as np
import matplotlib.pyplot as plt

D_G_z1_list = np.load('D_G_z1_list.npy')
D_G_z2_list = np.load('D_G_z2_list.npy')
D_x_list = np.load('D_x_list.npy')

train_AE_errs = np.load('train_AE_errs.npy')
valid_errs = np.load('valid_errs.npy')

train_GEN_errs = np.load('train_GEN_errs.npy')
train_DISC_fake_errs = np.load('train_DISC_fake_errs.npy')
train_DISC_real_errs = np.load('train_DISC_real_errs.npy')

stats = np.load('stats.npy',allow_pickle=True)

start_ = 0
end_ = -1

# Plot the errors
plt.figure(1, figsize=(1,1))
fig, ax1 = plt.subplots(1,1)
#fig.subplots_adjust(hspace=0.5)
fig.suptitle('Train- and Test-Error')
ax1.plot(train_AE_errs[start_:end_])
ax1.plot(valid_errs[start_:end_])
plt.legend(["Train Error","Test Error"])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

plt.figure(2, figsize=(1,1))
fig, ax1 = plt.subplots(1,1)
#fig.subplots_adjust(hspace=0.5)
fig.suptitle('GAN Training')
ax1.plot(train_GEN_errs[start_:end_])
ax1.plot(train_DISC_real_errs[start_:end_])
ax1.plot(train_DISC_fake_errs[start_:end_])
plt.legend(["Gen Error","Disc Real Error","Disc Fake Error"])
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

plt.figure(3, figsize=(1,1))
fig, ax1 = plt.subplots(1,1)
#fig.subplots_adjust(hspace=0.5)
fig.suptitle('Discriminator Output')
ax1.plot(D_x_list[start_:end_])
ax1.plot(D_G_z1_list[start_:end_])
ax1.plot(D_G_z2_list[start_:end_])
plt.legend(["D(x)","D(G(z))","D(G(z))"])
plt.xlabel("Epochs")
plt.ylabel("Output")
plt.show()

print(stats)
