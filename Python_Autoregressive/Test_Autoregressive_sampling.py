#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:15:18 2020

@author: alex
"""

import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NQS_pytorch import Psi, Op, O_local

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 3   # system size
burn_in=1000
N_samples=10000 # number of samples for the Monte Carlo chains

spin=0.5    # routine may not be optimized yet for spin!=0.5
evals=2*np.arange(-spin,spin+1)

# Define operators to use in the Hamiltonian
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

szsz = np.kron(sigmaz, sigmaz)

# initiate the operators and the matrix they are fed
nn_interaction=Op(-J*szsz)
b_field=Op(b*sigmax)

for i in range(L):  # Specify the sites upon which the operators act
    # specify the arbitrary sites which the operators will act on
    b_field.add_site([i])
    nn_interaction.add_site([i,(i+1)%L])

'''##### Define Neural Networks and initialization funcs for psi  #####'''

def psi_init(L, H=2*L, Form='euler'):
    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), 
                       nn.Linear(H,1))#,nn.Sigmoid()) 
    H2=round(H/2)
    toy_model2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),
                     nn.Linear(H2,1))#,nn.Sigmoid()) 

    ppsi=Psi(toy_model,toy_model2, L, form=Form)
    
    return ppsi


'''#################### Autoregressive Sampling ############################'''

ppsi=psi_init(L,2*L,'euler')  # without mult, initializes params randomly

sb=ppsi.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi),O_local(b_field,s.numpy(),ppsi)
E_loc=np.sum(H_nn+H_b,axis=1)


def sample_MH(self, N_samples, spin=None, evals=None, s0=None, rot=None):
    # need either the explicit evals or the spin 
    if spin is None and evals is None:
        raise ValueError('Either the eigenvalues of the system or the spin\
                         must be entered')
            
    # the rule for flipping/rotating a spin between it's eigenvalues
    if rot is None:
        if spin is None:
            dim=len(evals)
        else:
            dim = int(2*spin+1)
        rot =  2*np.pi/dim # assume a rotation that scales with # evals
        # note, can only rotate to 'intermediate/nearby' evals

    if evals is None:
        evals=2*np.arange(-spin,spin+1) # +1 is just so s=spin is included
        # times 2 is just the convention that's been used, spin evals of -1,1
    
    if s0 is None:
        s0=np.random.choice(evals,size=self.L)
    
    self.samples=np.zeros([N_samples,self.L])
    self.samples[0,:]=s0
    for n in range(N_samples-1):
        
        pos=np.random.randint(self.L) # position to change
        
        alt_state = self.samples[n,:].copy() # next potential state
        
        if np.random.rand()>=0.5:
            alt_state[pos] = np.real(np.exp(1j*rot)*alt_state[pos]) # flip next random position for spin
        else:
            alt_state[pos] = np.real(np.exp(-1j*rot)*alt_state[pos]) # same chance to flip other direction
        # Will have to generalize to complex vals sometime
        
        # Probabilty of the next state divided by the current
        prob = (np.square(np.abs(self.complex_out(torch.tensor(alt_state,dtype=torch.float)))))   \
        /(np.square(np.abs(self.complex_out(torch.tensor(self.samples[n,:],dtype=torch.float)))))
        
        A = min(1,prob) # Metropolis Hastings acceptance formula

        if A ==1: self.samples[n+1,:]=alt_state
        else: 
            if np.random.rand()<A: self.samples[n+1,:]=alt_state # accepting move with prob
            else: self.samples[n+1,:] = self.samples[n,:]
        
    return self.samples
