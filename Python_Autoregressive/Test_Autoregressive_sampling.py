#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:15:18 2020

@author: alex
"""

import numpy as np
from made import MADE
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NQS_pytorch import Psi, Op, O_local

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 3   # system size
N_samples=1000 # number of samples for the Monte Carlo chains

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

#def psi_init(L, H=2*L, Form='euler'):
#    toy_model=nn.Sequential(nn.Linear(L,H),nn.Sigmoid(), 
#                       nn.Linear(H,L),nn.Softmax()) 
#    H2=round(H/2)
#    toy_model2=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),
#                     nn.Linear(H2,L),nn.Softmax()) 
#
#    ppsi=Psi(toy_model,toy_model2, L, form=Form)
#    
#    return ppsi

# Neural Autoregressive Density Estimators (NADEs) output a list of 
# probabilities equal in size to the input. Futhermore, a softmax function is 
# handy as it ensures the output is probability like aka falls between 0 and 1.
# For a simple spin 1/2, only a single Lx1 output is needed as p(1)=1-P(-1),
# but with increasing number of eigenvalues, the probability and output becomes
# more complex. 

hidden_layer_sizes=[10,14]
nout=len(evals)*L
model_r=MADE(L,hidden_layer_sizes, nout, num_masks=1, natural_ordering=True)
#The MADE coded by Andrej Karpath uses Masks to ensure that the
# autoregressive property is upheld. natural_ordering=False 
# randomizes autoregressive ordering, while =True makes the autoregressive 
# order p1=f(s_1),p2=f(s_2,s_1)

model_i=MADE(L,hidden_layer_sizes, nout, num_masks=1, natural_ordering=True)

'''#################### Autoregressive Sampling ############################'''

N_samples=10

# initialize/start with a random vector
s0=torch.tensor(np.random.choice(evals,[N_samples,L]),dtype=torch.float)

outr=model_r(s0)
outi=model_i(s0)

# random ordering each time we sample
#order=np.random.permutation(range(L))

Ppsi=torch.ones([N_samples]); # the full Psi is a product of the conditionals, making a running product easy

new_s=torch.zeros_like(s0) 

# Begin the autoregressive sampling routine
for ii in range(0,nout,2):
    
    # normalized probability/wavefunction
    vi=outr[:,ii]
    vi2=outr[:,ii+1]
    exp_vi=torch.exp(vi) # unnorm prob of either 1 or 0 
    exp_vi2=torch.exp(vi2) 
    norm_const=torch.sqrt((torch.pow(torch.abs(exp_vi),2)+\
                           torch.pow(torch.abs(exp_vi2),2)))
    psi1=exp_vi/norm_const
    psi2=(exp_vi2)/norm_const
    
    born_psi1=torch.pow(torch.abs(psi1),2)
    born_psi2=torch.pow(torch.abs(psi2),2)
    
    # satisfy the normalization condition?
    assert torch.all(born_psi1+born_psi2-1<1e-6), "Psi not normalized correctly"

    # Now let's sample from the binary distribution
    rands=torch.rand(N_samples)
    
    new_s[:,round(ii/2)]=torch.ones([N_samples])+2*(rands<=born_psi1)*evals[0] 
                        #+(rands>born_psi2)*evals[1]
        
    # Accumulating Psi
    psi_s=(rands<born_psi1)*1*psi1+(rands>born_psi2)*1*psi2
    Ppsi=Ppsi*psi_s


        