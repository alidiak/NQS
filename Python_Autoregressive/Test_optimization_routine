#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:26:40 2020

@author: alex
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NQS_pytorch import Psi, Op, O_local

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 3   # system size

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

'''##### Define Neural Networks and the form for Psi (euler or vector) #####'''
H=2*L # hidden layer size

# creates an instance of the Sequential class nn.Sigmoid etc usually in forward section
real_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 

H2=2*L
imag_net=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),nn.Linear(H2,1),nn.Sigmoid()) 

# Test complex wavefunction object construction with modulus and angle
#ppsi=Psi(real_net,imag_net, L, form='euler')
# OR
ppsi=Psi(real_net,imag_net, L, form='vector')

'''################## Optimization/Simulation Routine ######################'''
# Enter simulation hyper parameters
N_iter=200
N_samples=1000
burn_in=200
lr=0.1

# make an initial s
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)

energy_n=np.zeros([N_iter,1])
for n in range(N_iter):
    
    # Get the energy at each iteration
    H_nn=O_local(nn_interaction,s.numpy(),ppsi)
    H_b=O_local(b_field,s.numpy(),ppsi)
    energy_per_sample = np.sum(H_nn+H_b,axis=1)
    energy=np.mean(energy_per_sample)
    energy_n[n]=np.real(energy)

    # apply the energy gradient, updates pars in Psi object
    ppsi.apply_energy_gradient(s,energy_per_sample,energy,lr)

    # before doing the actual sampling, we should do a burn in
    sburn=ppsi.sample_MH(burn_in,spin=0.5)

    # Now we sample from the state and recast this as the new s, s0 so burn in is used
    s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5,s0=sburn[-1,:]),dtype=torch.float)

    if n%10==0:
        print('percentage of iterations complete: ', (n/N_iter)*100)

plt.figure
plt.plot(range(N_iter),energy_n)
plt.xlabel('Iteration number')
plt.ylabel('Energy')
