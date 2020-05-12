#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:42:29 2020

@author: alex
"""

import torch
import torch.nn as nn
import numpy as np
from NQS_pytorch import Op, O_local, Psi
import itertools


'''
entry method inspired by NetKet-1.0.0
'''

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

'''##### Define Neural Networks and the form for psi (euler or vector) #####'''
H=2*L # hidden layer size

# creates an instance of the Sequential class nn.Sigmoid etc usually in forward section
real_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 

H2=2*L
imag_net=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),nn.Linear(H2,1),nn.Sigmoid()) 

# Test complex wavefunction object construction with modulus and angle
ppsi=Psi(real_net,imag_net, L, form='euler')
# OR
#ppsi=Psi(real_net,imag_net, L, form='vector')

'''##################### Testing O_local with L=3 #########################'''
spin=0.5    # routine may not be optimized yet for spin!=0.5
evals=2*np.arange(-spin,spin+1)
s=np.array(list(itertools.product(evals,repeat=L))) # each spin permutation

wvf=ppsi.complex_out(torch.tensor(s,dtype=torch.float))

S1=np.kron(np.kron(sigmax,np.eye(2)),np.eye(2))
S2=np.kron(np.kron(np.eye(2),sigmax),np.eye(2))
S3=np.kron(np.kron(np.eye(2),np.eye(2)),sigmax)

H_sx=b*(S1+S2+S3)
H_szsz=-J*(np.diag([-3,1,1,1,1,1,1,-3])) # from my ED Matlab code
H_tot=H_szsz+H_sx

E_sx=np.matmul(np.matmul(np.conjugate(wvf.T),H_sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_szsz=np.matmul(np.matmul(np.conjugate(wvf.T),H_szsz),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

H_nn=O_local(nn_interaction,s,ppsi)
H_b=O_local(b_field,s,ppsi)

print('For psi= \n', wvf, '\n\n the energy (using exact H) is: ', E_tot, '\n while that ' \
      'predicted with the O_local function is: ', np.sum(np.mean(H_b+H_nn,axis=0)), \
      '\n\n for the exact Sx H: ', E_sx, ' vs ',np.sum(np.mean(H_b,axis=0)), \
      '\n\n for exact SzSz H: ', E_szsz ,' vs ', np.sum(np.mean(H_nn,axis=0)))

print('\n\n also compare the predicted energy per sample of -sz*sz to the spins: '\
      'O_local sz*sz energy: \n', H_nn , '\n\n spins: \n', s )

#O_l=np.matmul(np.matmul(np.conjugate(wvf.T),Sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

''' Also checking with SxSx '''
sxsx=Op(np.kron(sigmax, sigmax))

L = 3
for i in range(L):  # Specify the sites upon which the operators act
    sxsx.add_site([i,(i+1)%L])

H_sxsx=O_local(sxsx,s,ppsi)

S1=np.kron(np.kron(sigmax,sigmax),np.eye(2))
S2=np.kron(np.kron(np.eye(2),sigmax),sigmax)
S3=np.kron(np.kron(sigmax,np.eye(2)),sigmax)

H_exact= S1+S2+S3+H_sx
E_exact=np.matmul(np.matmul(np.conjugate(wvf.T),H_exact),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

print('the energy of Sx*Sx (using exact H) is: ', E_exact, '\n with O_l it is: ' \
      ,np.sum(np.mean(H_sxsx+H_b,axis=0)) )

