#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:42:29 2020

@author: alex
"""

import torch
import torch.nn as nn
import numpy as np
from NQS_pytorch import Op, O_local, Psi, kron_matrix_gen
import itertools

'''
entry method inspired by NetKet-1.0.0
'''

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 10  # system size
# L=14 uses about 8Gb of system memory, so it's a suggested max if running many other programs

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
real_net=nn.Sequential(nn.Linear(L,1))#, nn.Sigmoid())#, nn.Linear(H,1),nn.Sigmoid()) 
#real_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 

H2=2*L
imag_net=nn.Sequential(nn.Linear(L,1))#,nn.Sigmoid()) #,nn.Linear(H2,1),nn.Sigmoid()) 
#imag_net=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),nn.Linear(H2,1),nn.Sigmoid()) 

# Test complex wavefunction object construction with modulus and angle
ppsi=Psi(real_net,imag_net, L, form='euler')
# OR
#ppsi=Psi(real_net,imag_net, L, form='vector')

'''##################### Testing O_local with L=3 #########################'''
spin=0.5    # routine may not be optimized yet for spin!=0.5
evals=2*np.arange(-spin,spin+1)
s=np.array(list(itertools.product(evals,repeat=L))) # each spin permutation

wvf=ppsi.complex_out(torch.tensor(s,dtype=torch.float))

# make the list of operators that act on neighbors
op_list=[]
for ii in range(2):
    op_list.append(sigmaz)
H_szsz=-J*kron_matrix_gen(op_list,len(evals),L,'periodic').toarray()

op_list2=[]
op_list2.append(sigmax)
H_sx=b*kron_matrix_gen(op_list2,len(evals),L,'periodic').toarray()

# H_szsz=-J*(np.diag([3,-1,-1,-1,-1,-1,-1,3])) # from my ED Matlab code
H_tot=H_szsz+H_sx

E_sx=np.matmul(np.matmul(np.conjugate(wvf.T),H_sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_szsz=np.matmul(np.matmul(np.conjugate(wvf.T),H_szsz),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))
E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

N_samples=10000
sn=ppsi.sample_MH(N_samples,spin=0.5)
H_nn=O_local(nn_interaction,sn,ppsi)
H_b=O_local(b_field,sn,ppsi)

print('For psi= \n', wvf, '\n\n the energy (using exact H) is: ', E_tot, '\n while that ' \
      'predicted with the O_local function is: ', np.sum(np.mean(H_b+H_nn,axis=0)), \
      '\n\n for the exact Sx H: ', E_sx, ' vs ',np.sum(np.mean(H_b,axis=0)), \
      '\n\n for exact SzSz H: ', E_szsz ,' vs ', np.sum(np.mean(H_nn,axis=0)))

#print('\n\n also compare the predicted energy per sample of -sz*sz to the spins: '\
#      'O_local sz*sz energy: \n', H_nn , '\n\n spins: \n', s )

#O_l=np.matmul(np.matmul(np.conjugate(wvf.T),Sx),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

''' Also checking with SxSx '''
sxsx=Op(np.kron(sigmax, sigmax))

for i in range(L):  # Specify the sites upon which the operators act
    sxsx.add_site([i,(i+1)%L])

H_sxsx=O_local(sxsx,sn,ppsi)

op_list=[]
for ii in range(2):
    op_list.append(sigmax)
H_sxsx2=kron_matrix_gen(op_list,len(evals),L,'periodic').toarray()

H_exact= H_sxsx2+H_sx
E_exact=np.matmul(np.matmul(np.conjugate(wvf.T),H_exact),wvf)/(np.matmul(np.conjugate(wvf.T),wvf))

print('\n\n the energy of Sx*Sx (using exact H) is: ', E_exact, '\n with O_l it is: ' \
      ,np.sum(np.mean(H_sxsx+H_b,axis=0)) )

''' Ensuring that <psi|H|psi> = \sum_s |psi(s)|^2 e_loc(s)   '''

H_sxsx_ex=O_local(sxsx,s,ppsi)
H_sx_ex=O_local(b_field,s,ppsi)
O_loc_analytic= np.sum(np.matmul((np.abs(wvf.T)**2),(H_sxsx_ex+H_sx_ex)))\
 /(np.matmul(np.conjugate(wvf.T),wvf))

print('\n\n Energy using O_local in the analytical expression: ',O_loc_analytic, \
      '\n vs. that calculated with matrices: ', E_exact )

