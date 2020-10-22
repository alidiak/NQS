#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:26:40 2020

@author: alex
"""

import itertools
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from NQS_pytorch import Psi, Op, kron_matrix_gen

# system parameters
b=0.0   # b-field strength
J= 1     # nearest neighbor interaction strength
L = 6   # system size

datatype=torch.double

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

spin=0.5    # routine may not be optimized yet for spin!=0.5
evals=2*np.arange(-spin,spin+1)

if L<=14:
    # this tool creates the full Hamiltonian and is used for energy comparison 
    # practical in use of L<=16, otherwise memory requirements can become an issue
    op_list=[] 
    for ii in range(2): # make sure to change to match Hamiltonian entered above
        op_list.append(sigmaz)
    H_szsz=-J*kron_matrix_gen(op_list,len(evals),L,'periodic').toarray()
    
    op_list2=[]
    op_list2.append(sigmax)
    H_sx=b*kron_matrix_gen(op_list2,len(evals),L,'periodic').toarray()
    
    H_tot=H_szsz+H_sx

    min_E=np.min(np.linalg.eigvals(H_tot))
    
    s2=torch.tensor(list(itertools.product(evals,repeat=L)),dtype=datatype)

'''##### Define Neural Networks and the form for Psi (euler or vector) #####'''
H=2*L # hidden layer size

# creates an instance of the Sequential class nn.Sigmoid etc usually in forward section
real_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,1), nn.LogSigmoid())#,nn.Softplus())#,nn.Sigmoid()) 
#autoreg_real_net=nn.Sequential(nn.Linear(L,H), # nn.Sigmoid(), nn.Linear(H,H), \
#    nn.Sigmoid(), nn.Linear(H,len(evals)*L)) # for autoregressive final layer must be nevals*L
#autoreg_imag_net=nn.Sequential(nn.Linear(L,H), #nn.Sigmoid(), nn.Linear(H,H), \
#    nn.Sigmoid(), nn.Linear(H,len(evals)*L)) 
#real_net=nn.Sequential(nn.Linear(L,1))#,nn.Softplus())#,nn.Sigmoid()) 
# Always be careful of activation layers that result in exactly 0. (1/Psi->nan) in O_loc 

H2=2*L
imag_net=nn.Sequential(nn.Linear(L,H2),nn.Sigmoid(),nn.Linear(H2,1,bias=False))#,nn.Sigmoid()) 

#imag_net=nn.Sequential(nn.Linear(L,1,bias=False))#,nn.Sigmoid(),nn.Linear(H2,1,bias=False),nn.Softplus())#,nn.Sigmoid()) 
# always set bias=False for imaginary comp if using exponential or euler forms 
# otherwise there is a strong bias term that is an artifact (& leads to singular values)



# Test complex wavefunction object construction with modulus and angle
ppsi=Psi(real_net,imag_net, L, form='euler',dtype=datatype)
#ppsi=Psi(real_net,imag_net, L, form='exponential',dtype=datatype)
# OR
#ppsi=Psi(real_net,imag_net, L, form='vector',dtype=datatype)
#ppsi=Psi(real_net,0, L, form='real',dtype=datatype)

'''################## Optimization/Simulation Routine ######################'''
# Enter simulation hyper parameters
N_iter=300
N_samples=10000
burn_in=1000
lr=0.1
real_time_plot=True
exact_energy=True

# S regularization parameters if using SR
lambduh0, b, lambduh_min = 100, 0.95, 1e-4

# make an initial s
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=datatype)

if real_time_plot:
    plt.figure()
    plt.axis([0, N_iter, min_E-0.5, L])
    plt.axhline(y=min_E,color='r',linestyle='-')

energy_n=np.zeros([N_iter,1])
for n in range(N_iter):
    
    # before doing the actual sampling, we should do a burn in
    sb=ppsi.sample_MH(burn_in,spin=0.5)
    
    start = time.time()
    # Now we sample from the state and recast this as the new s, s0 so burn in is used
    s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=datatype)
    end = time.time(); print(end - start) # MC Sampling is the real bottleneck
    
    if exact_energy and L<=14: # if want to test the energy without sampling
        wvf=ppsi.complex_out(s2)
        E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)\
        /(np.matmul(np.conjugate(wvf.T),wvf))
        
        s=s2
        # Need sampling, as s2 will have low prob states of Psi disproportionately represented
        # Get the energy at each iteration
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
        energy_per_sample = np.sum(H_nn+H_b,axis=1)
        energy_n[n]=E_tot
    else:
        # Get the energy at each iteration
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
        energy_per_sample=np.sum(H_nn+H_b,axis=1)
        energy_n[n] = np.real(np.mean(energy_per_sample))
    
    # calculate the energy gradient, updates pars in Psi object
#    ppsi.energy_gradient(s,energy_per_sample,energy_n[n]) # simple gradient descent
    l_iter=max(lambduh0*b**(n),lambduh_min)
    ppsi.SR(s,energy_per_sample, lambduh=l_iter)#, cutoff=1e-8)
    
    lr=lr*0.99
    ppsi.apply_grad(lr) # releases/updates parameters based on grad method (stored in pars.grad)

    if n%10==0:
        print('percentage of iterations complete: ', (n/N_iter)*100)
    
    if real_time_plot:
        if n>=1:
            plt.plot([n-1,n],[energy_n[n-1],energy_n[n]],'b-')
            plt.pause(0.05)
            plt.draw()

if not real_time_plot:
    plt.figure()
    if L<=14:
        plt.axhline(y=min_E,color='r',linestyle='-')
    plt.plot(range(N_iter),energy_n)
    plt.xlabel('Iteration number')
    plt.ylabel('Energy')
