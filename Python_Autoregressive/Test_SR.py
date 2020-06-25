#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:44:56 2020

@author: alex
"""

import time
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from NQS_pytorch import Op, Psi, O_local, kron_matrix_gen
import itertools

# system parameters
b=0.5   # b-field strength
J=1     # nearest neighbor interaction strength
L = 6   # system size
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

''' #################### Stochastic Reconfiguration ####################### '''

ppsi=psi_init(L,2*L,'euler')  # without mult, initializes params randomly

sb=ppsi.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi),O_local(b_field,s.numpy(),ppsi)
E_loc=np.sum(H_nn+H_b,axis=1)
#lambduh=1
def SR(ppsi,s,E_loc, lambduh=1):
           
    E0=np.real(np.mean(E_loc))
    N_samples=s.size(0)
    
    outr=ppsi.real_comp(s)
    outi=ppsi.imag_comp(s)
    
    if ppsi.form=='vector':
        if np.all(ppsi.complex==0):
            ppsi.complex_out(s)
    
    p_r=list(ppsi.real_comp.parameters())
    p_i=list(ppsi.imag_comp.parameters())
    
    grad_list_i=copy.deepcopy(p_i)
    with torch.no_grad():
    
        for param in grad_list_i:
            param.copy_(torch.zeros_like(param))
            param.requires_grad=False
    # have to make a copy to record the gradient variable Ok and the force DE
    Ok_list_r=[]
    Ok_list_i=[]
    with torch.no_grad():
        grad_list_r=copy.deepcopy(p_r)
        for ii in range(len(p_r)):
            grad_list_r[ii].copy_(torch.zeros_like(p_r[ii]))
            grad_list_r[ii].requires_grad=False
            if len(p_r[ii].size())==1:
                sz1,sz2=p_r[ii].size(0),1    
            else:
                sz1,sz2=p_r[ii].size()
            Ok_list_r.append(np.zeros([N_samples,sz1,sz2],dtype=complex))
            
        grad_list_i=copy.deepcopy(p_i)
        for ii in range(len(p_i)):
            grad_list_i[ii].copy_(torch.zeros_like(p_i[ii]))
            grad_list_i[ii].requires_grad=False
            if len(p_i[ii].size())==1:
                sz1,sz2=p_i[ii].size(0),1    
            else:
                sz1,sz2=p_i[ii].size()
            Ok_list_i.append(np.zeros([N_samples,sz1,sz2],dtype=complex))
            
    # what we calculated the gradients should be
    for n in range(N_samples):
        
        ppsi.real_comp.zero_grad()
        ppsi.imag_comp.zero_grad()
    
        outr[n].backward(retain_graph=True) # retain so that buffers aren't cleared 
        outi[n].backward(retain_graph=True)     # and it can be applied again
        
        # get the multipliers (Ok=dpsi*m) and the energy gradients for each term
        if ppsi.form=='vector':
            m_r=(1/ppsi.complex[n])
            m_i=1j*m_r
        else:
            m_r=1/outr[n].detach().numpy()
            m_i=1j
        
        # term for the force
        E_arg=(np.conj(E_loc[n])-np.conj(E0))
              
        for kk in range(len(p_r)):
            with torch.no_grad():
                grad_list_r[kk]+=(p_r[kk].grad)*torch.tensor(\
                (2*np.real(E_arg*m_r)/N_samples),dtype=torch.float)
                Ok=p_r[kk].grad.numpy()*m_r
                # to deal with 1-dim params
                if len(np.shape(Ok))==1:
                    Ok=Ok[:,None]
    #            E_Ok=np.mean(Ok,1)[:,None]
    #            S=2*np.real(np.matmul(np.conj(Ok),Ok.T)-\
    #                        np.matmul(np.conj(E_Ok),E_Ok.T))
                Ok_list_r[kk][n]=Ok
    
        for kk in range(len(p_i)):
            with torch.no_grad():
                grad_list_i[kk]+=(p_i[kk].grad)*torch.tensor(\
                (2*np.real(E_arg*m_i)/N_samples),dtype=torch.float)
                Ok=p_i[kk].grad.numpy()*m_i
                if len(np.shape(Ok))==1:
                    Ok=Ok[:,None]
                Ok_list_i[kk][n]=Ok
    # unfortunately, must record Ok for each sample so an expectation <Ok> can be taken
    # This could be a memory/speed issue, but I don't see an obvious route around it
                
    S_list_r=[]
    for kk in range(len(Ok_list_r)):
        Exp_Ok=np.mean(Ok_list_r[kk],0)  # conj(mean)=mean(conj)
    #    T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
        T1=np.einsum('kni,imk->nm',np.conj(Ok_list_r[kk]),Ok_list_r[kk].T)/N_samples
        # These are methods are equivalent! Good sanity check
        St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
        l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
        S_list_r.append(St+l_reg) 
    
    S_list_i=[]
    for kk in range(len(Ok_list_i)):
        Exp_Ok=np.mean(Ok_list_i[kk],0) 
        T1=np.einsum('kni,imk->nm',np.conj(Ok_list_i[kk]),Ok_list_i[kk].T)/N_samples
        St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
        l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
        S_list_i.append(St+l_reg) 
    
    for kk in range(len(p_r)):
        S_inv=torch.tensor(np.linalg.pinv(S_list_r[kk]),dtype=torch.float) # have to inverse S
        if len(grad_list_r[kk].size())==1: # deal with .mm issues when vector Mx1        
            p_r[kk].grad=(torch.mm(S_inv,grad_list_r[kk][:,None]))\
            .view(p_r[kk].size()).detach()
        else:
            p_r[kk].grad=torch.mm(S_inv,grad_list_r[kk])
    
    for kk in range(len(p_i)):
        S_inv=torch.tensor(np.linalg.pinv(S_list_i[kk]),dtype=torch.float) # have to inverse S
        if len(grad_list_i[kk].size())==1: # deal with .mm issues when vector Mx1        
            p_i[kk].grad=(torch.mm(S_inv,grad_list_i[kk][:,None]))\
            .view(p_i[kk].size()).detach()
        else:
            p_i[kk].grad=torch.mm(S_inv,grad_list_i[kk])
        
    return 

''' ############### Explicit Testing of the SR func ####################### '''

ppsi=psi_init(L,2*L,'vector')  # without mult, initializes params randomly

N_iter=5
E=np.zeros([N_iter,1])

for kk in range(N_iter):

    sb=ppsi.sample_MH(burn_in,spin=0.5)
    s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)
    
    [H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi),O_local(b_field,s.numpy(),ppsi)
    E_loc=np.sum(H_nn+H_b,axis=1)
    E[kk] = np.mean(E_loc)
    
    start = time.time()
    SR(ppsi,s,E_loc)
    end = time.time()
    print(end - start)

    ppsi.apply_grad()

# SR definitely takes far longer than SGD

plt.figure
plt.plot(range(N_iter),E)
plt.xlabel('Iteration number')
plt.ylabel('Energy')
