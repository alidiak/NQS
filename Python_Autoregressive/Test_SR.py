#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:44:56 2020

@author: alex
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from NQS_pytorch import Op, Psi, O_local
import autograd_hacks

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

'''########### SR with Autograd_hacks & Matrices Implementation ############'''

# The autograd_hacks method is accurate and multiple orders of magnitude faster.

ppsi=psi_init(L,2*L,'vector')  # without mult, initializes params randomly

sb=ppsi.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)

[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi),O_local(b_field,s.numpy(),ppsi)
E_loc=np.sum(H_nn+H_b,axis=1)
lambduh=1
                       
'''#### Defining a more succinct algorithm to perform real and imag sep ####'''

def SR(ppsi,s,E_loc, lambduh=1, cutoff=1e-8): 
    
    E0=np.real(np.mean(E_loc))

    if ppsi.form=='vector':
        m_r=(1/ppsi.complex).squeeze()
        m_i=1j*m_r
    else:
        m_r=1/ppsi.real_comp(s).detach().numpy().squeeze()
        m_i=(np.ones([s.shape[0],1])*1j).squeeze()
    E_arg=(np.conj(E_loc)-np.conj(E0))
    
#    # Compute SR for real component
#    SR_alg(ppsi.real_comp,s,m_r,E_arg,lambduh,cutoff)
#    # Compute SR for real component
#    SR_alg(ppsi.imag_comp,s,m_i,E_arg,lambduh,cutoff)
    
    for ii in range(2):
        if ii==0:# Compute SR for real component
            model=ppsi.real_comp; m=m_r
        else:
            model=ppsi.imag_comp; m=m_i
            
        model.zero_grad()
        N_samples=s.size(0)
        
        if not hasattr(model,'autograd_hacks_hooks'):             
            autograd_hacks.add_hooks(model)
        outr=model(s)
        outr.mean().backward()
        autograd_hacks.compute_grad1(model) #computes grad per sample for all samples
        autograd_hacks.clear_backprops(model)
        pars=list(model.parameters())
            
        for param in pars:
            with torch.no_grad():
                if len(param.size())==2:#different mat mul rules depending on mat shape
                    ein_str="i,ijk->ijk"
                elif len(param.size())==1:
                    ein_str="i,ik->ik"
                if len(param.size())>1:
                    if param.size(1)>param.size(0): # pytorch flips matrix pattern sometimes
    # have to manually flip it back. (else get S=scalar for Nx1 matrix xforms-can't be right)
                        param.grad1=param.grad1.view(param.grad1.size(0),param.size(1),param.size(0))
                Ok=np.einsum(ein_str,m,param.grad1.numpy())
                if len(np.shape(Ok))==2:
                    Ok=Ok[:,:,None] 
    # Vector bias values do not agree with original method if this is not present
    # When present though, returns values similar in order to the other param grad values...
                Exp_Ok=np.mean(Ok,0) # conj(mean)=mean(conj)
    #T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
                T1=np.einsum("kni,imk->nm",np.conj(Ok),Ok.T)/N_samples
            # These are methods are equivalent! Good sanity check (einsum more versitile)
                St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
     # In NKT, cutoff of 1e-10 is used for S before inverting - inc. numerical stability?
     # if diagS< cutoff, S(i,i)=1 and s.row=s.col=0. 
                St[St<cutoff]=0
                l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
                S_inv=torch.tensor(np.linalg.pinv(St+l_reg),dtype=torch.float) # S^-1 term with reg
                force=torch.einsum(ein_str,torch.tensor(np.real(2*E_arg*m)\
                ,dtype=torch.float),param.grad1).mean(0) # force/DE term
                # Compute SR 'gradient'
                if len(force.size())==1:  # deal with .mm issues when vector Mx1
                    param.grad=torch.mm(S_inv,force[:,None]).view(param.size()).detach() 
                else:
                    param.grad=torch.mm(S_inv,force).view(param.size()).detach()
    
    return
    
#def SR_alg(model,s,m,E_arg,lambduh, cutoff):
#    
#    model.zero_grad()
#    N_samples=s.size(0)
#    
#    if not hasattr(model,'autograd_hacks_hooks'):             
#        autograd_hacks.add_hooks(model)
#    outr=model(s)
#    outr.mean().backward()
#    autograd_hacks.compute_grad1(model) #computes grad per sample for all samples
#    autograd_hacks.clear_backprops(model)
#    pars=list(model.parameters())
#        
#    for param in pars:
#        with torch.no_grad():
#            if len(param.size())==2:#different mat mul rules depending on mat shape
#                ein_str="i,ijk->ijk"
#            elif len(param.size())==1:
#                ein_str="i,ik->ik"
#            if len(param.size())>1:
#                if param.size(1)>param.size(0): # pytorch flips matrix pattern sometimes
## have to manually flip it back. (else get S=scalar for Nx1 matrix xforms-can't be right)
#                    param.grad1=param.grad1.view(param.grad1.size(0),param.size(1),param.size(0))
#            Ok=np.einsum(ein_str,m,param.grad1.numpy())
#            if len(np.shape(Ok))==2:
#                Ok=Ok[:,:,None] 
## Vector bias values do not agree with original method if this is not present
## When present though, returns values similar in order to the other param grad values...
#            Exp_Ok=np.mean(Ok,0) # conj(mean)=mean(conj)
##T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
#            T1=np.einsum("kni,imk->nm",np.conj(Ok),Ok.T)/N_samples
#        # These are methods are equivalent! Good sanity check (einsum more versitile)
#            St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
# # In NKT, cutoff of 1e-10 is used for S before inverting - inc. numerical stability?
# # if diagS< cutoff, S(i,i)=1 and s.row=s.col=0. 
#            St[St<cutoff]=0
#            l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
#            S_inv=torch.tensor(np.linalg.pinv(St+l_reg),dtype=torch.float) # S^-1 term with reg
#            force=torch.einsum(ein_str,torch.tensor(np.real(2*E_arg*m)\
#            ,dtype=torch.float),param.grad1).mean(0) # force/DE term
#            # Compute SR 'gradient'
#            if len(force.size())==1:  # deal with .mm issues when vector Mx1
#                param.grad=torch.mm(S_inv,force[:,None]).view(param.size()).detach() 
#            else:
#                param.grad=torch.mm(S_inv,force).view(param.size()).detach()
#    
#    return
    
''' ############### Explicit Testing of the SR func ####################### '''

ppsi=psi_init(L,2*L,'euler')  # without mult, initializes params randomly

N_iter=300
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

    ppsi.apply_grad(lr=0.1)

# SR definitely takes far longer than SGD

plt.figure
plt.plot(range(N_iter),E)
plt.xlabel('Iteration number')
plt.ylabel('Energy')








''' ### First Matrix Implementation ###'''

#ppsi=psi_init(L,2*L,'vector')  # without mult, initializes params randomly
#
#sb=ppsi.sample_MH(burn_in,spin=0.5)
#s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)
#
#[H_nn, H_b]=O_local(nn_interaction,s.numpy(),ppsi),O_local(b_field,s.numpy(),ppsi)
#E_loc=np.sum(H_nn+H_b,axis=1)
#lambduh=1
#
#E0=np.real(np.mean(E_loc))
#
#ppsi.real_comp.zero_grad()
#ppsi.imag_comp.zero_grad()
#
#if not hasattr(ppsi.real_comp,'autograd_hacks_hooks'):             
#    autograd_hacks.add_hooks(ppsi.real_comp)
#outr=ppsi.real_comp(s)
#outr.mean().backward()
#autograd_hacks.compute_grad1(ppsi.real_comp)
#autograd_hacks.clear_backprops(ppsi.real_comp)
#p_r=list(ppsi.real_comp.parameters())
#
#if not hasattr(ppsi.imag_comp,'autograd_hacks_hooks'): 
#    autograd_hacks.add_hooks(ppsi.imag_comp)
#outi=ppsi.imag_comp(s)
#outi.mean().backward()
#autograd_hacks.compute_grad1(ppsi.imag_comp)
#autograd_hacks.clear_backprops(ppsi.imag_comp)
#p_i=list(ppsi.imag_comp.parameters())
##
### get the multipliers (Ok=dpsi*m) and the energy gradients for each term
#if ppsi.form=='vector':
#    m_r=(1/ppsi.complex).squeeze()
#    m_i=1j*m_r
#else:
#    m_r=1/outr.detach().numpy().squeeze()
#    m_i=1j
#    
## term for the force
#E_arg=(np.conj(E_loc)-np.conj(E0))
##
##for param in p_r:
#    with torch.no_grad():
#        if len(param.size())==2:#different mat mul rules depending on mat shape
#            ein_str="i,ijk->ijk"
#        elif len(param.size())==1:
#            ein_str="i,ik->ik"
#        if len(param.size())>1:
#            if param.size(1)>param.size(0): # pytorch flips matrix pattern sometimes
## have to manually flip it back. (else get S=scalar for Nx1 matrix xforms-can't be right)
#                param.grad1=param.grad1.view(param.grad1.size(0),param.size(1),param.size(0))
#        Ok=np.einsum(ein_str,m_r,param.grad1.numpy())
#        if len(np.shape(Ok))==2:
#            Ok=Ok[:,:,None] 
## Vector bias values do not agree with original method if this is not present
## When present though, returns values similar in order to the other param grad values...
#        Exp_Ok=np.mean(Ok,0) # conj(mean)=mean(conj)
##        T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
#        T1=np.einsum("kni,imk->nm",np.conj(Ok),Ok.T)/N_samples
#        # These are methods are equivalent! Good sanity check (einsum more versitile)
#        St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
#        l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
#        S_inv=torch.tensor(np.linalg.pinv(St+l_reg),dtype=torch.float) # S^-1 term with reg
#        force=torch.einsum(ein_str,torch.tensor(np.real(2*E_arg*m_r)\
#        ,dtype=torch.float),param.grad1).mean(0) # force/DE term
#        # Compute SR 'gradient'
#        if len(force.size())==1:  # deal with .mm issues when vector Mx1
#            param.grad=torch.mm(S_inv,force[:,None]).view(param.size()).detach() 
#        else:
#            param.grad=torch.mm(S_inv,force).view(param.size()).detach()
#
#for param in p_i:
#    with torch.no_grad():
#        if len(param.size())==2:
#            ein_str="i,ijk->ijk"
#        elif len(param.size())==1:
#            ein_str="i,ik->ik"
#        if len(param.size())>1:
#            if param.size(1)>param.size(0): 
#                param.grad1=param.grad1.view(param.grad1.size(0),param.size(1),param.size(0))
#        Ok=np.einsum(ein_str,m_i,param.grad1.numpy())
#        if len(np.shape(Ok))==2:
#            Ok=Ok[:,:,None] 
#        Exp_Ok=np.mean(Ok,0) # conj(mean)=mean(conj)
#        T1=np.einsum("kni,imk->nm",np.conj(Ok),Ok.T)/N_samples
#        St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
#        l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
#        S_inv=torch.tensor(np.linalg.pinv(St+l_reg),dtype=torch.float) # S^-1 term with reg
#        force=torch.einsum(ein_str,torch.tensor(np.real(2*E_arg*m_i)\
#        ,dtype=torch.float),param.grad1).mean(0) # force/DE term
#        if len(force.size())==1:  # deal with .mm issues when vector Mx1
#            param.grad=torch.mm(S_inv,force[:,None]).view(param.size()).detach() 
#        else:
#            param.grad=torch.mm(S_inv,force).view(param.size()).detach()
