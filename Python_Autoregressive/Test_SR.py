#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:44:56 2020

@author: alex
"""

import copy
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import itertools
from NQS_pytorch import Op, Psi, kron_matrix_gen
import autograd_hacks

# system parameters
b=0.5   # b-field strength
J= 1   # nearest neighbor interaction strength
L = 2   # system size
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

s2=np.array(list(itertools.product(evals,repeat=L)))

'''##### Define Neural Networks and initialization funcs for psi  #####'''

def psi_init(L, H=2*L, Form='euler'):
    toy_model=nn.Sequential(nn.Linear(L,1))#,nn.Tanh(), 
#               nn.Linear(H,1))#,nn.ReLU())#,nn.Linear(H,1),nn.Tanh())
#    H2=H #round(H/2)
    toy_model2=nn.Sequential(nn.Linear(L,1))#,nn.Tanh(),
#        nn.Linear(H2,1),nn.Tanh())#,nn.Linear(H,1),nn.Tanh())#,nn.Sigmoid()) 

    ppsi=Psi(toy_model,toy_model2, L, form=Form)
    
    return ppsi

'''########### SR with Autograd_hacks & Matrices Implementation ############'''

# The autograd_hacks method is accurate and multiple orders of magnitude faster.

ppsi=psi_init(L,2*L,'vector')  # without mult, initializes params randomly

sb=ppsi.sample_MH(burn_in,spin=0.5)
s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)

[H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
E_loc=np.sum(H_nn+H_b,axis=1)
lambduh=1
                       
'''#### Defining a more succinct algorithm to perform real and imag sep ####'''

def SR(ppsi,s,E_loc, E0=None, lambduh=1):#, cutoff=1e-8): 
    
    if E0 is None:
        E0=np.real(np.mean(E_loc))
    N_samples=s.shape[0]
    
    if ppsi.form.lower()=='vector':
        if np.all(ppsi.complex==0): 
            ppsi.complex_out(s)
        m_r=(1/ppsi.complex).squeeze()
        m_i=1j*m_r
    elif ppsi.form.lower()=='euler' or ppsi.form.lower()=='exponential'\
         or ppsi.form.lower()=='real':
        if ppsi.form.lower()=='euler' or ppsi.form.lower()=='real':
            m_r=1/ppsi.real_comp(s).detach().numpy().squeeze()
        else:
            m_r=(np.ones([N_samples,1])).squeeze()
        m_i=(np.ones([N_samples,1])*1j).squeeze()  
        
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
        
        if not hasattr(model,'autograd_hacks_hooks'):             
            autograd_hacks.add_hooks(model)
        outr=model(s)
        outr.mean().backward()
        autograd_hacks.compute_grad1(model) #computes grad per sample for all samples
        autograd_hacks.clear_backprops(model)
        pars=list(model.parameters())
        
        for param in pars:
            with torch.no_grad():
                par_size=param.size() # record original param shape for reshaping
                Ok=np.einsum("i,ik->ik",m,param.grad1.view([N_samples,-1]).numpy())
                print('grad: ', param.grad1 )
                print('\n\n mult: ', m)
                Exp_Ok=np.mean(Ok,0)[:,None] # gives another axis, necessary for matmul
        #        T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
                T1=np.einsum("kn,mk->nm",np.conj(Ok),Ok.T)/N_samples
                # These are methods are equivalent! Good sanity check (einsum more versitile)
#                S=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
                # folowing same reg/style as senior design matlab code
#                print(np.matmul(np.conj(Exp_Ok),Exp_Ok.T))
                S=T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)
                S=S+1e-5*np.diag(np.diag(S))#+1e-5*np.eye(S.shape[0],S.shape[1])
                print('\n\n S: ', S)
                S_inv=np.linalg.inv(S)
#                S[S<cutoff]=0
#                l_reg=lambduh*np.eye(S.shape[0],S.shape[1])*np.diag(S) # regulation term
                # SVD Inversion alg
#                [U,D,VT]=np.linalg.svd(S+l_reg)
#                D=np.diag(1/D) # inverting the D matrix, for SVD, M'=V (D^-1) U.T = (U(D^-1)V.T).T
#                S_inv=torch.tensor(np.matmul(np.matmul(U,D),VT).T,dtype=torch.float)
#                S_inv=torch.tensor(np.linalg.pinv(S+l_reg),dtype=torch.float) # S^-1 term with reg
                force=torch.einsum("i,ik->ik",torch.tensor(np.real(2*E_arg*m)\
                ,dtype=torch.float),param.grad1.view([N_samples,-1])).mean(0) # force/DE term
                # Compute SR 'gradient'
                param.grad=torch.tensor(np.real(np.matmul(S_inv,force[:,None]\
                .detach().numpy())),dtype=torch.float).view(par_size).detach()   # matlab code similar
#                param.grad=torch.mm(S_inv,force[:,None]).view(par_size).detach()
    
    return S
    
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
#            par_size=param.size() # record original param shape for reshaping
#            Ok=np.einsum("i,ik->ik",m,param.grad1.view([N_samples,-1]).numpy())
#            Exp_Ok=np.mean(Ok,0)[:,None] # gives another axis, necessary for matmul
#    #        T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
#            T1=np.einsum("kn,mk->nm",np.conj(Ok),Ok.T)/N_samples
#            # These are methods are equivalent! Good sanity check (einsum more versitile)
#            S=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
#            S[S<cutoff]=0
#            l_reg=lambduh*np.eye(S.shape[0],S.shape[1])*np.diag(S) # regulation term
#            S_inv=torch.tensor(np.linalg.pinv(S+l_reg),dtype=torch.float) # S^-1 term with reg
#            force=torch.einsum("i,ik->ik",torch.tensor(np.real(2*E_arg*m)\
#            ,dtype=torch.float),param.grad1.view([N_samples,-1])).mean(0) # force/DE term
#            # Compute SR 'gradient'
#            param.grad=torch.mm(S_inv,force[:,None]).view(par_size).detach() 
#    
#    return

''' ############### Explicit Testing of the SR func ####################### '''

ppsi=psi_init(L,L,'euler')  # without mult, initializes params randomly
lambduh0, b, lambduh_min = 100, 0.9, 1e-4
lr=0.1; 

N_iter, N_samples=20, 1000
E=np.zeros([N_iter,1])

no_sample=True

plt.figure()
plt.axis([0, N_iter, min_E-0.5, round(L/2)])
plt.axhline(y=min_E,color='r',linestyle='-')

p_r=list(ppsi.real_comp.parameters())
p_i=list(ppsi.imag_comp.parameters())

p_r_list=[]
p_i_list=[]
for param in p_r:
    if len(param.size())==1:
        sz1,sz2=param.size(0),1    
    else:
        sz1,sz2=param.size()
    p_r_list.append(torch.zeros([N_iter,sz1,sz2],dtype=torch.float))
    p_i_list.append(torch.zeros([N_iter,sz1,sz2],dtype=torch.float))
    
for kk in range(N_iter):
#    start = time.time()
    if no_sample: # if want to test the energy without sampling
        s=torch.tensor(s2,dtype=torch.float)
        wvf=ppsi.complex_out(torch.tensor(s2,dtype=torch.float))
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s2),ppsi.O_local(b_field,s2)
        E_loc = np.sum(H_nn+H_b,axis=1)
        E_tot=np.matmul(np.matmul(np.conjugate(wvf.T),H_tot),wvf)\
        /(np.matmul(np.conjugate(wvf.T),wvf))
        E[kk]=E_tot
    else:
        sb=ppsi.sample_MH(burn_in,spin=0.5)
        s=torch.tensor(ppsi.sample_MH(N_samples,spin=0.5, s0=sb[-1]),dtype=torch.float)
        
        [H_nn, H_b]=ppsi.O_local(nn_interaction,s.numpy()),ppsi.O_local(b_field,s.numpy())
        E_loc=np.sum(H_nn+H_b,axis=1)
        E[kk] = np.mean(E_loc)
              
    #    lr=lr*0.99
        
    l_iter=lambduh0*b
    l=max(l_iter,lambduh_min)
    S=SR(ppsi,s,E_loc, E[kk], lambduh=l)
    
    p_r=list(ppsi.real_comp.parameters())
    p_i=list(ppsi.imag_comp.parameters())
    
    for n in range(len(p_r)):
            p_r_list[n][kk]=p_r[n]#.detach().numpy()
            p_i_list[n][kk]=p_i[n]#.detach().numpy()
    
    ppsi.apply_grad(lr)
    
#    end = time.time()
#    print('iteration #: ', kk) # Seems that the sampling is the real bottleneck
    if kk>=1:
        plt.plot([kk-1,kk],[E[kk-1],E[kk]],'b-')
        plt.pause(0.1)
        plt.draw()
        
        
plt.figure()
nonzero_pr=p_r_list[0][p_r_list[0]!=0].detach().numpy()
plt.plot(range(len(nonzero_pr[0::2])),nonzero_pr[0::2],nonzero_pr[1::2])
    
plt.figure()
plt.plot(range(N_iter),E)
plt.axhline(y=min_E,color='r',linestyle='-')
plt.xlabel('Iteration number')
plt.ylabel('Energy')

p_r=list(ppsi.real_comp.parameters())
p_i=list(ppsi.imag_comp.parameters())

'''### First Matrix Implementation (Useful for explicit testing, coding) ###'''

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
#
''' ############ The Corrected Alternative ################### '''
#
#for param in p_r:
#    with torch.no_grad():      
#        par_size=param.size() # record original param shape for reshaping
#        Ok=np.einsum("i,ik->ik",m_r,param.grad1.view([N_samples,-1]).numpy())
#        Exp_Ok=np.mean(Ok,0)[:,None] # gives another axis, necessary for matmul
##        T1=np.tensordot(np.conj(Ok_list[kk]),Ok_list[kk].T, axes=((0,2),(2,0)))/N_samples
#        T1=np.einsum("kn,mk->nm",np.conj(Ok),Ok.T)/N_samples
#        # These are methods are equivalent! Good sanity check (einsum more versitile)
#        St=2*np.real(T1-np.matmul(np.conj(Exp_Ok),Exp_Ok.T)) # the S+c.c. term
#        l_reg=lambduh*np.eye(St.shape[0],St.shape[1])*np.diag(St) # regulation term
#        S_inv=torch.tensor(np.linalg.pinv(St+l_reg),dtype=torch.float) # S^-1 term with reg
#        force=torch.einsum("i,ik->ik",torch.tensor(np.real(2*E_arg*m_r)\
#        ,dtype=torch.float),param.grad1.view([N_samples,-1])).mean(0) # force/DE term
#        # Compute SR 'gradient'
#        param.grad=torch.mm(S_inv,force[:,None]).view(par_size).detach() 
#
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
