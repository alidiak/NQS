#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:20:16 2020

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import os, os.path



def O_local(basis, sites, ops,s):  
    '''
    mm=int(size/2)
    for mm in range(len(sites)):
        site=sites[mm]
        op = ops[mm]
        alt_state = s.copy()
        # alt_state[:,mm] = -1*alt_state[:,mm] # The effect of sigmax_mm (flips the mmth spin)
        # OPS times altstate at mm
    O_loc[t] = np.mean(ppf.ppsi(W_t,alt_state.T)/ppf.ppsi(W_t,temp_chains.T)) '''
    return O_loc


#def basismap(N, max_spin): # returns the basis map/permutations
#    #here N is the lattice size
#    
#    D = 2*max_spin+1
#    perms = np.zeros([(D**N),N])
#    jarray = np.arange(0,(D**N))
#    j = np.floor(jarray/D)
#    perms[:,0] = 2*(max_spin-jarray+D*j)
#    
#    for n in range(1,N):
#        perms[:,n] = 2*(max_spin - j + D*np.floor(j/D))
#        j = np.floor(j/D)
#        
#    return perms
    
#def ppsi(W, spins, machine='RbmSpin', a=None,b=None):
#    
#    N = np.size(spins) # when psi is to be caclulated for only 1 spin config
#    
#    ### RBM ALG
#    if machine=='RbmSpin':
#        if b is not None: theta = np.squeeze(b)+np.matmul(np.transpose(W),spins)
#        else: theta = np.matmul(np.transpose(W),spins)
#
#        prodcosh = np.exp(np.sum(np.log(np.cosh(theta)),0)) # log to reduce errors
#        
#        if a is not None:
#            if np.size(a)==1 and np.size(b)==1:
#                #wavefunction = np.exp(a*spins)*np.prod(np.cosh(theta)) 
#                wavefunction= prodcosh*np.exp(np.matmul(a*spins))
#            else:
#                #wavefunction = np.exp(np.matmul(a,spins))*np.prod(np.cosh(theta)) 
#                wavefunction = prodcosh*np.exp(np.matmul(a,spins))
#        else: 
#            #wavefunction = np.prod(np.cosh(theta),0) 
#            wavefunction = prodcosh
#    
#    #### JASTROW ALG        
#    elif machine =='Jastrow':       
#        sigma_sum=0
#        for j in range(N):
#            i=0
#            while i<j:
#                sigma_sum =sigma_sum + W[i,j]*spins[i]*spins[j]
#                i=i+1
#        wavefunction = np.exp(sigma_sum)
#    
#    # wavefunction/np.linalg.norm(wavefunction)
#    return wavefunction 

def Metropolis_Hastings(chain_size, W, a=None,b=None, s0 =None, Size=None,
                        machine = 'RbmSpin'):
    if s0 is None:
        if Size is None: 
            Size = np.shape(W)[0]
        
        s0 = np.random.randint(2, size=Size) # creates Size binary rand [0,1...]
        s0[s0==0]=-1.0 # rewrites s0=0 spots as -1
    else:
        Size = np.size(s0)
    
    MC_chain = np.zeros([chain_size,Size])
    MC_chain[0,:] = s0
    for n in range(chain_size-1):
        pos = np.random.randint(Size) # position to change
        
        alt_state = MC_chain[n,:].copy() # next potential state
        alt_state[pos] = -1*alt_state[pos] # flip next random position for spin
        
        # Probabilty of the next state divided by the current
        prob = (np.square(np.abs(ppsi(W, alt_state, machine,a,b))))   \
        /(np.square(np.abs(ppsi(W, MC_chain[n,:], machine,a,b))))
        #print(prob)
        A = min(1,prob) # Metropolis Hastings acceptance formula
        
        if A ==1: MC_chain[n+1,:]=alt_state
        else: 
            if np.random.rand()<A: MC_chain[n+1,:]=alt_state # accepting move with prob
            else: MC_chain[n+1,:] = MC_chain[n,:]
            
    return MC_chain

#def Metropolis_Hastings_MPS(chain_size, psi,rot,evals,s0=None):
#    
#    Size = np.size(psi)
#
#    if s0 is None:
#        s0 = np.random.choice(evals,size=Size) # creates Size binary rand [0,1...]
#    
#    MC_chain = np.zeros([chain_size,Size],complex)
#    MC_chain[0,:] = s0
#    for n in range(chain_size-1):
#        pos = np.random.randint(Size) # position to change
#        
#        alt_state = MC_chain[n,:].copy() # next potential state
#        
#        if np.random.rand()>=0.5:
#            alt_state[pos] = np.exp(1j*rot)*alt_state[pos] # flip next random position for spin
#        else:
#            alt_state[pos] = np.exp(-1j*rot)*alt_state[pos] # same chance to flip other direction
#        
#        # Probabilty of the next state divided by the current
#        prob = (np.square(np.abs(MPS_psi(psi, alt_state, evals))))   \
#        /(np.square(np.abs(MPS_psi(psi, MC_chain[n,:], evals))))
#        #print(prob)
#        A = min(1,prob) # Metropolis Hastings acceptance formula
#        
#        if A ==1: MC_chain[n+1,:]=alt_state
#        else: 
#            if np.random.rand()<A: MC_chain[n+1,:]=alt_state # accepting move with prob
#            else: MC_chain[n+1,:] = MC_chain[n,:]
#            
#    return MC_chain
#
#def MPS_psi(psi,s,evals,tol=1e-4):
#
#    d=np.size(evals)
#    
#    coef=1
#    for ii in range(np.size(psi)):
#        for jj in range(d):
#            if abs(evals[jj]-s[ii])<tol:
#                eval_slice=jj
#                
#        A=psi[ii][:,eval_slice,:]
#            
#        if ii==0:
#            coef=A   
#        else:
#            coef=np.matmul(coef,A)
#        
#    return coef

