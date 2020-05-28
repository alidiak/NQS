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
real_net=nn.Sequential(nn.Linear(L,1))#, nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 
#real_net=nn.Sequential(nn.Linear(L,H), nn.Sigmoid(), nn.Linear(H,1),nn.Sigmoid()) 

H2=2*L
imag_net=nn.Sequential(nn.Linear(L,1))#,nn.Sigmoid(),nn.Linear(H2,1),nn.Sigmoid()) 
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

'''                 '''

def O_local(operator,s, psi): # potential improvement to use all tensor funcs so
    
    sites=operator.sites.copy()
    
    [n_sites,op_span]= np.shape(sites) # get lattice list length and operator span
                                        # former is often equal to L (lat size) if applied to all sites
    [N_samples,L]=np.shape(s)  # Get the number of samples and lattice size from samples
        
    # For now, assuming we are working in the Sz basis with spin=1/2
    spin=0.5 # presumambly will be entered by the user in later versions
    dim = int(2*spin+1)
    evals=2*np.arange(-spin,spin+1)
    # multiplied by 2 simply because integers are easier to work with
    
    op_size=np.log((np.shape(operator.matrix)[0]))/np.log(dim)
    if not op_size==op_span:
        raise ValueError('Operator size ', op_size, ' does not match the number' \
                         ' of sites entered ', op_span, 'to be acted upon')
    
    O_loc=np.zeros([N_samples,L],dtype=complex) 
    #this construction allows us to get local expectation vals
    # and the energy for each sample (which we can use to backprop)
    
    # cycle through the sites and apply each operator to state s_i (x s_i+1...)
    for i in range(n_sites):
        
        s_prime=s.copy() # so it's a fresh copy each loop
        
        # Set up spin config representation in Sz basis for just the acted upon spins
        # need to generalize for arbitrary spin
        sz_basis=np.zeros([N_samples,op_span,dim])
        s_loc=s[:,sites[i]]
        # can iterate over this for spin!=0.5. use evals
        sz_basis[np.where(s_loc==1)[0],np.where(s_loc==1)[1],:]=np.array([1,0]) 
        sz_basis[np.where(s_loc==-1)[0],np.where(s_loc==-1)[1],:]=np.array([0,1])
        
        if op_span>1: # extend the size of the basis for multi-site operators
            basis=sz_basis[:,0,:]
            for j in range(1,op_span): 
                basis=np.einsum('nk,nl->nkl',basis,sz_basis[:,j,:]).reshape(basis.shape[0],-1)
                # einstein summation func, forces kron product over 2nd axis by 
                # by taking matching inputs from the nth col, avoids kron over samples
                # cycle through the states acted on by the multi-site operator
        else:
            basis=sz_basis[:,0,:]
            
        # S[sites[i]] transformed by Op still in extended basis
        # Should not matter which side the operator acts on as Op must be hermitian
        # as we act on the left here, 
        xformed_state=np.squeeze(np.matmul(basis,operator.matrix)) 
  
        # just so the alg can handle single sample input. adds a singleton dim
        if len(xformed_state.shape)==1:
            xformed_state=xformed_state[None,:]

        ## Generating all possible permutations of the local spins
        perms=np.array(list(itertools.product(evals,repeat=op_span)))
        
        # do a loop over all of the possible permutations
        for kk in range(len(perms)): # xformed_state.shape[1]
            
            # change the local spins in s' for each config
            s_prime[:,sites[i]]=perms[(kk)]
            print(perms[-(kk+1)])
            # -(kk+1) is used because the ordering is opposite of how the 
            # slicing is organized. Ex, -1,-1 corresponds to the last 
            # slice (0,0,0,1) and 1,1 to the first (1,0,0,0) with the 
            # 1 state = (1,0) and -1 state = (0,1) convention.
            
            with torch.no_grad():
                log_psi_diff=np.log(psi.complex_out(torch.tensor(s_prime,\
                dtype=torch.float)).flatten())-np.log(psi.complex_out(\
                torch.tensor(s,dtype=torch.float))).flatten()
                O_loc[:,i]+= xformed_state[:,kk]*np.exp(log_psi_diff)
            # each slice of the transformed state acts as a multiplier to 
            # its respective local spin configuration state
                
    return O_loc
