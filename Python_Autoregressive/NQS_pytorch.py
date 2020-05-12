#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:26:01 2020

Here we keep classes and functions for the Pytorch Quantum States library.


@author: alex
"""

import itertools
import numpy as np
import torch
import copy

class Op:
    
    def __init__(self, matrix):
        self.matrix=matrix
        self.sites=[]
        
    def add_site(self,new_site):
        self.sites.append(new_site)
        
    # I could potentially add the O_loc function here as a method to Op
        
        return

'''###################### Complex Psi ######################################'''

class Psi:
    
    def __init__(self,real_comp,imag_comp,L,form=None):
        # options for form are 'euler' or 'vector' - corresponding to 2 forms 
        # of complex number notation
        self.real_comp=real_comp
        self.imag_comp=imag_comp
        self.complex=0
        self.L=L
        self.samples=0
        if form==None: # setting the default form as magnitude/angle(phase)
            self.form='euler'
        else:
            self.form=form
        
    # Method to return the complex number specified by the state of the 
    # 2 ANNs real_comp and imag_comp and an input state s
    def complex_out(self, s):
        self.complex=np.zeros(s.size(0),dtype=complex) # complex number for each sample
        if self.form.lower()=='euler':
            self.complex=self.real_comp(s).detach().numpy()*    \
            np.exp(1j*self.imag_comp(s).detach().numpy())
        elif self.form.lower()=='vector':
            self.complex=self.real_comp(s).detach().numpy()+    \
            1j*self.imag_comp(s).detach().numpy()
        else:
            raise Warning('Specified form', self.form, ' for complex number is'\
            ' ambiguous, use either "euler": real_comp*e^(i*imag_comp) or vector":'\
            ' "real_comp+1j*imag_comp. This output was calculated using "euler".')
            self.form='euler'
            self.complex=self.real_comp(s).detach().numpy()*    \
            np.exp(1j*self.imag_comp(s).detach().numpy())
        return self.complex

    '''##################### Energy Gradient ############################'''
    ''' This method will apply the energy gradient to each ANN network param for 
    a given form of Psi. It does simple gradient descent (no SR or anything).
    It does so given an E_local, Energy E, and wavefunc Psi over sample set s.'''

    def apply_energy_gradient(self,s,E_loc,E, lr=0.03): # add Pytorch optimizer) (fixed lr for now)
        
        outr = self.real_comp(s).flatten()
        outi = self.imag_comp(s).flatten()
        
        E=np.conj(E)
        E_loc=np.conj(E_loc)
        diff=(E_loc-E)
        mult=torch.tensor(np.real(2*diff),dtype=torch.float)
        
        self.real_comp.zero_grad()
        self.imag_comp.zero_grad()
        # should be the simpler form to apply dln(Psi)/dw_i
        if self.form.lower()=='euler':
            # each form has a slightly different multiplication form
            (outr.log()*mult).mean().backward()
            # calling this applies autograd to tensor .grad object i.e. out*mult
            # which corresponds to dpsi_real(s)/dpars. 
            # must include the averaging over # samples factor myself 
            
            # Angle
            multiplier = 2*np.imag(-E_loc)
            multiplier=torch.tensor(multiplier,dtype=torch.float)
            (multiplier*outi).mean().backward()
            
        elif self.form.lower()=='vector':
            if np.all(self.complex==0):
                self.complex_out(self,s) # define self.complex
            
            N_samples=s.size(0)
            
#            psi0=self.complex.flatten() # the original psi
            p_r=list(self.real_comp.parameters())
            p_i=list(self.imag_comp.parameters())   
            
            grad_list_r=copy.deepcopy(p_r)
            grad_list_i=copy.deepcopy(p_i)
            with torch.no_grad():
                for param in grad_list_r:
                    param.copy_(torch.zeros_like(param))
                    param.requires_grad=False
                for param in grad_list_i:
                    param.copy_(torch.zeros_like(param))
                    param.requires_grad=False

            # what we calculated the gradients should be
            for n in range(N_samples):
                
                self.real_comp.zero_grad() # important so derivatives aren't summed
                self.imag_comp.zero_grad()    
                outr[n].backward(retain_graph=True) # retain so that buffers aren't cleared 
                                                    # and it can be applied again
                outi[n].backward(retain_graph=True)
                                                    
                with torch.no_grad():        
                    m= ((E_loc[n]-E)/self.complex[n]) 
                    # [E_l*-E]/Psi according to derivative
                    m_r=torch.tensor(2*np.real(m) ,dtype=torch.float)
                    m_i=torch.tensor(2*np.real(1j*m) ,dtype=torch.float)
                    
                for kk in range(len(p_r)):
                    with torch.no_grad():
                        grad_list_r[kk]+=(p_r[kk].grad)*(m_r/N_samples)
                for kk in range(len(p_r)):
                    with torch.no_grad():
                        grad_list_i[kk]+=(p_i[kk].grad)*(m_i/N_samples)
            
            # manually do the mean
            for kk in range(len(p_r)):
                p_r[kk].grad=grad_list_r[kk]
            for kk in range(len(p_i)):
                p_i[kk].grad=grad_list_i[kk]            
                
        params=list(self.real_comp.parameters()) # get the parameters
        
        # for testing purposes
        pr1_grad=params[0].grad
        
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad # apply the Energy gradient descent
    
        ## Now do the same for the imaginary network. note, they are not done 
        # in parallel here as the networks and number of parameters therein can 
        # vary arbitrarily, so the for loop has to be over each ANN separately.
        # Could also make this way less memory intensive as out, mult, etc. 
        # can easily be overwritten for the angle/magnitude network
                
        params=list(self.imag_comp.parameters()) # get the parameters
        
        pi1_grad=params[0].grad
        
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad
            
        return pr1_grad, pi1_grad

    '''###################### Sampling function ################################'''
    def sample_MH(self, N_samples, spin=None, evals=None, s0=None, rot=None):
        # need either the explicit evals or the spin 
        if spin is None and evals is None:
            raise ValueError('Either the eigenvalues of the system or the spin\
                             must be entered')
                
        # the rule for flipping/rotating a spin between it's eigenvalues
        if rot is None:
            if spin is None:
                dim=len(evals)
            else:
                dim = int(2*spin+1)
            rot =  2*np.pi/dim # assume a rotation that scales with # evals
            # note, can only rotate to 'intermediate/nearby' evals
    
        if evals is None:
            evals=2*np.arange(-spin,spin+1) # +1 is just so s=spin is included
            # times 2 is just the convention that's been used, spin evals of -1,1
        
        if s0 is None:
            s0=np.random.choice(evals,size=self.L)
        
        self.samples=np.zeros([N_samples,self.L])
        self.samples[0,:]=s0
        for n in range(N_samples-1):
            
            pos=np.random.randint(self.L) # position to change
            
            alt_state = self.samples[n,:].copy() # next potential state
            
            if np.random.rand()>=0.5:
                alt_state[pos] = np.real(np.exp(1j*rot)*alt_state[pos]) # flip next random position for spin
            else:
                alt_state[pos] = np.real(np.exp(-1j*rot)*alt_state[pos]) # same chance to flip other direction
            # Will have to generalize to complex vals sometime
            
            # Probabilty of the next state divided by the current
            prob = (np.square(np.abs(self.complex_out(torch.tensor(alt_state,dtype=torch.float32)))))   \
            /(np.square(np.abs(self.complex_out(torch.tensor(self.samples[n,:],dtype=torch.float32)))))
            
            A = min(1,prob) # Metropolis Hastings acceptance formula

            if A ==1: self.samples[n+1,:]=alt_state
            else: 
                if np.random.rand()<A: self.samples[n+1,:]=alt_state # accepting move with prob
                else: self.samples[n+1,:] = self.samples[n,:]
            
        return self.samples
    
'''############################ O_local #######################################
Now find O_local where O is an arbitrary operator acting on sites entered. This 
function returns the O_local operator summed over the 'allowed' transitions 
between the given input spin s and any non-zero transition to spin config s'. 
This operator also depends upon the current wavefunction psi. Psi is a Neural 
Network object that itself is fed s. 
'''

def O_local(operator,s, psi): # potential improvement to use all tensor funcs so
    
    # Testing if it is a Hamiltonian object
#    if hasattr(operator,'Op_list'):
#        N_ops=len(operator.Op_list)
#    else:
#        N_ops=1
#        
    #if not np.all(np.conjugate(np.transpose(operator.matrix))==operator.matrix):
    #    raise Warning('Operator matrix ', operator.matrix, 'is not Hermitian,'\
    #                  ' Observable may be non-real and unphysical')
                 # using CUDA devices could potentially accelerate this func?
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
        
#        if np.all((basis>0)==(np.abs(xformed_state)>0)):
#            # can skip changing s', nothing's changed in the basis
#            basis[basis==0]=1 # just doing this so that there's no division by 0
#            div=xformed_state/basis
#            multiplier=div[div>0] 
#            # returns all of the differences in magnitude of the otherwise unchanged basis state
#            pass
#        else:
                        
        ''' 
        Decomposing the kronicker product back into the alternate s'
        requires recursively splitting the vector in 1/dim, if the 1 is in  
        the top (bottom) section, the first kronickered index is 1 (0). Keep 
        splitting the vector untill all indices have been reassigned and a 
        vec of size dim is left - which will be the very last site index.
        I assume the states s are ordered where the state with eval spin is 
        (1,0...,0), that with (0,1,0...,0) is spin-1, etc. 
        '''

        # just so the alg can handle single sample input. adds a singleton dim
        if len(xformed_state.shape)==1:
            xformed_state=xformed_state[None,:]

        ## Generating all possible permutations of the local spins
        perms=np.array(list(itertools.product(evals,repeat=op_span)))
        
        # do a loop over all of the possible permutations
        for kk in range(len(perms)): # xformed_state.shape[1]
            
            # change the local spins in s' for each config
            s_prime[:,sites[i]]=perms[-(kk+1)]
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
