#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:04:45 2020

@author: alex
"""

import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''################## Class and function definitions, can skip #############'''

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
        
        out = self.real_comp(s).flatten()
        
        # should be the simpler form to apply dln(Psi)/dw_i
        if self.form.lower()=='euler':
            # each form has a slightly different multiplication form
            multiplier = 2*np.real((E_loc-E)/(out.detach().numpy()))   
            
        elif self.form.lower()=='vector':
            if np.all(self.complex==0):
                self.complex_out(self,s) # define self.complex
            
            multiplier = 2*np.real((E_loc-E)/self.complex.flatten())
            
        self.real_comp.zero_grad()
        out.backward(torch.tensor(multiplier)*out)
        # calling this applies autograd to only tensor .grad object i.e. out
        # which corresponds to dpsi_real(s)/dpars. 
        
        params=list(self.real_comp.parameters()) # get the parameters
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad # apply the Energy gradient descent
            
        ## Now do the same for the imaginary network. note, they are not done 
        # in parallel here as the networks and number of parameters therein can 
        # vary arbitrarily, so the for loop has to be over each ANN separately
        # this way is also less memory intensive as out, mult, etc. are overwritten
        out = self.imag_comp(s).flatten()

        # complex versions of the multiplier 
        if self.form.lower()=='euler':
            multiplier = 2*np.real((E_loc-E)*1j)
            # note, to accelerate, Energy E could not appear here because it 
            # should be real. Where it is not are sampling/accuracy errors
            
        elif self.form.lower()=='vector':
            # self.complex should have been defined in previous section of method
            multiplier = 2*np.real(1j*(E_loc-E)/self.complex.flatten())
            
        self.imag_comp.zero_grad()
        out.backward(torch.tensor(multiplier)*out)
        
        params=list(self.imag_comp.parameters()) # get the parameters
        with torch.no_grad():
            for param in params:
                param -= lr*param.grad
        
            
        return 

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
            
            pos=np.random.randint(L) # position to change
            
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

def O_local(operator,s, psi, multiplier=None): # potential improvement to use all tensor funcs so
    
    if not np.all(np.conjugate(np.transpose(operator.matrix))==operator.matrix):
        raise Warning('Operator matrix ', operator.matrix, 'is not Hermitian,'\
                      ' Observable may be non-real and unphysical')
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

        ## Generating all possible permutations of the local spins
        perms=np.array(list(itertools.product(evals,repeat=op_span)))
        
        # do a loop over all of the possible permutations
        for kk in range(xformed_state.shape[1]):
            
            # change the local spins in s' for each config
            s_prime[:,sites[i]]=perms[-(kk+1)]
            # -(kk+1) is used because the ordering is opposite of how the 
            # slicing is organized. Ex, -1,-1 corresponds to the last 
            # slice (0,0,0,1) and 1,1 to the first (1,0,0,0) with the 
            # 1 state = (1,0) and -1 state = (0,1) convention.
            
            with torch.no_grad():
                O_loc[:,i]+= xformed_state[:,kk]* \
            (psi.complex_out(torch.tensor(s_prime,dtype=torch.float))  \
        /psi.complex_out(torch.tensor(s,dtype=torch.float))).flatten()
            # each slice of the transformed state acts as a multiplier to 
            # its respective local spin configuration state
                
    return O_loc

'''############### End of Class and function definitions ###################'''

'''
entry method inspired by NetKet-1.0.0
'''

# Define operators to use in the Hamiltonian
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
#sigmay = (1/1j)*np.array([[0,1],[-1,0]])
# imaginary numbers aren't really supported in the code yet

#matricks=np.array([[2,3],[3,4]]) # for the obs to be physical, it must be Hermitian

#rando_mat=np.kron(matricks,2*matricks) 

#szsz = np.kron(0.5*sigmax, 0.5*sigmax)
#szsz = np.kron(np.kron(np.kron(sigmax, sigmax),sigmax),sigmax)

# initiate the operators and the matrix they are fed
b=0.5
#nn_interaction=Op(szsz)
b_field=Op(b*sigmax)
#rando_op=Op(rando_mat)
#play_op=Op(matricks)

L = 20
for i in range(L):  # Specify the sites upon which the operators act
    # specify the arbitrary sites which the operators will act on
    b_field.add_site([i])
    
    # for n body interactions, n sites must be added
    #nn_interaction.add_site([i, (i + 1) % L,(i+2)%L,(i+3)%L]) 
#    nn_interaction.add_site([i,(i+1)%L])
    
#    play_op.add_site([i])
#    rando_op.add_site([i,(i+1)%L])

# Make sure to re-instanciate the Op class instances otherwise the site lists will keep appending

'''##################### SIMPLE FFNN DEFINITION ###############################
Here is an example of how to use this function with a Pytorch neural network
(here a simple FFNN created with the class Sequential)
'''

L=20
H=40 # hidden layer size
N_samples=100

s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)

# creates an instance of the Sequential class nn.Sigmoid etc usually in forward section
toy_model=nn.Sequential(nn.Linear(L,H),
                        nn.Sigmoid(),
                        nn.Linear(H,1),
                        nn.Sigmoid()) # relu can be an important ending func as it
             #is always positive for which is necessary for applying log later

H2=20
toy_model2=nn.Sequential(nn.Linear(L,H2),
                        nn.Sigmoid(),
                        nn.Linear(H2,1),
                        nn.Sigmoid()) 

# Test complex wavefunction object construction with modulus and angle
ppsi_mod=Psi(toy_model,toy_model2, L, form='euler')

z1=ppsi_mod.complex_out(s)

# or try the network as a simple z=a+bi complex vector representation
ppsi_vec=Psi(toy_model,toy_model2, L, form='vector')

z2=ppsi_vec.complex_out(s)

print('Complex Psi with Euler representation: \n', z1 , '\n\n and with vector rep: \n', z2)

''' Now let's calculate the loss/cost function (energy) using the O_local function '''

#H_nn=O_local(nn_interaction,s.numpy(),ppsi_mod)
H_b=O_local(b_field,s.numpy(),ppsi_mod)

# Calculate the total energy (the sum of these operator expectation val terms)
energy_per_sample = np.sum(H_b,axis=1)
energy=np.mean(energy_per_sample)
print('Total energy is: ', energy)

#with torch.no_grad(): # important to use no_grad here so incorrect grad fns aren't propagated
#    energy_per_sample = torch.sum(H_nn+H_b,axis=1)

'''     Applying the gradient of E to each param, simple gradient descent   '''

ppsi_mod.apply_energy_gradient(s,energy_per_sample,energy)
ppsi_vec.apply_energy_gradient(s,energy_per_sample,energy)
# this function applies the energy gradient to both the real and imaginary
# networks defining psi. works for both forms, euler: a*e^bi and vector: a+bi
 
''' Now let's test the Metropolis-Hastings sampling method in the class Psi '''

N_samples=100
samples=ppsi_mod.sample_MH(N_samples,spin=0.5)


''' Ok, let's see how the opimization goes '''

# Enter simulation hyper parameters
N_iter=100
N_samples=1000
burn_in=200
lr=0.03

# make an initial s
s=np.random.randint(-1,high=1,size=[N_samples,L]); s[s==0]=1; 
s=torch.tensor(s,dtype=torch.float)

energy_n=np.zeros([N_iter,1])
for n in range(N_iter):
    
    # Get the energy at each iteration
#    H_nn=O_local(nn_interaction,s.numpy(),ppsi_mod)
    H_b=O_local(b_field,s.numpy(),ppsi_mod)
    energy_per_sample = np.sum(H_b,axis=1)
    energy=np.mean(energy_per_sample)
    energy_n[n]=np.real(energy)

    # apply the energy gradient, updates pars in Psi object
    ppsi_mod.apply_energy_gradient(s,energy_per_sample,energy,lr)

    # before doing the actual sampling, we should do a burn in
    sburn=ppsi_mod.sample_MH(burn_in,spin=0.5)

    # Now we sample from the state and recast this as the new s, s0 so burn in is used
    s=torch.tensor(ppsi_mod.sample_MH(N_samples,spin=0.5,s0=sburn[-1,:]),dtype=torch.float)

    if n%10==0:
        print('percentage of iterations complete: ', (n/N_iter)*100)

plt.figure
plt.plot(range(N_iter),energy_n)
plt.xlabel('Iteration number')
plt.ylabel('Energy')









'''   Could also try to use the energy for each sample as a loss function 
                    (without the explicit energy gradient)             '''

#E_real=torch.tensor(np.real(energy_per_sample), dtype=torch.float32)
## should this be np.abs if form is euler, np.real if vector?
#
#E_imag=torch.tensor(np.imag(energy_per_sample), dtype=torch.float32)
## np.abs if form is euler, np.real if vector?
#
#            ##### Applying to the real/magnitude network #####
#out=toy_model(s).flatten() 
#toy_model.zero_grad()
#out.backward(E_real) 
#
#params=list(toy_model.parameters()) # record the parameters
#with torch.no_grad():
#    for param in params:
#        param -= 0.1 * param.grad
#        
#            ##### Applying to the imag/angle network ######
#out=toy_model2(s).flatten() 
#toy_model2.zero_grad()
#out.backward(E_imag) 
#
#params=list(toy_model.parameters()) # record the parameters
#with torch.no_grad():
#    for param in params:
#        param -= 0.1 * param.grad


'''                 First simple attempted approach,
 this method is inccorrect, it does dln(Ann_w1)/dw1 not dln(Psi(w1,w2))/dw1 '''

#            ##### Applying to the real/magnitude network #####
#out=torch.log(ppsi_mod.real_comp(s)).flatten()
#ppsi_mod.real_comp.zero_grad()
#out.backward(2*E_real*out-2*np.real(energy)*out) 
## seems weird, but Idk how else it'd be done. should be 2*Re(<E_loc*dpsi>-<E><dpsi>)
#
#params=list(ppsi_mod.real_comp.parameters()) # record the parameters
#with torch.no_grad():
#    for param in params:
#        param -= 0.1 * param.grad # modify to be (E_loc-E)*ln(param).grad
#
#            ##### Applying to the imag/angle network ####
#out=torch.log(ppsi_mod.imag_comp(s)).flatten()
#ppsi_mod.imag_comp.zero_grad()
#out.backward(2*E_real*out-2*np.real(energy)*out) 
## seems weird, but Idk how else it'd be done. should be 2*Re(<E_loc*dpsi>-<E><dpsi>)
#
#params=list(ppsi_mod.imag_comp.parameters()) # record the parameters
#with torch.no_grad():
#    for param in params:
#        param -= 0.1 * param.grad      

'''
This is a previously working O_local that calculated the input operator O_loc 
for a single sample at a time. The function was generalized to process a batch 
of samples.
'''
#def O_local(operator,s, psi):  
#    
#    sites=operator.sites.copy()
#    
#    [n_sites,op_span]= np.shape(sites) # get lattice list length and operator span
#                                        # former is often equal to L (lat size)
#        
#    # For now, assuming we are working in the Sz basis with spin=1/2
#    spin=0.5 # presumambly will be entered by the user in later versions
#    dim = int(2*spin+1)
#    # sz_tilde=np.zeros([L,dim**op_span])
#    
#    op_size=np.log((np.shape(operator.matrix)[0]))/np.log(dim)
#    if not op_size==op_span:
#        raise ValueError('Operator size ', op_size, ' does not match the number' \
#                         ' of sites entered ', op_span, 'to be acted upon')
#    
#    O_loc=0
#    # cycle through the sites and apply each operator to state s_i (x s_i+1...)
#    for i in range(n_sites):
#        
#        s_prime=s.copy() # so it's a fresh copy each loop
#        
#        # Set up spin config representation in Sz basis for just the acted upon spins
#        sz_basis=np.zeros([op_span,dim])
#        s_loc=s[sites[i]]
#        sz_basis[np.where(s_loc==1),...]=np.array([1,0]) # need to generalize for arbitrary spin
#        sz_basis[np.where(s_loc==-1),...]=np.array([0,1])
#        
#        if op_span>1: # extend the size of the basis for multi-site operators
#            basis=sz_basis[0]
#            for j in range(1,op_span): 
#                basis=np.kron(basis,sz_basis[j]) 
#                # cycle through the states acted on by the multi-site operator
#        else:
#            basis=sz_basis[0]
#            
#        # S[sites[i]] transformed by Op still in extended basis
#        xformed_state=np.squeeze(np.matmul(basis,operator.matrix)) 
#        
#        if np.all(xformed_state==basis): # can skip changing s', nothing's changed
#            pass
#        else:
#                        
#            ''' 
#            Decomposing the kronicker product back into the alternate s'
#            requires recursively splitting the vector in 1/dim, if the 1 is in  
#            the top (bottom) section, the first kronickered index is 1 (0). Keep 
#            splitting the vector untill all indices have been reassigned and a 
#            vec of size dim is left - which will be the very last site index.
#            I assume the states s are ordered where the state with eval spin is 
#            (1,0...,0), that with (0,1,0...,0) is spin-1, etc. 
#            '''
#            spl=xformed_state.copy()
#            alt_ind=[]
#            while np.max(np.shape(spl))>dim: # need to keep splitting
#            
##            for nn in range(len(xformed_state)/dim):
#                spl=np.split(spl,dim)
#                for kk in range(dim):
#                    if any(spl[kk]):                        
#                        # depending how you set up spin, may need to be changed
#                        alt_ind.append(2*(spin-kk))
#                        # new split section to split again
#                        spl=spl[kk]
#                        break
#            
#            # exiting while loop when = dim, this is the last index
#            alt_ind.append(2*(spin-np.squeeze(np.where(spl==1))))
#                    
#        # now we have identified s'!!
#        # Each local op will only effect op_span number of sites in configuration s
#        s_prime[sites[i]]=alt_ind
#        
#        # Place-holder for wavefunction psi calculation that must be summed over
#        O_loc+=psi(torch.tensor(s_prime))
#        
#    return O_loc


''' ########################## NORMALIZATION ###############################'''

# The simplest method to normalize would just to divide by the sum of the output
#out=out/out.sum() # e.x. for a given 

# actually, I shouldn't need normalization if it's <psi|H|psi>/<psi|psi>
